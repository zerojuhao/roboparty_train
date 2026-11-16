"""MBFB residual dynamics and physical replay, adapted to RP1's xyzw state."""

import torch
from collections.abc import Sequence
from torch import nn
from torch.nn import functional as F

from .world_model import quat_apply, quat_conjugate, quat_from_rotvec, quat_mul, rotvec_from_quat, yaw_quat


class MBFBDynamics(nn.Module):
    """Full position, orientation, world velocities, joints and executed last action."""

    state_dim = 85

    def __init__(self, context_dim: int, hidden_dims: Sequence[int], dt: float, *, use_bf16_amp: bool = True) -> None:
        super().__init__()
        self.dt = dt
        if type(use_bf16_amp) is not bool:
            raise ValueError("Dynamics use_bf16_amp must be a boolean")
        self.use_bf16_amp = use_bf16_amp
        layers, width = [], 109 + context_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            width = hidden
        layers.append(nn.Linear(width, 60))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("increment_var", torch.ones(60))
        self.register_buffer(
            "error_floor", torch.tensor([3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3 + [3e-3] * 24 + [5e-2] * 24)
        )

    @staticmethod
    def split(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split position, xyzw rotation, velocities, joints and last action."""
        return state.split((3, 4, 3, 3, 24, 24, 24), -1)

    def features(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Build heading-invariant inputs for the residual dynamics network."""
        pos, quat, vel, ang, joints, rates, last = self.split(state)
        heading = yaw_quat(quat)
        tilt = quat_mul(quat_conjugate(heading), quat)
        x, z = torch.zeros_like(pos), torch.zeros_like(pos)
        x[:, 0], z[:, 2] = 1, 1
        return torch.cat(
            (
                pos[:, 2:3],
                quat_apply(tilt, x),
                quat_apply(tilt, z),
                0.5 * quat_apply(quat_conjugate(heading), vel),
                0.25 * quat_apply(quat_conjugate(quat), ang),
                joints,
                0.1 * rates,
                0.2 * last,
                0.2 * action,
                context,
            ),
            -1,
        )

    def integrate(self, state: torch.Tensor, delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply learned residuals to a constant-velocity state transition."""
        pos, quat, vel, ang, joints, rates, _ = self.split(state)
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 24, 24), -1)
        heading = yaw_quat(quat)
        rotation = self.dt * ang + quat_apply(quat, 0.1 * dr)
        return torch.cat(
            (
                pos + self.dt * vel + quat_apply(heading, 0.03 * dp),
                F.normalize(quat_mul(quat_from_rotvec(rotation), quat), dim=-1),
                vel + quat_apply(heading, 0.5 * dv),
                ang + quat_apply(quat, dw),
                joints + self.dt * rates + 0.05 * dq,
                rates + 2 * dqd,
                action,
            ),
            -1,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Predict the next state with configurable network precision and FP32 integration."""
        with torch.autocast(state.device.type, dtype=torch.bfloat16,
                            enabled=state.is_cuda and self.use_bf16_amp):
            delta = self.net(self.features(state, action, context))
        return self.integrate(state, delta.float(), action)

    def constant_velocity(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the validation baseline without learned residuals."""
        return self.integrate(state, state.new_zeros(state.shape[0], 60), action)

    def local_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Measure physical-state errors in the target's local frames."""
        p, q, v, w, jq, jv, _ = self.split(target)
        ph, qh, vh, wh, jqh, jvh, _ = self.split(prediction)
        inverse = quat_conjugate(q)
        return torch.cat(
            (
                quat_apply(quat_conjugate(yaw_quat(q)), ph - p),
                rotvec_from_quat(quat_mul(inverse, qh)),
                quat_apply(inverse, vh - v),
                quat_apply(inverse, wh - w),
                jqh - jq,
                jvh - jv,
            ),
            -1,
        )

    @torch.no_grad()
    def observe_increments(self, state: torch.Tensor, target: torch.Tensor) -> None:
        """Update error scales from real transitions across all ranks."""
        variance = self.local_error(target, state).square().mean(0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(variance)
            variance.div_(torch.distributed.get_world_size())
        self.increment_var.lerp_(variance, 0.01)

    def normalized_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean squared, scale-normalized error for each state."""
        scale = torch.maximum(self.increment_var.sqrt(), self.error_floor)
        return (self.local_error(prediction, target) / scale).square().mean(-1)


class PhysicalReplay:
    """GPU ring; a reset or external push restarts the next autoregressive step."""

    def __init__(self, capacity: int, num_envs: int, context_dim: int, device: torch.device) -> None:
        if num_envs < 2:
            raise ValueError("MBFB replay requires separate training and validation environments")
        self.capacity, self.size, self.ptr = capacity, 0, 0
        widths = dict(state=85, next_state=85, action=24, context=context_dim, restart=1)
        self.data = {
            key: torch.empty(
                capacity, num_envs, width, device=device, dtype=torch.bool if key == "restart" else torch.float32
            )
            for key, width in widths.items()
        }
        lanes = torch.arange(num_envs, device=device)
        self.train_envs, self.val_envs = lanes[lanes % 16 != 0], lanes[lanes % 16 == 0]

    @torch.no_grad()
    def add(self, **data: torch.Tensor) -> None:
        """Append one real transition per environment to the ring buffer."""
        for key, value in data.items():
            self.data[key][self.ptr].copy_(value)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, horizon: int, *, validation: bool = False) -> dict[str, torch.Tensor]:
        """Sample contiguous windows from disjoint training or validation environments."""
        if self.size < horizon:
            raise ValueError("Insufficient MBFB physical replay")
        lanes = self.val_envs if validation else self.train_envs
        env = lanes[torch.randint(len(lanes), (count,), device=lanes.device)]
        start = torch.randint(self.size - horizon + 1, (count,), device=lanes.device)
        oldest = self.ptr if self.size == self.capacity else 0
        rows = (oldest + start[:, None] + torch.arange(horizon, device=lanes.device)) % self.capacity
        batch = {key: value[rows, env[:, None]] for key, value in self.data.items()}
        batch["continues"] = torch.cat(
            (torch.ones(count, 1, dtype=torch.bool, device=lanes.device), ~batch["restart"][:, :-1, 0]), 1
        )
        return batch


def rollout_error(dynamics: MBFBDynamics, batch: dict[str, torch.Tensor], *, baseline: bool = False) -> torch.Tensor:
    """Evaluate autoregressive windows, restarting at resets and external pushes."""
    state, errors = batch["state"][:, 0], []
    for step in range(batch["action"].shape[1]):
        state = torch.where(batch["continues"][:, step, None], state, batch["state"][:, step])
        action = batch["action"][:, step]
        state = (
            dynamics.constant_velocity(state, action)
            if baseline
            else dynamics(state, action, batch["context"][:, step])
        )
        errors.append(dynamics.normalized_error(state, batch["next_state"][:, step]))
    return torch.stack(errors, 1)
