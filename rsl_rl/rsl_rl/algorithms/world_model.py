"""Real-replay dynamics and differentiable imagination for UFO.

No learned termination: timeout ends replay windows, never the value bootstrap.
The existing FB representations and critics consume only simulator transitions.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules if module is not None for p in module.parameters()]
    previous = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in zip(parameters, previous):
            p.requires_grad_(enabled)


def mlp(input_dim, hidden_dims, output_dim, *, zero_output=False):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    if zero_output:
        nn.init.zeros_(layers[-1].weight)
    else:
        nn.init.normal_(layers[-1].weight, std=1e-3)
    nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std used for world-model inputs and physical increments."""

    def __init__(self, dim, *, eps=1e-2, max_count=2e8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = float(eps)
        self.max_count = float(max_count)

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, values.shape[-1]).float()
        if values.numel() == 0 or self.count >= self.max_count:
            return
        count = values.new_tensor([float(values.shape[0])])
        stats = torch.cat((count, values.sum(0), values.square().sum(0)))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        count = stats[:1].squeeze(0)
        mean = stats[1:1 + values.shape[-1]] / count
        var = (stats[1 + values.shape[-1]:] / count - mean.square()).clamp_min(0.0)
        total = self.count + count
        delta = mean - self.mean
        self.mean.add_(delta * count / total)
        self.var.copy_((self.var * self.count + var * count + delta.square() * self.count * count / total) / total)
        self.count.copy_(total)

    def forward(self, values):
        return (values - self.mean) / self.std()

    def std(self):
        return torch.sqrt(self.var + self.eps ** 2)


def quat_from_rotvec(rotvec):
    """Convert a rotation vector to an xyzw quaternion, including at zero."""
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = 0.5 * torch.sinc(half / math.pi)
    return F.normalize(torch.cat((rotvec * scale, torch.cos(half)), dim=-1), dim=-1)


def rotvec_from_quat(quat):
    """Convert an xyzw quaternion to the shortest rotation vector."""
    quat = F.normalize(quat, dim=-1)
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    sin_half = torch.linalg.vector_norm(quat[..., :3], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 3:4].clamp_min(0.0))
    # For a unit quaternion, angle / sin(angle / 2) = 2 / sinc(angle / (2*pi)).
    # This denominator stays in [2/pi, 1], so zero rotations have finite gradients.
    scale = 2.0 / torch.sinc(angle / (2.0 * math.pi))
    return quat[..., :3] * scale


class UFOStateCodec(nn.Module):
    # z, quaternion xyzw, link v/w, q/dq. The previous action is a policy
    # observation and is carried separately from the physical state. The
    # learned dynamics output uses 58 structured increments: dz, dtheta,
    # dv, dw, dq, and dqd.
    physical_dim = 59
    delta_dim = 58
    state_dim = 59
    history_dim = 312

    def __init__(self, default_joint_pos, default_joint_vel, obs_scales, noise_lower,
                 noise_upper, action_rescale, clip_actions, root_com_pos=(0.0, 0.0, 0.0)):
        super().__init__()
        for name, value in (
            ("default_joint_pos", default_joint_pos), ("default_joint_vel", default_joint_vel),
            ("root_com_pos", root_com_pos),
            ("obs_scales", obs_scales), ("noise_lower", noise_lower), ("noise_upper", noise_upper),
            ("action_contract", [float(action_rescale), float("inf") if clip_actions is None else float(clip_actions)]),
            ("state_scale", [1.] * 5 + [3.] * 3 + [10.] * 3 + [1.] * 24 + [10.] * 24
             ),
            ("delta_floor", [3e-3] + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3
             + [3e-3] * 24 + [5e-2] * 24),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).detach().clone())
        self.action_rescale = float(action_rescale)
        self.clip_actions = clip_actions

    def env_action(self, action):
        action = (action * self.action_rescale).clamp(-self.action_rescale, self.action_rescale)
        return action if self.clip_actions is None else action.clamp(-self.clip_actions, self.clip_actions)

    @staticmethod
    def rotate_state(state, rotation):
        quat = F.normalize(quat_mul(rotation, state[:, 1:5]), dim=-1)
        quat = torch.where(quat[:, 3:4] < 0, -quat, quat)
        return torch.cat((state[:, :1], quat, quat_apply(rotation, state[:, 5:8]),
                          quat_apply(rotation, state[:, 8:11]), state[:, 11:]), dim=-1)

    def local_state(self, state):
        return self.rotate_state(state, quat_conjugate(yaw_quat(state[:, 1:5])))

    def delta_from_states(self, state, target):
        """Return frame-local physical increments from ``state`` to ``target``."""
        frame = quat_conjugate(yaw_quat(state[:, 1:5]))
        current = self.rotate_state(state, frame)
        target = self.rotate_state(target, frame)
        dtheta = rotvec_from_quat(quat_mul(target[:, 1:5], quat_conjugate(current[:, 1:5])))
        return torch.cat((
            target[:, :1] - current[:, :1], dtheta,
            target[:, 5:8] - current[:, 5:8],
            target[:, 8:11] - current[:, 8:11],
            target[:, 11:35] - current[:, 11:35],
            target[:, 35:59] - current[:, 35:59],
        ), dim=-1)

    def state_error(self, prediction, target):
        """Frame-local state error in the same component layout as increments."""
        return self.delta_from_states(prediction, target)

    def apply_delta(self, local_state, delta):
        """Apply a frame-local increment while keeping the quaternion unit norm."""
        dheight, dtheta, dlin, dang, dq, dqd = delta.split((1, 3, 3, 3, 24, 24), dim=-1)
        next_quat = quat_mul(quat_from_rotvec(dtheta), local_state[:, 1:5])
        next_quat = F.normalize(next_quat, dim=-1)
        next_quat = torch.where(next_quat[:, 3:4] < 0, -next_quat, next_quat)
        return torch.cat((
            local_state[:, :1] + dheight, next_quat,
            local_state[:, 5:8] + dlin, local_state[:, 8:11] + dang,
            local_state[:, 11:35] + dq, local_state[:, 35:59] + dqd,
        ), dim=-1)

    def observation(self, state, previous_action, history, heading_target, velocity_target, valid, *, noisy):
        quat = state[:, 1:5]
        # Warp's rotation formula differs by this term for slightly non-unit
        # simulator quaternions. Preserve its observed gravity/angular velocity.
        norm_correction = 2 * (quat.square().sum(-1, keepdim=True) - 1)
        gravity = torch.zeros_like(state[:, 5:8])
        gravity[:, 2] = -1
        policy = torch.cat((state[:, 11:35] - self.default_joint_pos,
                            state[:, 35:59] - self.default_joint_vel,
                            quat_apply_inverse(quat, gravity) + norm_correction * gravity,
                            quat_apply_inverse(quat, state[:, 8:11]) + norm_correction * state[:, 8:11]), -1)
        if noisy:
            policy = policy + self.noise_lower + torch.rand_like(policy) * (self.noise_upper - self.noise_lower)
        policy = policy * self.obs_scales
        frame = yaw_quat(quat)
        target_heading = quat_apply_inverse(frame, F.pad(heading_target, (0, 1)))[:, :2]
        heading = torch.cat((1 - target_heading[:, :1], target_heading[:, 1:2]), -1) * valid[:, :1]
        velocity = quat_apply_inverse(frame, F.pad(velocity_target, (0, 1)))[:, :2]
        return {"state": policy, "last_action": previous_action, "history_actor": history,
                "heading": heading,
                "velocity_command": torch.cat((velocity * valid[:, 1:2], valid[:, 1:2]), -1),
                "velocity_target_yaw": quat_apply_inverse(frame, state[:, 5:8])}

    @staticmethod
    def advance_history(history, obs):
        values = (obs["last_action"], obs["state"][:, 51:54], obs["state"][:, :24],
                  obs["state"][:, 24:48], obs["state"][:, 48:51])
        result, offset = [], 0
        for value in values:
            width = value.shape[-1]
            previous = history[:, offset:offset + 4 * width].reshape(-1, 4, width)
            result.append(torch.cat((previous[:, 1:], value[:, None]), 1).flatten(1))
            offset += 4 * width
        return torch.cat(result, -1)


class UFOWorldModel(nn.Module):
    # The learned transition ends before an external push. Replay carries the
    # sampled post-physics COM velocity override separately for autoregressive fitting.
    version = 9

    def __init__(self, cfg, codec_kwargs, xi_dim, actor_obs_dim, z_dim, device, *, kinematics=None):
        super().__init__()
        self.cfg_dict = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
        self.cfg = cfg = SimpleNamespace(**self.cfg_dict)
        # Older programmatic callers retain the original context-only objective.
        defaults = dict(objective="context", validation_max_p95=0.5,
                        imagination_max_disagreement=0.25, imagination_action_radius=0.5,
                        fb_covariance_batch_size=1024, fb_covariance_ridge=1e-3,
                        fb_max_projection_residual=0.1, fit_batches=1, bootstrap_members=False,
                        validation_batch_size=cfg.batch_size, fb_covariance_chunk_size=1024,
                        imagination_interval=1, imagination_microbatch_size=64,
                        scoring_chunk_size=2048, sequence_context_length=10)
        for key, value in defaults.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
                self.cfg_dict[key] = value
        if cfg.objective not in {"context", "fb_value", "fb_sequence"}:
            raise ValueError(f"Unknown world model objective: {cfg.objective}")
        if cfg.objective in {"fb_value", "fb_sequence"} and kinematics is None:
            raise ValueError("FB imagination requires verified privileged-state kinematics")
        if cfg.objective in {"fb_value", "fb_sequence"} and cfg.ensemble_size < 2:
            raise ValueError("FB imagination requires at least two dynamics members")
        for key in ("ensemble_size", "window_length", "horizon", "batch_size", "imagination_batch_size",
                    "update_interval", "ramp_updates", "validation_interval", "fb_covariance_batch_size",
                    "fit_batches", "validation_batch_size", "fb_covariance_chunk_size", "imagination_interval",
                    "imagination_microbatch_size", "scoring_chunk_size", "sequence_context_length"):
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
                raise ValueError(f"world_model.{key} must be a positive integer")
        if type(cfg.bootstrap_members) is not bool:
            raise ValueError("world_model.bootstrap_members must be a boolean")
        for key in ("learning_rate", "grad_clip", "validation_max_p95",
                    "imagination_max_disagreement", "imagination_action_radius", "fb_covariance_ridge",
                    "fb_max_projection_residual"):
            if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
                raise ValueError(f"world_model.{key} must be finite and positive")
        for key in ("actor_weight", "heading_weight", "velocity_weight"):
            if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
                raise ValueError(f"world_model.{key} must be finite and nonnegative")
        if (any(type(getattr(cfg, key)) is not int or getattr(cfg, key) < 0
                for key in ("warmup_updates", "value_warmup_updates"))
                or not 0 < cfg.value_tau <= 1 or not 0 <= cfg.value_lambda <= 1):
            raise ValueError("Invalid world model warmup or value_tau")
        if not cfg.validation_horizons or any(type(h) is not int or h < 1 for h in cfg.validation_horizons):
            raise ValueError("validation_horizons must contain positive integers")
        self.codec = UFOStateCodec(**codec_kwargs)
        self.kinematics = kinematics
        self.input_dim = self.codec.state_dim + 24 + xi_dim
        self.members = nn.ModuleList([
            mlp(self.input_dim, cfg.hidden_dims, self.codec.delta_dim, zero_output=True)
            for _ in range(cfg.ensemble_size)
        ])
        self.input_norm = RunningNorm(self.input_dim)
        self.delta_norm = RunningNorm(self.codec.delta_dim, eps=1e-4)
        # This value is trained on the same fixed z/commands as imagination.
        self.value = mlp(self.codec.state_dim + xi_dim + actor_obs_dim + z_dim,
                         cfg.value_hidden_dims, 1)
        self.target_value = copy.deepcopy(self.value).requires_grad_(False)
        if cfg.objective == "fb_sequence":
            from .world_model_sequence import SequenceValue

            self.sequence_value = SequenceValue(
                self.codec.state_dim, xi_dim, actor_obs_dim, z_dim,
                cfg.sequence_context_length, cfg.value_hidden_dims,
            )
            self.target_sequence_value = copy.deepcopy(self.sequence_value).requires_grad_(False)
        self.register_buffer("sequence_value_updates", torch.zeros((), dtype=torch.long))
        # Keep historical warmup/validation units: updates counts completed phases.
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("optimizer_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self.register_buffer("value_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("imagination_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("validated_at", torch.full((), -1, dtype=torch.long))
        self.register_buffer("validation_p95", torch.full((), float("inf")))
        self.register_buffer("fb_covariance", torch.eye(z_dim))
        self.register_buffer("fb_covariance_samples", torch.zeros((), dtype=torch.long))
        self.to(device)
        self.optimizer = torch.optim.Adam(self.members.parameters(), lr=cfg.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=cfg.learning_rate)
        if cfg.objective == "fb_sequence":
            self.sequence_value_optimizer = torch.optim.Adam(self.sequence_value.parameters(), lr=cfg.learning_rate)
        self._imagination_batch = None

    @property
    def device(self):
        return self.updates.device

    def validate_checkpoint(self, state):
        contract = {f"codec.{key}": value for key, value in self.codec.state_dict().items()}
        for key, current in contract.items():
            saved = state.get(key)
            if saved is None or not torch.equal(current.cpu(), saved.cpu()):
                raise ValueError(f"World model checkpoint contract mismatch: {key}. "
                                 "Use the original environment settings or start a fresh world model.")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Validate before loading any learned parameter or environment-derived buffer.
        self.validate_checkpoint(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def predict(self, state, action, xi, member=0):
        heading = yaw_quat(state[:, 1:5])
        local = self.codec.rotate_state(state, quat_conjugate(heading))
        executed = self.codec.env_action(action)
        features = torch.cat((local / self.codec.state_scale, executed / self.codec.action_rescale, xi), -1)
        output = self.members[member](self.input_norm(features))
        delta = output * self.delta_norm.std() + self.delta_norm.mean
        next_local = self.codec.apply_delta(local, delta)
        return self.codec.rotate_state(next_local, heading)

    def _scaled_state_error(self, prediction, target):
        error = self.codec.state_error(prediction, target)
        scale = torch.maximum(self.delta_norm.std(), self.codec.delta_floor)
        return (error / scale).square().mean(-1)

    def _apply_disturbance(self, state, disturbance, mask):
        """Replay the simulator's root COM velocity override in link coordinates."""
        com_offset_world = quat_apply(state[:, 1:5], self.codec.root_com_pos.expand(state.shape[0], -1))
        angular = disturbance[:, 3:6]
        linear = disturbance[:, :3] - torch.linalg.cross(angular, com_offset_world)
        velocity = torch.where(mask.reshape(-1, 1).bool(), torch.cat((linear, angular), -1), state[:, 5:11])
        return torch.cat((state[:, :5], velocity, state[:, 11:]), dim=-1)

    @torch.no_grad()
    def _update_norms(self, batch):
        state = batch["state"].reshape(-1, self.codec.state_dim)
        target = batch["next_state"].reshape(-1, self.codec.state_dim)
        action = batch["action"].reshape(-1, 24)
        xi = batch["xi"].reshape(-1, batch["xi"].shape[-1])
        local = self.codec.local_state(state)
        features = torch.cat((local / self.codec.state_scale, self.codec.env_action(action) / self.codec.action_rescale, xi), -1)
        self.input_norm.update(features)
        self.delta_norm.update(self.codec.delta_from_states(state, target))

    def _sample(self, replay, length, batch_size, *, validation=False):
        holdout = max(1, replay.num_envs // 8)
        split = replay.num_envs - holdout
        env_ids = torch.arange(split, replay.num_envs) if validation else torch.arange(split)
        batch = replay.sample(batch_size * length, seq_length=length, env_ids=env_ids)
        fields = {k: v.to(self.device).reshape(batch_size, length, -1) for k, v in batch["world_model"].items()}
        fields["action"] = batch["action"].to(self.device).reshape(batch_size, length, -1)
        fields["history"] = batch["observation"]["history_actor"].to(self.device).reshape(batch_size, length, -1)
        fields["z"] = batch["z"].to(self.device).reshape(batch_size, length, -1)
        if "aux_rewards" in batch and "heading_context_id" in batch["aux_rewards"]:
            fields["context_id"] = batch["aux_rewards"]["heading_context_id"].to(self.device).reshape(batch_size, length, -1)
        return fields

    def _available(self, replay, length, validation=False):
        _, starts, _, _ = replay._sampling_metadata(length)
        split = replay.num_envs - max(1, replay.num_envs // 8)
        mask = starts[:, 1] >= split if validation else starts[:, 1] < split
        ready = torch.tensor(int(bool(mask.any())), device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        return bool(ready)

    def fit(self, replay, agent):
        self._imagination_batch = None
        self.calls.add_(1)
        cfg = self.cfg
        if int(self.calls) % cfg.update_interval or not self._available(replay, cfg.window_length):
            return {}
        phase_losses, phase_gradients = [], []
        for _ in range(cfg.fit_batches):
            batches = [self._sample(replay, cfg.window_length, cfg.batch_size)
                       for _ in range(cfg.ensemble_size if cfg.bootstrap_members else 1)]
            for batch in batches:
                self._update_norms(batch)
            self.optimizer.zero_grad(set_to_none=True)
            member_losses = []
            for member in range(cfg.ensemble_size):
                batch = batches[member if cfg.bootstrap_members else 0]
                state, losses = batch["state"][:, 0], []
                for step in range(cfg.window_length):
                    state = self.predict(state, batch["action"][:, step], batch["xi"][:, step], member)
                    losses.append(self._scaled_state_error(state, batch["next_state"][:, step]).mean())
                    state = self._apply_disturbance(
                        state, batch["disturbance"][:, step], batch["disturbance_mask"][:, step]
                    )
                loss = torch.stack(losses).mean() / cfg.ensemble_size
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite world model fitting loss")
                # Release each member's graph before fitting the next member.
                loss.backward()
                member_losses.append(loss.detach())
            agent._avg_grad(self.members.parameters())
            grad = nn.utils.clip_grad_norm_(self.members.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_steps.add_(1)
            phase_losses.append(torch.stack(member_losses).sum())
            phase_gradients.append(grad.detach())
        self.updates.add_(1)
        imagination_length = (cfg.horizon + cfg.sequence_context_length if cfg.objective == "fb_sequence"
                              else cfg.horizon + 1 if cfg.objective == "fb_value" else 1)
        if (cfg.actor_weight > 0 and int(self.updates) > cfg.warmup_updates
                and int(self.updates) % cfg.imagination_interval == 0
                and self._available(replay, imagination_length)):
            self._imagination_batch = self._sample(replay, imagination_length, cfg.imagination_batch_size)
        mean_loss = torch.stack(phase_losses).mean()
        metrics = {"wm/loss": mean_loss, "wm/state_mse": mean_loss,
                   "wm/grad_norm": torch.stack(phase_gradients).mean(),
                   "wm/updates": self.updates.float().clone(),
                   "wm/fit_phases": self.updates.float().clone(),
                   "wm/optimizer_steps": self.optimizer_steps.float().clone()}
        if cfg.actor_weight > 0 and cfg.objective in {"fb_value", "fb_sequence"}:
            metrics.update(self._update_fb_covariance(replay, agent))
        if int(self.updates) % cfg.validation_interval == 0:
            metrics.update(self.validate(replay, agent))
        return metrics

    @torch.no_grad()
    def _update_fb_covariance(self, replay, agent):
        # FB trains its occupancy factorization against uniform real next states.
        # Use that same marginal, not expert data or the dynamics holdout split.
        covariance = torch.zeros_like(self.fb_covariance)
        count = covariance.new_zeros(())
        for start in range(0, self.cfg.fb_covariance_batch_size, self.cfg.fb_covariance_chunk_size):
            size = min(self.cfg.fb_covariance_chunk_size, self.cfg.fb_covariance_batch_size - start)
            sample = replay.sample(size)
            obs = {key: value.to(self.device) for key, value in sample["next"]["observation"].items()}
            with agent._autocast_context():
                embeddings = agent._target_backward_map(agent._b_in(obs)).float()
            covariance.add_(embeddings.T @ embeddings)
            count.add_(embeddings.shape[0])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(covariance)
            torch.distributed.all_reduce(count)
        covariance.div_(count)
        if not torch.isfinite(covariance).all():
            raise FloatingPointError("Non-finite FB embedding covariance")
        self.fb_covariance.copy_(covariance)
        self.fb_covariance_samples.copy_(count.long())
        eigenvalues = torch.linalg.eigvalsh(covariance)
        return {"wm/fb_covariance_min_eigenvalue": eigenvalues[0],
                "wm/fb_covariance_max_eigenvalue": eigenvalues[-1],
                "wm/fb_covariance_samples": count}

    def _value_input(self, agent, state, xi, obs, z):
        normalized = agent._obs_normalizer.normalize_dict(obs)
        return torch.cat((self.codec.local_state(state) / self.codec.state_scale,
                          xi, agent._actor_input(normalized), z), -1)

    def _reward(self, state, heading, velocity, valid, velocity_scale):
        forward = torch.zeros_like(state[:, 5:8])
        forward[:, 0] = 1
        heading_now = F.normalize(quat_apply(state[:, 1:5], forward)[:, :2], dim=-1, eps=1e-6)
        heading_reward = ((heading_now * heading).sum(-1, keepdim=True).clamp(-1, 1) - 1) * valid[:, :1]
        error = torch.linalg.vector_norm((state[:, 5:7] - velocity) / velocity_scale, dim=-1, keepdim=True)
        velocity_reward = -F.huber_loss(error, torch.zeros_like(error), reduction="none") * valid[:, 1:2]
        return self.cfg.heading_weight * heading_reward + self.cfg.velocity_weight * velocity_reward

    def actor_loss(self, agent):
        if self.cfg.objective == "fb_sequence":
            raise RuntimeError("Sequence imagination requires add_imagination_gradients for its joint value update")
        cfg, batch = self.cfg, self._imagination_batch
        self._imagination_batch = None
        if batch is None or cfg.actor_weight == 0 or int(self.updates) <= cfg.warmup_updates:
            return None, {}
        if cfg.objective == "fb_value":
            from .world_model_imagination import fb_imagination_loss

            return fb_imagination_loss(self, agent, batch)
        actor_ready = int(self.value_updates) >= cfg.value_warmup_updates
        weight = cfg.actor_weight * min(1., max(0, int(self.value_updates) - cfg.value_warmup_updates + 1)
                                       / cfg.ramp_updates) if actor_ready else 0.
        state, xi, previous_action, history, z = (
            batch[k][:, 0].detach() for k in ("state", "xi", "previous_action", "history", "z")
        )
        heading, velocity, valid = (batch[k][:, 0].detach() for k in ("heading", "velocity", "valid"))
        active = valid.any(dim=-1, keepdim=True).float()
        reward_list, value_inputs = [], []
        member = int(self.updates) % cfg.ensemble_size
        with frozen(self.members, self.target_value, getattr(agent._actor, "velocity_estimator", None)), \
                torch.set_grad_enabled(actor_ready):
            for _ in range(cfg.horizon):
                obs = self.codec.observation(
                    state, previous_action, history, heading, velocity, valid, noisy=True
                )
                value_inputs.append(self._value_input(agent, state, xi, obs, z).detach())
                # _norm_parts deliberately detaches for ordinary FB updates. BPTT must not.
                normalized = agent._obs_normalizer.normalize_dict(obs)
                with agent._autocast_context():
                    action = agent._actor_forward(normalized, z, imagination=True).mean.float()
                history = self.codec.advance_history(history, obs)
                previous_action = self.codec.env_action(action)
                state = self.predict(state, action, xi, member)
                reward_list.append(self._reward(state, heading, velocity, valid, agent.cfg.velocity_error_scale_mps))
            obs = self.codec.observation(
                state, previous_action, history, heading, velocity, valid, noisy=False
            )
            bootstrap = self.target_value(self._value_input(agent, state, xi, obs, z))
            total = bootstrap
            for reward in reversed(reward_list):
                total = reward + agent.cfg.discount * total
            denominator = active.sum().clamp_min(1.)
            loss = -weight * (total * active).sum() / denominator
        with torch.no_grad():
            values = list(self.target_value(torch.cat(value_inputs)).chunk(cfg.horizon)) + [bootstrap.detach()]
            targets = self.lambda_returns(reward_list, values, agent.cfg.discount, cfg.value_lambda)
        value_prediction = self.value(torch.cat(value_inputs))
        value_mask = active.repeat(cfg.horizon, 1)
        value_loss = (F.smooth_l1_loss(value_prediction, torch.cat(targets), reduction="none") * value_mask).sum()
        value_loss = value_loss / value_mask.sum().clamp_min(1.)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        agent._avg_grad(self.value.parameters())
        nn.utils.clip_grad_norm_(self.value.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        self.value_optimizer.step()
        self.value_updates.add_(1)
        # Target updates happen after the actor backward (its graph still reads these weights).
        return loss if actor_ready else None, {"wm/actor_loss": loss.detach(), "wm/actor_weight": loss.new_tensor(weight),
                      "wm/imagined_return": (total.detach() * active).sum() / denominator,
                      "wm/value_loss": value_loss.detach(), "wm/active_fraction": active.mean(),
                      "wm/value_updates": self.value_updates.float().clone()}

    def add_actor_gradients(self, agent, loss):
        """Average and clip only the model-based addition, preserving UFO's update."""
        parameters = tuple(p for p in agent._actor.parameters() if p.requires_grad)
        extra = torch.autograd.grad(loss, parameters, allow_unused=True)
        original = tuple(p.grad for p in parameters)
        try:
            for parameter, gradient in zip(parameters, extra):
                parameter.grad = gradient
            agent._avg_grad(parameters)
            norm = nn.utils.clip_grad_norm_(parameters, self.cfg.grad_clip, error_if_nonfinite=True)
            averaged = tuple(p.grad for p in parameters)
        finally:
            for parameter, gradient in zip(parameters, original):
                parameter.grad = gradient
        with torch.no_grad():
            base_square = sum((gradient.square().sum() for gradient in original if gradient is not None),
                              loss.new_zeros(()))
            extra_square = sum((gradient.square().sum() for gradient in averaged if gradient is not None),
                               loss.new_zeros(()))
            dot = sum(((base * addition).sum() for base, addition in zip(original, averaged)
                       if base is not None and addition is not None), loss.new_zeros(()))
            for parameter, addition in zip(parameters, averaged):
                if addition is not None:
                    if parameter.grad is None:
                        parameter.grad = addition
                    else:
                        parameter.grad.add_(addition)
        return {"wm/actor_grad_norm": norm.detach(),
                "wm/actor_grad_ratio": extra_square.sqrt() / base_square.sqrt().clamp_min(1e-8),
                "wm/actor_grad_cosine": dot / (base_square * extra_square).sqrt().clamp_min(1e-8)}

    @staticmethod
    @torch.no_grad()
    def lambda_returns(rewards, values, discount, value_lambda):
        total = values[-1]
        targets = []
        for step in reversed(range(len(rewards))):
            total = rewards[step] + discount * ((1 - value_lambda) * values[step + 1] + value_lambda * total)
            targets.append(total)
        return list(reversed(targets))

    @torch.no_grad()
    def update_target(self):
        for source, target in zip(self.value.parameters(), self.target_value.parameters()):
            target.lerp_(source, self.cfg.value_tau)

    @torch.no_grad()
    def validate(self, replay, agent=None):
        metrics = {}
        horizons = set(self.cfg.validation_horizons)
        if self.cfg.objective in {"fb_value", "fb_sequence"}:
            horizons.add(self.cfg.horizon)
        longest = next((h for h in sorted(horizons, reverse=True)
                        if self._available(replay, h, validation=True)), None)
        if longest is None:
            return metrics
        horizons = sorted(h for h in horizons if h <= longest)
        batch = self._sample(replay, longest, self.cfg.validation_batch_size, validation=True)
        member_states, member_errors, member_physical = [], [], []
        for member in range(self.cfg.ensemble_size):
            state = batch["state"][:, 0]
            states, errors, physical = [], [], []
            for step in range(longest):
                state = self.predict(state, batch["action"][:, step], batch["xi"][:, step], member)
                states.append(state)
                errors.append(self._scaled_state_error(state, batch["next_state"][:, step]).sqrt())
                physical.append(self.codec.state_error(state, batch["next_state"][:, step]))
                if step + 1 < longest:
                    state = self._apply_disturbance(
                        state, batch["disturbance"][:, step], batch["disturbance_mask"][:, step]
                    )
            member_states.append(torch.stack(states, 1))
            member_errors.append(torch.stack(errors, 1))
            member_physical.append(torch.stack(physical, 1))
        states = torch.stack(member_states)
        all_errors, all_physical = torch.stack(member_errors), torch.stack(member_physical)
        frame = quat_conjugate(yaw_quat(batch["state"][:, 0, 1:5]))
        for horizon in horizons:
            errors = all_errors[:, :, horizon - 1].mean(0)
            path_errors = all_errors[:, :, :horizon].amax(dim=(0, 2))
            metrics[f"wm/validation_{horizon}_mean"] = errors.mean()
            metrics[f"wm/validation_{horizon}_p95"] = torch.quantile(errors, .95)
            metrics[f"wm/validation_{horizon}_path_mean"] = path_errors.mean()
            metrics[f"wm/validation_{horizon}_path_p95"] = torch.quantile(path_errors, .95)
            if horizon == self.cfg.horizon:
                # The complete pre-push path and worst member must pass on every rank.
                quality = metrics[f"wm/validation_{horizon}_path_p95"].clone()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(quality, op=torch.distributed.ReduceOp.MAX)
                self.validation_p95.copy_(quality)
                self.validated_at.copy_(self.updates)
            predictions = []
            reference = self.codec.rotate_state(batch["next_state"][:, horizon - 1], frame)
            for member in range(self.cfg.ensemble_size):
                prediction = self.codec.rotate_state(states[member, :, horizon - 1], frame)
                sign = torch.where((prediction[:, 1:5] * reference[:, 1:5]).sum(-1, keepdim=True) < 0, -1., 1.)
                prediction = torch.cat((prediction[:, :1], prediction[:, 1:5] * sign, prediction[:, 5:]), -1)
                predictions.append(prediction[:, :59] / self.codec.state_scale[:59])
            metrics[f"wm/validation_{horizon}_disagreement"] = torch.stack(predictions).std(0, unbiased=False).mean()
            for suffix, physical_error in (("", all_physical[:, :, horizon - 1]),
                                           ("path_", all_physical[:, :, :horizon])):
                for name, error in zip(
                    ("height_m", "orientation_rad", "linear_velocity_mps", "angular_velocity_radps",
                     "joint_position_rad", "joint_velocity_radps"),
                    physical_error.split((1, 3, 3, 3, 24, 24), dim=-1),
                ):
                    metrics[f"wm/validation_{horizon}_{suffix}{name}_rmse"] = error.square().mean().sqrt()
        if self.cfg.objective in {"fb_value", "fb_sequence"} and agent is not None:
            metrics.update(self._fb_validation_metrics(states, batch, horizons, agent))
        return metrics

    @torch.no_grad()
    def _fb_validation_metrics(self, predictions, batch, horizons, agent):
        """Compare frozen B on clean physical paths; noise is excluded from both sides."""
        members, count, length, dim = predictions.shape
        physical = torch.cat((batch["next_state"].reshape(-1, dim), predictions.reshape(-1, dim)))
        embeddings = []
        for state in physical.split(self.cfg.fb_covariance_chunk_size):
            zeros = state.new_zeros
            obs = self.codec.observation(state, zeros((state.shape[0], 24)), zeros((state.shape[0], 312)),
                                         zeros((state.shape[0], 2)), zeros((state.shape[0], 2)),
                                         zeros((state.shape[0], 2)), noisy=False)
            obs["privileged_state"] = self.kinematics(state)
            with agent._autocast_context():
                embeddings.append(agent._target_backward_map(agent._b_in(obs)).float())
        embeddings = torch.cat(embeddings).reshape(members + 1, count, length, -1)
        difference = embeddings[1:] - embeddings[:1]
        metrics = {}
        for horizon in horizons:
            metrics[f"wm/validation_{horizon}_fb_embedding_rmse"] = difference[:, :, horizon - 1].square().mean().sqrt()
            metrics[f"wm/validation_{horizon}_path_fb_embedding_rmse"] = difference[:, :, :horizon].square().mean().sqrt()
        covariance = self.fb_covariance.float()
        if not torch.isfinite(covariance).all():
            raise FloatingPointError("Non-finite FB validation covariance")
        score_valid = bool(covariance.diagonal().mean() > 0)
        metrics["wm/validation_fb_score_valid"] = covariance.new_tensor(float(score_valid))
        if not score_valid:
            return metrics
        ridge = self.cfg.fb_covariance_ridge * covariance.diagonal().mean()
        regularized = covariance + ridge * torch.eye(covariance.shape[0], device=self.device)
        weights = torch.linalg.solve(regularized, batch["z"][:, 0].float().T).T
        score_error = (difference * weights[None, :, None]).sum(-1)
        for horizon in horizons:
            metrics[f"wm/validation_{horizon}_fb_score_rmse"] = score_error[:, :, horizon - 1].square().mean().sqrt()
            metrics[f"wm/validation_{horizon}_path_fb_score_rmse"] = score_error[:, :, :horizon].square().mean().sqrt()
        return metrics
