"""On-demand expert observations with the original UFO trajectory sampling API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from rsl_rl.storage.replay_buffer import TrajectoryDictBuffer
from torch.utils._pytree import tree_map

from robolab.tasks.manager_based.ufo.motion.torch_utils import quat_rotate_inverse

from .motion_observations import compute_humanoid_observations_max
from .pufo_observations import heading_xy

if TYPE_CHECKING:
    from robolab.tasks.manager_based.ufo.motion.motion_lib_npz import MotionLibNPZ


def expert_observations(
    motion_res: dict[str, torch.Tensor], default_dof_pos: torch.Tensor, base_ang_vel_scale: float
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Reconstruct UFO observations and world-frame commands from sampled motion frames."""
    positions = motion_res["rg_pos_t"]
    rotations = motion_res["rg_rot_t"]
    velocities = motion_res["body_vel_t"]
    angular_velocities = motion_res["body_ang_vel_t"]
    obs = compute_humanoid_observations_max(
        positions,
        rotations,
        velocities,
        angular_velocities,
        local_root_obs=True,
        root_height_obs=True,
    )
    base_quat = rotations[:, 0]
    dof_pos = motion_res["dof_pos"] - default_dof_pos.to(positions.device)[0]
    ang_vel = quat_rotate_inverse(base_quat, angular_velocities[:, 0], w_last=True)
    gravity = positions.new_tensor([0.0, 0.0, -1.0]).expand(len(positions), -1)
    projected_gravity = quat_rotate_inverse(base_quat, gravity, w_last=True)
    state = torch.cat((dof_pos, motion_res["dof_vel"], projected_gravity, ang_vel * base_ang_vel_scale), -1)
    return {
        "observation": {
            "state": state,
            "last_action": torch.zeros_like(dof_pos),
            "privileged_state": torch.cat(list(obs.values()), -1),
        },
        "heading_forward_xy": heading_xy(base_quat),
        "reference_velocity_world_xy": velocities[:, 0, :2],
        "reference_velocity_world": velocities[:, 0, :3],
        "reference_velocity_valid": torch.ones(len(state), 1, dtype=torch.bool, device=state.device),
    }


class MotionExpertBuffer(TrajectoryDictBuffer):
    """Reuse trajectory sampling while materializing only requested expert frames."""

    def __init__(
        self,
        motion_lib: MotionLibNPZ,
        dt: float,
        default_dof_pos: torch.Tensor,
        *,
        base_ang_vel_scale: float = 1.0,
        seq_length: int = 1,
        output_key_t: Sequence[str] = ("observation",),
        output_key_tp1: Sequence[str] = ("observation",),
        compile_mode: str | None = None,
    ) -> None:
        """Index motion clips without precomputing all expert observations.

        Args:
            motion_lib: Loaded source motions and their sampling device.
            dt: Time between expert frames, in seconds.
            default_dof_pos: Joint offsets with shape (1, joints) or (environments, joints).
            base_ang_vel_scale: Scale applied to local root angular velocity.
            seq_length: Default number of consecutive frames per sampled sequence.
            output_key_t: Fields returned for current frames.
            output_key_tp1: Fields returned for successor frames.
            compile_mode: PyTorch compile mode for CUDA reconstruction, or None to run eagerly.
        """
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Expert sampling dt must be finite and positive")
        self.motion_lib, self.dt = motion_lib, dt
        self.device = motion_lib.device
        self.default_dof_pos = default_dof_pos[:1].detach().to(self.device).clone()
        self.base_ang_vel_scale = base_ang_vel_scale
        self._observe = expert_observations
        if compile_mode is not None and torch.device(self.device).type == "cuda":
            self._observe = torch.compile(
                self._observe, dynamic=True, fullgraph=True, options=torch._inductor.list_mode_options(compile_mode)
            )
        self.seq_length = seq_length
        self.output_key_t, self.output_key_tp1 = list(output_key_t), list(output_key_tp1)
        self.end_key = "truncated"
        self.lengths = torch.ceil(motion_lib._motion_lengths / dt).long()
        if not bool((self.lengths > 0).all()):
            raise ValueError("Every expert clip must contain at least one sampled frame")
        ends = self.lengths.cumsum(0)
        self.start_idx = (ends - self.lengths)[:, None]
        self.stop_idx = (ends - 1)[:, None]
        self._capacity = int(ends[-1])
        self.motion_ids = list(range(len(self.lengths)))
        self.file_names = list(motion_lib._motion_data_keys)
        self.priorities = torch.ones_like(self.lengths, dtype=torch.float32)
        self.storage = {}
        self._sampling_cache = {}
        self.phase_probabilities = None
        self.phase_bin_steps = 1

    @property
    def capacity(self) -> int:
        """Number of addressable expert frames across all clips."""
        return self._capacity

    @torch.no_grad()
    def _gather(self, indices, keys):
        if not keys:
            return {}
        indices = indices[0] if isinstance(indices, tuple) else indices
        motion_ids = torch.searchsorted(self.stop_idx[:, 0].contiguous(), indices)
        times = (indices - self.start_idx[motion_ids, 0]).float() * self.dt
        motion = self.motion_lib.get_motion_state(motion_ids, times)
        fields = self._observe(motion, self.default_dof_pos, self.base_ang_vel_scale)
        return {key: fields[key] for key in keys}

    def _gather_transition(self, indices, next_indices):
        if not self.output_key_t or not self.output_key_tp1:
            return super()._gather_transition(indices, next_indices)
        indices = indices[0] if isinstance(indices, tuple) else indices
        next_indices = next_indices[0] if isinstance(next_indices, tuple) else next_indices
        count = len(indices)
        keys = tuple(dict.fromkeys((*self.output_key_t, *self.output_key_tp1)))
        # Reconstruct both sides in one batch to share motion queries and GPU dispatch.
        fields = self._gather(torch.cat((indices, next_indices)), keys)
        output = tree_map(lambda value: value[:count], {key: fields[key] for key in self.output_key_t})
        output["next"] = tree_map(lambda value: value[count:], {key: fields[key] for key in self.output_key_tp1})
        return output


def load_expert_trajectories_from_motion_lib(
    motion_lib: MotionLibNPZ,
    dt: float,
    default_dof_pos: torch.Tensor,
    device: str | torch.device = "cpu",
    base_ang_vel_scale: float = 1.0,
):
    """Materialize complete expert episodes for callers using the eager loading API."""
    motion_lib.load_motions_for_training()
    episodes = []
    file_names = []
    for i in range(motion_lib._num_unique_motions):
        motion_times = torch.arange(int(torch.ceil(motion_lib._motion_lengths[i] / dt)), device=device) * dt
        motion_id = torch.tensor([i]).to(device).repeat(motion_times.shape[0])
        motion_res = motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(motion_lib._motion_data_keys[i])

        fields = expert_observations(motion_res, default_dof_pos, base_ang_vel_scale)
        curr_motion_len = motion_times.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool, device=device)
        truncated[-1] = True

        ep = {
            **fields,
            "terminated": torch.zeros(curr_motion_len, dtype=bool, device=device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long, device=device) * i,
        }
        episodes.append(ep)
    return episodes, file_names
