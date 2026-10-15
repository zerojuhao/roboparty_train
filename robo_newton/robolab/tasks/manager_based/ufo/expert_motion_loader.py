"""Expert trajectory loading from the vendored UFO motion library.

Port of `humanoidverse/agents/envs/expert_motion_loader.py`: builds the expert
replay episodes (obs only, no actions needed) consumed by the discriminator
and the backward-map goal encoder.
"""

from __future__ import annotations

import numpy as np
import torch

from .mdp.motion_observations import compute_humanoid_observations_max
from .mdp.pufo_observations import heading_xy
from robolab.tasks.manager_based.ufo.motion.torch_utils import quat_rotate_inverse


def load_expert_trajectories_from_motion_lib(motion_lib, dt: float, default_dof_pos: torch.Tensor, device="cpu", base_ang_vel_scale: float = 1.0):
    motion_lib.load_motions_for_training()
    episodes = []
    file_names = []
    for i in range(motion_lib._num_unique_motions):
        motion_times = torch.arange(int(np.ceil((motion_lib._motion_lengths[i] / dt).cpu()))).to(device) * dt
        motion_id = torch.tensor([i]).to(device).repeat(motion_times.shape[0])
        motion_res = motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(motion_lib._motion_data_keys[i])

        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots = motion_res["rg_rot_t"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=True,
        )
        max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

        base_quat = ref_body_rots[:, 0]
        reference_heading = heading_xy(base_quat)
        ref_dof_pos = motion_res["dof_pos"] - default_dof_pos.to(motion_res["dof_pos"].device)[0]
        ref_dof_vel = motion_res["dof_vel"]
        ref_ang_vel = quat_rotate_inverse(base_quat, ref_body_angular_vels[:, 0], w_last=True)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=ref_ang_vel.device).repeat(max_local_self_obs.shape[0], 1)
        projected_gravity = quat_rotate_inverse(base_quat, gravity_vec, w_last=True)
        bogus_actions = ref_dof_pos * 0

        state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel * base_ang_vel_scale], dim=-1)

        curr_motion_len = state.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool, device=device)
        truncated[-1] = True

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(curr_motion_len, dtype=bool, device=device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long, device=device) * i,
            "heading_forward_xy": reference_heading,
            "reference_velocity_world_xy": ref_body_vels[:, 0, :2],
            "reference_velocity_valid": torch.ones(
                curr_motion_len, 1, dtype=torch.bool, device=device
            ),
        }
        episodes.append(ep)
    return episodes, file_names
