"""Observation terms for the UFO task.

The policy observation is pure proprioception (the motion goal is carried by
the z latent). The critic observation is the UFO "max_local_self" feature
(373-dim) computed with the vendored `compute_humanoid_observations_max`.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

from .motion_observations import compute_humanoid_observations_max


def _as_tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def local_self_obs(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    body_pos_w = _as_tensor(robot.data.body_pos_w)
    body_quat_w = _as_tensor(robot.data.body_quat_w)
    body_lin_vel_w = _as_tensor(robot.data.body_link_lin_vel_w)
    body_ang_vel_w = _as_tensor(robot.data.body_link_ang_vel_w)
    obs_dict = compute_humanoid_observations_max(
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
        body_ang_vel_w,
        local_root_obs=True,
        root_height_obs=True,
    )
    return torch.cat([v for v in obs_dict.values()], dim=-1)
