# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Left-right symmetry for RP1 parkour (Isaac Lab BFS joint order, 24 DoF)."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 24

# Bilateral swap pairs in Isaac Lab joint order.
LEFT_JOINT_INDICES = [0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22]
RIGHT_JOINT_INDICES = [1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23]

# Sign flip after L-R swap: roll / yaw / waist joints (same convention as RPO symmetry).
NEGATE_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment nested policy/critic observations and actions with RP1 symmetry."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["critic"][:batch_size] = obs["critic"]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"])
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(env, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _grid_scan_left_right_dims(
    env: ManagerBasedRLEnv,
    scanner_attr: str,
    hist_length: int,
) -> tuple[int, int, int]:
    cfg = getattr(env, "unwrapped", env).cfg
    pat = getattr(cfg.scene, scanner_attr).pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height-map L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist_length, ny, nx


def _swap_pair_and_flip_y(feats: torch.Tensor, hist: int) -> torch.Tensor:
    """Mirror flattened ``[hist, 2, 3]`` positions: swap L/R and negate Y."""
    out = feats.view(feats.shape[0], hist, 2, 3).clone()
    swapped = out.clone()
    swapped[:, :, 0] = out[:, :, 1]
    swapped[:, :, 1] = out[:, :, 0]
    swapped[..., 1] = -swapped[..., 1]
    return swapped.reshape(feats.shape)


def _transform_height_scan_left_right(
    hs: torch.Tensor,
    hist: int,
    ny: int,
    nx: int,
) -> torch.Tensor:
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: TensorDict) -> TensorDict:
    obs = obs.clone()
    cfg = getattr(env, "unwrapped", env).cfg.observations.critic
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "foot_pos_b" in obs:
        hist = cfg.foot_pos_b.history_length or 1
        obs["foot_pos_b"] = _swap_pair_and_flip_y(obs["foot_pos_b"], hist)
    if "height_scan" in obs:
        hist, ny, nx = _grid_scan_left_right_dims(
            env, "height_scanner", cfg.height_scan.history_length
        )
        obs["height_scan"] = _transform_height_scan_left_right(obs["height_scan"], hist, ny, nx)
    if "left_foot_height_map" in obs and "right_foot_height_map" in obs:
        hist, ny, nx = _grid_scan_left_right_dims(
            env, "left_height_scanner", cfg.left_foot_height_map.history_length
        )
        left = _transform_height_scan_left_right(obs["left_foot_height_map"], hist, ny, nx)
        right = _transform_height_scan_left_right(obs["right_foot_height_map"], hist, ny, nx)
        obs["left_foot_height_map"] = right
        obs["right_foot_height_map"] = left
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, NUM_JOINTS)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions)
    return actions


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and negate RP1 reflection axes."""
    mirrored = joint_data.clone()
    mirrored[..., LEFT_JOINT_INDICES] = joint_data[..., RIGHT_JOINT_INDICES]
    mirrored[..., RIGHT_JOINT_INDICES] = joint_data[..., LEFT_JOINT_INDICES]
    mirrored[..., NEGATE_JOINT_INDICES] = -mirrored[..., NEGATE_JOINT_INDICES]
    return mirrored

"""
Isaac Lab joint ordering for RP1 (BFS):
[
    'left_hip_pitch_joint',      # 0
    'right_hip_pitch_joint',     # 1
    'waist_roll_joint',          # 2
    'left_hip_roll_joint',       # 3
    'right_hip_roll_joint',      # 4
    'waist_yaw_joint',           # 5
    'left_hip_yaw_joint',        # 6
    'right_hip_yaw_joint',       # 7
    'left_shoulder_pitch_joint', # 8
    'right_shoulder_pitch_joint',# 9
    'left_knee_joint',           # 10
    'right_knee_joint',          # 11
    'left_shoulder_roll_joint',  # 12
    'right_shoulder_roll_joint', # 13
    'left_ankle_pitch_joint',    # 14
    'right_ankle_pitch_joint',   # 15
    'left_shoulder_yaw_joint',   # 16
    'right_shoulder_yaw_joint',  # 17
    'left_ankle_roll_joint',     # 18
    'right_ankle_roll_joint',    # 19
    'left_elbow_joint',          # 20
    'right_elbow_joint',         # 21
    'left_wrist_joint',          # 22
    'right_wrist_joint',         # 23
]
"""
