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

from robolab.tasks.direct.base.mdp.symmetry.rp1 import mirror_joints

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment nested policy/critic observations and actions with RP1 symmetry."""
    joint_names = _joint_names(env)
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2, *([1] * (obs.batch_dims - 1)))
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["critic"][:batch_size] = obs["critic"]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"], joint_names)
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(env, obs["critic"], joint_names)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions, joint_names)
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
    """Mirror lateral (y) grid rows; scalars only reorder (unlike ``depth_image``, scan is flat w.r.t. spatial axes)."""
    out = hs.reshape(*hs.shape[:-1], hist, ny, nx).flip(dims=[-2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict, joint_names: list[str]) -> TensorDict:
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"], joint_names)
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"], joint_names)
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"], joint_names)
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(
    env: ManagerBasedRLEnv, obs: TensorDict, joint_names: list[str]
) -> TensorDict:
    obs = obs.clone()
    cfg = getattr(env, "unwrapped", env).cfg.observations.critic
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"], joint_names)
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"], joint_names)
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"], joint_names)
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


def _switch_joints_left_right_flat(joint_data: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, len(joint_names))
    joint_data = mirror_joints(joint_data, joint_names)
    return joint_data.reshape(joint_data_shape)


def _transform_actions_left_right(actions: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    return mirror_joints(actions, joint_names)


def _joint_names(env: ManagerBasedRLEnv) -> list[str]:
    return list(getattr(env, "unwrapped", env).scene["robot"].joint_names)
