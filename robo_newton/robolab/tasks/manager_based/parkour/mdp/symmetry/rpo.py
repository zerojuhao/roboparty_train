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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    ``env`` is kept for compatibility with RSL-RL's symmetry callback signature.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _grid_scan_left_right_dims(
    env: ManagerBasedRLEnv,
    scanner_attr: str,
    hist_length: int,
) -> tuple[int, int, int]:
    """(history_length, ny, nx) from the running env cfg (dataclass fields are not on config *classes*)."""
    cfg = getattr(env, "unwrapped", env).cfg
    pat = getattr(cfg.scene, scanner_attr).pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height-map L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` (same arange endpoints / step).
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
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Left-right mirror for policy observations (``ObservationsCfg.PolicyCfg`` with ``concatenate_terms=False``)."""
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
    """Left-right mirror for critic observations."""
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
    """Apply a left-right symmetry transformation to depth image observations."""
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, 23)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[           
'left_thigh_yaw_joint',   #0
'right_thigh_yaw_joint',  #1
'torso_joint',            #2
'left_thigh_roll_joint',  #3
'right_thigh_roll_joint', #4
'left_arm_pitch_joint',   #5
'right_arm_pitch_joint',  #6
'left_thigh_pitch_joint', #7
'right_thigh_pitch_joint',#8
'left_arm_roll_joint',    #9
'right_arm_roll_joint',   #10
'left_knee_joint',        #11
'right_knee_joint',       #12
'left_arm_yaw_joint',     #13
'right_arm_yaw_joint',    #14
'left_ankle_pitch_joint', #15
'right_ankle_pitch_joint',#16
'left_elbow_pitch_joint', #17
'right_elbow_pitch_joint',#18
'left_ankle_roll_joint',  #19
'right_ankle_roll_joint', #20
'left_elbow_yaw_joint',   #21
'right_elbow_yaw_joint'   #22
]

"""


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = joint_data[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
    
    joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] = -1 * joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]]

    return joint_data_switched