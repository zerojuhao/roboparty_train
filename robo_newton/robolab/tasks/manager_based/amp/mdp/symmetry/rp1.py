# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry for RP1 AMP observations (24 DoF)."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 24
HISTORY_LENGTH = 3

# RP1 IsaacLab BFS joint order. Each left/right pair is adjacent except waist joints.
LEFT_JOINT_INDICES = [0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22]
RIGHT_JOINT_INDICES = [1, 4, 7, 9, 11, 13, 15, 17, 19, 21, 23]

# Ankle pitch keeps its sign; all other coordinates negate after reflection.
NEGATE_JOINT_INDICES = [index for index in range(NUM_JOINTS) if index not in (14, 15)]


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
        obs_aug["critic"][batch_size:] = _transform_critic_obs_left_right(obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Mirror policy terms by name."""
    if isinstance(obs, torch.Tensor):
        return _named_policy_to_flat(_transform_policy_obs_left_right(_flat_to_named_policy(obs)))
    mirrored = obs.clone()
    mirrored["base_ang_vel"] = _apply_xyz_sign(mirrored["base_ang_vel"], [-1, 1, -1])
    mirrored["projected_gravity"] = _apply_xyz_sign(mirrored["projected_gravity"], [1, -1, 1])
    mirrored["velocity_commands"] = _apply_xyz_sign(mirrored["velocity_commands"], [1, -1, -1])
    mirrored["joint_pos"] = _switch_joints_left_right_flat(mirrored["joint_pos"])
    mirrored["joint_vel"] = _switch_joints_left_right_flat(mirrored["joint_vel"])
    mirrored["actions"] = _switch_joints_left_right_flat(mirrored["actions"])
    return mirrored


def _transform_critic_obs_left_right(obs: TensorDict) -> TensorDict:
    """Mirror privileged critic terms by name."""
    if isinstance(obs, torch.Tensor):
        return _named_critic_to_flat(_transform_critic_obs_left_right(_flat_to_named_critic(obs)))
    mirrored = obs.clone()
    mirrored["base_lin_vel"] = _apply_xyz_sign(mirrored["base_lin_vel"], [1, -1, 1])
    mirrored["base_ang_vel"] = _apply_xyz_sign(mirrored["base_ang_vel"], [-1, 1, -1])
    mirrored["projected_gravity"] = _apply_xyz_sign(mirrored["projected_gravity"], [1, -1, 1])
    mirrored["velocity_commands"] = _apply_xyz_sign(mirrored["velocity_commands"], [1, -1, -1])
    mirrored["joint_pos"] = _switch_joints_left_right_flat(mirrored["joint_pos"])
    mirrored["joint_vel"] = _switch_joints_left_right_flat(mirrored["joint_vel"])
    mirrored["actions"] = _switch_joints_left_right_flat(mirrored["actions"])
    return mirrored


def _flat_to_named_policy(obs: torch.Tensor) -> TensorDict:
    """Adapt RSL-RL's flat rollout tensor to named AMP policy terms."""
    terms = torch.split(obs, [9, 9, 9, 72, 72, 72], dim=-1)
    return TensorDict(
        dict(zip(("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"), terms)),
        batch_size=obs.shape[:-1],
    )


def _flat_to_named_critic(obs: torch.Tensor) -> TensorDict:
    """Adapt RSL-RL's flat rollout tensor to named AMP critic terms."""
    terms = torch.split(obs, [9, 9, 9, 9, 72, 72, 72], dim=-1)
    return TensorDict(
        dict(zip(("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"), terms)),
        batch_size=obs.shape[:-1],
    )


def _named_policy_to_flat(obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[name] for name in ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")], dim=-1)


def _named_critic_to_flat(obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[name] for name in ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")], dim=-1)


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    """Apply an xyz sign pattern to each history frame."""
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    return (obs * obs.new_tensor(signs)).reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    """Apply the RP1 joint transform to each history frame."""
    obs_shape = joint_data.shape
    joint_data = joint_data.reshape(*obs_shape[:-1], -1, NUM_JOINTS)
    return _switch_joints_left_right(joint_data).reshape(obs_shape)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply the RP1 joint transform to actions."""
    return _switch_joints_left_right(actions)


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and negate RP1 reflection axes."""
    mirrored = joint_data.clone()
    mirrored[..., LEFT_JOINT_INDICES] = joint_data[..., RIGHT_JOINT_INDICES]
    mirrored[..., RIGHT_JOINT_INDICES] = joint_data[..., LEFT_JOINT_INDICES]
    mirrored[..., NEGATE_JOINT_INDICES] = -mirrored[..., NEGATE_JOINT_INDICES]
    return mirrored
