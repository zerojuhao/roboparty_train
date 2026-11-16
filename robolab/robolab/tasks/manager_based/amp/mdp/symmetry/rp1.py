# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

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
    """Augment RP1 observations and actions with left-right symmetry."""

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2, *([1] * (obs.batch_dims - 1)))
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])
        if 'force_history' in obs:
            obs_aug['force_history'][batch_size:] = _transform_policy_obs_left_right(env, obs['force_history'], 'force_history')
        if 'force_vel_target' in obs:
            from ..force_estimation import mirror_force_vel_target
            obs_aug['force_vel_target'][batch_size:] = mirror_force_vel_target(obs['force_vel_target'])
        for group in ('force_target', 'interaction_critic'):
            if group not in obs:
                continue
            target = obs[group].clone()
            if group == 'force_target' and target.shape[-1] in (13, 14, 18, 28):
                from rsl_rl.modules.force_estimator import mirror_compact_force
                obs_aug[group][batch_size:] = mirror_compact_force(target)
                continue
            if target.shape[-1] in (81, 84):
                from rsl_rl.modules.force_estimator import mirror_multi_force
                target[..., :81] = mirror_multi_force(target[..., :81])
                if group == 'interaction_critic':
                    target[..., 82] *= -1
                obs_aug[group][batch_size:] = target
                continue
            target[..., [1, 4]] *= -1
            target[..., 6:13] = target[..., [6, 8, 7, 10, 9, 12, 11]]
            if group == 'interaction_critic':
                target[..., 14] *= -1
            elif target.shape[-1] == 15:
                target[..., 14] *= -1
            obs_aug[group][batch_size:] = target
        if 'compact_interaction_critic' in obs:
            from ..force_estimation import mirror_compact_critic_state
            obs_aug['compact_interaction_critic'][batch_size:] = mirror_compact_critic_state(
                obs['compact_interaction_critic'])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(env, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _history_length(env: ManagerBasedRLEnv, group_name: str) -> int:
    cfg = getattr(env, "unwrapped", env).cfg
    history_length = getattr(getattr(cfg.observations, group_name), "history_length", 0)
    return history_length if history_length is not None and history_length > 0 else 1


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor, group_name='policy') -> torch.Tensor:
    """Left-right mirror for flat policy observations using IsaacLab term-major history layout."""

    obs_shape = obs.shape
    num_actions = _num_actions(env)
    history_length = _history_length(env, group_name)
    expected_dim = history_length * (3 + 3 + 3 + num_actions + num_actions + num_actions)
    assert obs_shape[-1] == expected_dim, f"Expected policy obs dim to be {expected_dim}, got {obs_shape[-1]}."

    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim

    term_dim = num_actions * history_length
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror for flat critic observations using IsaacLab term-major history layout."""

    obs_shape = obs.shape
    num_actions = _num_actions(env)
    history_length = _history_length(env, "critic")
    expected_dim = history_length * (3 + 3 + 3 + 3 + num_actions + num_actions + num_actions)
    assert obs_shape[-1] == expected_dim, f"Expected critic obs dim to be {expected_dim}, got {obs_shape[-1]}."

    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim

    term_dim = num_actions * history_length
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(env, obs[..., offset : offset + term_dim])
    offset += term_dim
    return obs


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(env: ManagerBasedRLEnv, joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    num_actions = _num_actions(env)
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, num_actions)
    joint_data = _switch_joints_left_right(env, joint_data)
    return joint_data.reshape(joint_data_shape)


def _transform_actions_left_right(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    return _switch_joints_left_right(env, actions)


def _num_actions(env: ManagerBasedRLEnv) -> int:
    robot = getattr(env, "unwrapped", env).scene["robot"]
    return robot.data.joint_pos.torch.shape[1]


def _joint_names(env: ManagerBasedRLEnv) -> list[str]:
    return list(getattr(env, "unwrapped", env).scene["robot"].joint_names)


def _switch_joints_left_right(env: ManagerBasedRLEnv, joint_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry using the robot's Isaac joint names."""
    return mirror_joints(joint_data, _joint_names(env))
