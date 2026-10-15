# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry for RP1 AMP observations (24 DoF)."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Append mirrored samples to the flat RSL-RL observations and actions."""
    env = env.unwrapped
    joint_names = tuple(env.scene["robot"].joint_names)
    action_names = tuple(env.action_manager.get_term("joint_pos")._joint_names)
    if action_names != joint_names:
        raise ValueError("RP1 AMP symmetry expects actions and joint observations in articulation order")

    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = _mirror_observation_group(
            obs["policy"], env.observation_manager, "policy", joint_names
        )
        obs_aug["critic"][batch_size:] = _mirror_observation_group(
            obs["critic"], env.observation_manager, "critic", joint_names
        )

    actions_aug = None
    if actions is not None:
        mirrored_actions = _switch_joints_left_right(actions, joint_names)
        actions_aug = torch.cat((actions, mirrored_actions), dim=0)

    return obs_aug, actions_aug


def _mirror_observation_group(
    obs: torch.Tensor,
    observation_manager,
    group_name: str,
    joint_names: tuple[str, ...],
) -> torch.Tensor:
    """Mirror each configured term, applying the same rule to every history frame."""
    term_names = observation_manager.active_terms[group_name]
    term_shapes = observation_manager.group_obs_term_dim[group_name]
    widths = [shape[-1] for shape in term_shapes]
    mirrored = obs.clone()
    terms = dict(zip(term_names, mirrored.split(widths, dim=-1), strict=True))

    # Terms are views into mirrored. Stride 3 selects one xyz component in every frame.
    if group_name == "critic":
        terms["base_lin_vel"][..., 1::3] *= -1

    terms["base_ang_vel"][..., 0::3] *= -1
    terms["base_ang_vel"][..., 2::3] *= -1
    terms["projected_gravity"][..., 1::3] *= -1
    terms["velocity_commands"][..., 1::3] *= -1
    terms["velocity_commands"][..., 2::3] *= -1

    # Only the opt-in path task has these four-component observation frames.
    if "command_path_state" in terms:
        terms["command_path_state"][..., 1::4] *= -1
        terms["command_path_state"][..., 2::4] *= -1

    for name in ("joint_pos", "joint_vel", "actions"):
        terms[name].copy_(_switch_joints_left_right(terms[name], joint_names))
    return mirrored


def _switch_joints_left_right(joint_data: torch.Tensor, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Mirror every joint frame, including flattened observation histories."""
    left_indices, right_indices, negate_indices = _joint_mirror_indices(joint_names)
    num_joints = len(joint_names)
    frames = joint_data.reshape(*joint_data.shape[:-1], -1, num_joints)
    mirrored = frames.clone()
    mirrored[..., left_indices] = frames[..., right_indices]
    mirrored[..., right_indices] = frames[..., left_indices]
    mirrored[..., negate_indices] *= -1
    return mirrored.reshape_as(joint_data)


@lru_cache(maxsize=8)
def _joint_mirror_indices(joint_names: tuple[str, ...]) -> tuple[list[int], list[int], list[int]]:
    """Build left/right pairs from runtime names, independent of joint order."""
    left_indices = []
    right_indices = []
    negate_indices = []
    for index, name in enumerate(joint_names):
        if name.startswith("left_"):
            right_name = "right_" + name.removeprefix("left_")
            left_indices.append(index)
            right_indices.append(joint_names.index(right_name))

        # Only ankle pitch keeps its sign; waist joints negate without swapping.
        if name not in {"left_ankle_pitch_joint", "right_ankle_pitch_joint"}:
            negate_indices.append(index)

    return left_indices, right_indices, negate_indices
