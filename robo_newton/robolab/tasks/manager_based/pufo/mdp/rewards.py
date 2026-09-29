"""PUFO context rewards used by the heading and velocity critics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from robolab.tasks.manager_based.ufo.mdp.rewards import (
    compute_aux_rewards as _compute_ufo_aux_rewards,
    initialize_aux_rewards,
    zero_reward,
)


def _reference_velocity_yaw(env):
    command = env.command_manager.get_term("motion")
    reference = command.get_motion_state()
    robot = env.scene["robot"]
    quat = robot.data.body_quat_w[:, 0]
    root_vel = robot.data.body_link_lin_vel_w[:, 0]
    root_ang_vel = robot.data.body_link_ang_vel_w[:, 0]
    from isaaclab.utils.math import quat_apply_inverse
    target_xy = quat_apply_inverse(quat, reference["root_vel"])[:, :2]
    actual_xy = quat_apply_inverse(quat, root_vel)[:, :2]
    target = torch.cat((target_xy, reference["root_ang_vel"][:, 2:3]), dim=-1)
    actual = torch.cat((actual_xy, root_ang_vel[:, 2:3]), dim=-1)
    return actual, target


def heading_tracking(env):
    from .observations import heading_context
    return heading_context(env)[:, 0]


def velocity_tracking(env, error_scale_mps: float = 0.5):
    actual, target = _reference_velocity_yaw(env)
    error = (actual[:, :2] - target[:, :2]) / float(error_scale_mps)
    scaled_norm = torch.linalg.vector_norm(error, dim=-1)
    return F.huber_loss(scaled_norm, torch.zeros_like(scaled_norm), reduction="none", delta=1.0)


def compute_aux_rewards(env):
    aux = dict(_compute_ufo_aux_rewards(env))
    aux["heading_tracking"] = heading_tracking(env)
    aux["velocity_tracking"] = velocity_tracking(
        env,
        float(getattr(env, "_pufo_velocity_error_scale_mps", 0.5)),
    )
    return aux


__all__ = [
    "compute_aux_rewards",
    "heading_tracking",
    "initialize_aux_rewards",
    "velocity_tracking",
    "zero_reward",
]
