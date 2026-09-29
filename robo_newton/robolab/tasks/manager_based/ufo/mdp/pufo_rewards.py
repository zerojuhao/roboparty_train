"""PUFO auxiliary penalties and motion-context rewards."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat

# WBC regularization weights (RP1 config), reused verbatim.
ANGULAR_MOMENTUM_RATE_MAX_PENALTY = 1.0e4
YAW_ANGULAR_MOMENTUM_MAX_PENALTY = 1.0e3

# Contact regularization follows the WBC/IsaacLab convention of using the
# normal component of the contact force.  A contact in any sample of the
# three-frame sensor history activates the per-body penalty.
UNDESIRED_CONTACT_FORCE_THRESHOLD = 1.0
UNDESIRED_CONTACT_HISTORY_WINDOW = 3
UNDESIRED_CONTACT_HEIGHT_THRESHOLD = 0.30
UNDESIRED_CONTACT_HEIGHT_BAND = 0.05


def _t(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def initialize_aux_rewards(env: ManagerBasedEnv) -> None:
    """Bind the fixed robot/sensor layout and clear history before collecting rollouts."""
    robot = env.scene["robot"]
    names = robot.data.body_names
    feet = [i for i, name in enumerate(names) if "ankle_roll_link" in name][:2]
    env._ufo_feet_ids = torch.tensor(feet, dtype=torch.long, device=env.device)
    joint_names = list(getattr(robot.data, "joint_names", ()))
    env._ufo_ankle_roll_ids = torch.tensor(
        [i for i, name in enumerate(joint_names) if "ankle_roll" in name],
        dtype=torch.long,
        device=env.device,
    )
    env._ufo_undesired_body_ids = torch.tensor(
        [i for i in range(len(names)) if i not in feet], dtype=torch.long, device=env.device
    )
    history = _t(env.scene["contact_forces"].data.net_normal_forces_w_history)
    if history.ndim != 4 or history.shape[1] < UNDESIRED_CONTACT_HISTORY_WINDOW:
        raise ValueError("UFO requires normal-force history shaped [env, history >= 3, body, xyz].")
    if _t(robot.data.body_mass).shape != (env.num_envs, len(names)):
        raise ValueError("UFO requires per-environment body masses shaped [env, body].")
    if _t(robot.data.body_inertia).shape != (env.num_envs, len(names), 9):
        raise ValueError("UFO requires body inertias shaped [env, body, 9].")
    env._ufo_step_dt = float(env.step_dt)
    if not math.isfinite(env._ufo_step_dt) or env._ufo_step_dt <= 0.0:
        raise ValueError(f"Environment step_dt must be finite and positive, got {env._ufo_step_dt}.")
    env._ufo_aux_cache = None
    env._ufo_jerk_prev_action = None
    env._ufo_jerk_prev_prev_action = None
    env._ufo_jerk_steps = None
    env._ufo_l_com_prev = None
    env._ufo_l_com_valid = None


def _height_gated_undesired_contacts(
    normal_forces_w_history: torch.Tensor,
    body_heights_w: torch.Tensor,
    body_ids: torch.Tensor,
) -> torch.Tensor:
    """Count undesired contacts over a short history with a smooth per-body height gate."""
    force_magnitudes = torch.linalg.vector_norm(
        normal_forces_w_history[:, :UNDESIRED_CONTACT_HISTORY_WINDOW, body_ids, :], dim=-1
    )
    undesired_contact = (force_magnitudes > UNDESIRED_CONTACT_FORCE_THRESHOLD).any(dim=1)

    height_gate = torch.sigmoid(
        (body_heights_w[:, body_ids] - UNDESIRED_CONTACT_HEIGHT_THRESHOLD)
        / UNDESIRED_CONTACT_HEIGHT_BAND
    )
    # BeyondMimic/IsaacLab counts every body with an undesired contact.  The
    # smooth gate makes this a fractional count near the ground.
    return (height_gate * undesired_contact.float()).sum(dim=1)


def _peak_contact_forces(normal_forces_w_history: torch.Tensor) -> torch.Tensor:
    """Select the strongest contact-force sample for every body."""
    force_norm = torch.linalg.vector_norm(normal_forces_w_history, dim=-1)
    peak_index = force_norm.argmax(dim=1, keepdim=True)
    gather_index = peak_index.unsqueeze(-1).expand(-1, -1, -1, 3)
    return torch.gather(normal_forces_w_history, dim=1, index=gather_index).squeeze(1)


def _body_contact_severity(
    contact_force: torch.Tensor,
    body_velocity: torch.Tensor,
    robot_weight: torch.Tensor,
) -> torch.Tensor:
    """Match H200's loaded-contact impact penalty for non-foot bodies."""
    if contact_force.shape[1] == 0:
        return torch.zeros(contact_force.shape[0], device=contact_force.device, dtype=contact_force.dtype)
    weight = robot_weight.reshape(-1, 1).clamp_min(1.0e-6)
    force_norm = torch.linalg.vector_norm(contact_force, dim=-1)
    force_direction = contact_force / force_norm.unsqueeze(-1).clamp_min(1.0e-6)
    impact_velocity = torch.relu((body_velocity * force_direction).sum(dim=-1))
    force_ratio = torch.relu(force_norm / weight - 0.05)
    force_severity = F.smooth_l1_loss(
        force_ratio,
        torch.zeros_like(force_ratio),
        beta=0.25,
        reduction="none",
    )
    impact_gain = 1.0 + torch.clamp(impact_velocity, min=0.0, max=2.0).square()
    return torch.clamp((force_severity * impact_gain).sum(dim=1), max=4.0)


def _feet_stumble_penalty(foot_force_history: torch.Tensor) -> torch.Tensor:
    """Penalize horizontal-dominant foot impacts using H200's thresholds."""
    force_xy = torch.linalg.vector_norm(foot_force_history[..., :2], dim=-1)
    force_z = torch.abs(foot_force_history[..., 2])
    excess = torch.relu(force_xy - force_z - 20.0)
    return torch.clamp(excess / 100.0, max=1.0).amax(dim=(1, 2))


def _tangential_contact_speed(
    contact_force: torch.Tensor,
    body_velocity: torch.Tensor,
    robot_weight: torch.Tensor,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot velocity tangent to a currently loaded contact."""
    force_norm = torch.linalg.vector_norm(contact_force, dim=-1)
    force_direction = contact_force / force_norm.unsqueeze(-1).clamp_min(1.0e-6)
    normal_velocity = (body_velocity * force_direction).sum(dim=-1, keepdim=True)
    tangent_velocity = body_velocity - normal_velocity * force_direction
    loaded = force_norm > force_threshold
    return (torch.linalg.vector_norm(tangent_velocity, dim=-1) * loaded).sum(dim=1)


def _finite_bounded_penalty(penalty: torch.Tensor, max_penalty: float) -> torch.Tensor:
    penalty = torch.nan_to_num(penalty, nan=max_penalty, posinf=max_penalty, neginf=0.0)
    return penalty.clamp(max=max_penalty)


def _reset_mask(env: ManagerBasedEnv) -> torch.Tensor | None:
    """Per-env reset flags for the current step, when the env exposes them."""
    mask = None
    for attr in ("reset_terminated", "reset_time_outs"):
        value = getattr(env, attr, None)
        if value is None:
            continue
        value = _t(value).reshape(-1).bool()
        mask = value if mask is None else (mask | value)
    return mask


def _com_angular_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """Total CoM angular momentum (orbital + spin), using live mass and inertia."""
    robot = env.scene["robot"]
    body_pos_w = _t(robot.data.body_com_pos_w)
    body_lin_vel_w = _t(robot.data.body_com_lin_vel_w)
    body_ang_vel_w = _t(robot.data.body_com_ang_vel_w)
    body_quat_w = _t(robot.data.body_com_quat_w)
    masses = _t(robot.data.body_mass)
    inertias = _t(robot.data.body_inertia).reshape(*body_pos_w.shape[:2], 3, 3)

    total_mass = masses.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    com_pos_w = (body_pos_w * masses.unsqueeze(-1)).sum(dim=1) / total_mass
    com_vel_w = (body_lin_vel_w * masses.unsqueeze(-1)).sum(dim=1) / total_mass

    r_rel = body_pos_w - com_pos_w.unsqueeze(1)
    v_rel = body_lin_vel_w - com_vel_w.unsqueeze(1)
    l_orbital = torch.cross(r_rel, v_rel, dim=-1) * masses.unsqueeze(-1)

    rot_matrix = matrix_from_quat(body_quat_w)
    i_world = torch.matmul(rot_matrix, torch.matmul(inertias, rot_matrix.transpose(-1, -2)))
    l_spin = torch.matmul(i_world, body_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    return (l_orbital + l_spin).sum(dim=1)


def _action_jerk(env: ManagerBasedEnv, actions: torch.Tensor) -> torch.Tensor:
    """Second-order action difference, ported from the WBC ``action_jerk_l2``."""
    prev = env._ufo_jerk_prev_action
    prev_prev = env._ufo_jerk_prev_prev_action
    steps = env._ufo_jerk_steps
    if prev is None:
        env._ufo_jerk_prev_action = actions.clone()
        env._ufo_jerk_prev_prev_action = actions.clone()
        env._ufo_jerk_steps = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
        return torch.zeros(actions.shape[0], device=actions.device)

    reset_mask = _reset_mask(env)
    if reset_mask is not None:
        prev[reset_mask] = actions[reset_mask]
        prev_prev[reset_mask] = actions[reset_mask]
        steps[reset_mask] = 0

    jerk = actions - 2.0 * prev + prev_prev
    env._ufo_jerk_prev_prev_action = prev.clone()
    env._ufo_jerk_prev_action = actions.clone()
    steps += 1

    penalty = torch.sum(torch.square(jerk), dim=-1)
    return torch.where(steps >= 2, penalty, torch.zeros_like(penalty))


def _angular_momentum_rate(env: ManagerBasedEnv, l_com: torch.Tensor) -> torch.Tensor:
    """Rate of change of total CoM angular momentum, bounded like the WBC term."""
    prev = env._ufo_l_com_prev
    valid = env._ufo_l_com_valid
    current_finite = torch.isfinite(l_com).all(dim=-1)
    if prev is None:
        env._ufo_l_com_prev = torch.where(current_finite.unsqueeze(-1), l_com, torch.zeros_like(l_com)).clone()
        env._ufo_l_com_valid = current_finite.clone()
        return torch.zeros(l_com.shape[0], dtype=l_com.dtype, device=l_com.device)

    reset_mask = _reset_mask(env)
    valid_rate = valid & current_finite
    if reset_mask is not None:
        valid_rate = valid_rate & ~reset_mask
    rate = (l_com - prev) / env._ufo_step_dt
    penalty = _finite_bounded_penalty(
        torch.sum(torch.square(rate), dim=-1), ANGULAR_MOMENTUM_RATE_MAX_PENALTY
    )
    penalty = torch.where(valid_rate, penalty, torch.zeros_like(penalty))

    env._ufo_l_com_prev = torch.where(current_finite.unsqueeze(-1), l_com, prev).clone()
    env._ufo_l_com_valid = current_finite if reset_mask is None else (current_finite & ~reset_mask)
    return penalty


def zero_reward(env: ManagerBasedEnv) -> torch.Tensor:
    """Constant-zero placeholder: UFO trains on agent-side aux rewards."""
    return torch.zeros(env.num_envs, device=env.device)


def compute_base_aux_rewards(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    cache = env._ufo_aux_cache
    step = env.common_step_counter
    if cache is not None and cache[0] == step:
        return cache[1]

    robot = env.scene["robot"]
    torques = _t(robot.actuators.applied_effort)
    joint_vel = _t(robot.data.joint_vel)
    dof_pos = _t(robot.data.joint_pos)
    actions = env.action_manager.action
    last_actions = env.action_manager.prev_action
    contact_sensor = env.scene["contact_forces"]
    contact_data = contact_sensor.data
    normal_history = _t(contact_data.net_normal_forces_w_history)
    normal_forces = _t(contact_data.net_normal_forces_w)
    peak_forces = _peak_contact_forces(normal_history)
    robot_weight = _t(robot.data.body_mass).sum(dim=-1)

    aux = {}
    aux["penalty_torques"] = torch.sum(torch.square(torques), dim=1)
    aux["penalty_action_rate"] = torch.sum(torch.square(last_actions - actions), dim=1)
    limits = _t(robot.data.soft_joint_pos_limits)
    lower, upper = limits[..., 0], limits[..., 1]
    aux["limits_dof_pos"] = torch.sum((-(dof_pos - lower).clip(max=0.0)) + ((dof_pos - upper).clip(min=0.0)), dim=1)

    feet_idx = env._ufo_feet_ids
    body_idx = env._ufo_undesired_body_ids
    body_velocity = _t(robot.data.body_link_lin_vel_w)
    aux["penalty_body_impact"] = _body_contact_severity(
        peak_forces[:, body_idx], body_velocity[:, body_idx], robot_weight
    )
    ankle_roll_idx = env._ufo_ankle_roll_ids
    aux["penalty_ankle_roll"] = (
        torch.sum(torch.square(dof_pos[:, ankle_roll_idx]), dim=1)
        if ankle_roll_idx.numel()
        else torch.zeros(dof_pos.shape[0], device=dof_pos.device, dtype=dof_pos.dtype)
    )
    aux["feet_stumble"] = _feet_stumble_penalty(normal_history[:, :, feet_idx, :])
    aux["height_gated_undesired_contacts"] = _height_gated_undesired_contacts(
        normal_history,
        _t(robot.data.body_link_pos_w)[..., 2],
        env._ufo_undesired_body_ids,
    )

    aux["penalty_slippage"] = _tangential_contact_speed(
        normal_forces[:, feet_idx], body_velocity[:, feet_idx], robot_weight
    )

    # WBC-shared regularization terms.
    joint_acc = _t(robot.data.joint_acc)
    aux["joint_acc_l2"] = torch.sum(torch.square(joint_acc), dim=1)
    aux["joint_velocity_l2"] = torch.sum(torch.square(joint_vel), dim=1)
    aux["joint_energy"] = torch.sum(torch.abs(torques * joint_vel), dim=1)

    actuator = robot.actuators["all_joints"]
    joint_vel_act = joint_vel[:, actuator.joint_indices]
    rated_velocity = _t(actuator.rated_velocity)
    rated_effort = _t(actuator.rated_effort)
    excess_velocity = torch.relu(joint_vel_act.abs() / rated_velocity - 1.0)
    aux["joint_rated_velocity"] = torch.sum(excess_velocity.square(), dim=1)
    excess_effort = torch.relu(_t(actuator.applied_effort).abs() / rated_effort - 1.0)
    aux["joint_rated_effort"] = torch.sum(excess_effort.square(), dim=1)

    aux["action_jerk_l2"] = _action_jerk(env, actions)

    l_com = _com_angular_momentum(env)
    aux["yaw_angular_momentum_l2"] = _finite_bounded_penalty(
        torch.square(l_com[:, 2]), YAW_ANGULAR_MOMENTUM_MAX_PENALTY
    )
    aux["angular_momentum_rate_l2"] = _angular_momentum_rate(env, l_com)

    env._ufo_aux_cache = (step, aux)
    return aux


def _reference_velocity_yaw(env: ManagerBasedEnv):
    command = env.command_manager.get_term("motion")
    reference = command.get_motion_state()
    robot = env.scene["robot"]
    quat = _t(robot.data.body_quat_w)[:, 0]
    root_vel = _t(robot.data.body_link_lin_vel_w)[:, 0]
    root_ang_vel = _t(robot.data.body_link_ang_vel_w)[:, 0]
    from isaaclab.utils.math import quat_apply_inverse

    target_xy = quat_apply_inverse(quat, reference["root_vel"])[:, :2]
    actual_xy = quat_apply_inverse(quat, root_vel)[:, :2]
    target = torch.cat((target_xy, reference["root_ang_vel"][:, 2:3]), dim=-1)
    actual = torch.cat((actual_xy, root_ang_vel[:, 2:3]), dim=-1)
    return actual, target


def heading_tracking(env: ManagerBasedEnv) -> torch.Tensor:
    from .pufo_observations import heading_context

    return heading_context(env)[:, 0]


def velocity_tracking(env: ManagerBasedEnv, error_scale_mps: float = 0.5) -> torch.Tensor:
    actual, target = _reference_velocity_yaw(env)
    error = (actual[:, :2] - target[:, :2]) / float(error_scale_mps)
    scaled_norm = torch.linalg.vector_norm(error, dim=-1)
    return F.huber_loss(scaled_norm, torch.zeros_like(scaled_norm), reduction="none", delta=1.0)


def compute_aux_rewards(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    aux = dict(compute_base_aux_rewards(env))
    aux["heading_tracking"] = heading_tracking(env)
    aux["velocity_tracking"] = velocity_tracking(
        env,
        float(getattr(env, "_pufo_velocity_error_scale_mps", 0.5)),
    )
    return aux
