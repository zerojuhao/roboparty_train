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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_torch(value) -> torch.Tensor:
    """Return a torch view for Newton proxy arrays and ordinary tensors."""
    return value.torch if hasattr(value, "torch") else value


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_braking_reference: bool = False,
    braking_reward_scale: float = 1.0,
    braking_min_speed: float = 0.2,
    braking_max_command: float = 0.1,
) -> torch.Tensor:
    """Reward xy tracking with optional lower pressure during an abrupt stop."""
    if not math.isfinite(braking_reward_scale) or braking_reward_scale <= 0.0:
        raise ValueError("braking_reward_scale must be finite and positive.")
    if braking_min_speed < 0.0 or braking_max_command < 0.0:
        raise ValueError("braking speed and command thresholds must be non-negative.")
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = (env.command_manager.get_term(command_name).tracking_command
               if use_braking_reference else env.command_manager.get_command(command_name))
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(
            command[:, :2] - _as_torch(asset.data.root_lin_vel_b)[:, :2]
        ),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-_as_torch(asset.data.projected_gravity_b)[:, 2], 0, 0.7) / 0.7
    if braking_reward_scale != 1.0:
        actual_speed = torch.linalg.vector_norm(_as_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
        braking = (
            torch.linalg.vector_norm(command[:, :2], dim=1) <= braking_max_command
        ) & (actual_speed > braking_min_speed)
        reward = torch.where(braking, reward * braking_reward_scale, reward)
    return reward


def lateral_command_offset_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_yaw_command: float = 0.1,
    min_linear_command: float = 0.1,
    std: float = 0.5,
) -> torch.Tensor:
    """Penalize accumulated lateral displacement with a smooth Huber penalty.

    The command generator stores the pose at the start of each command segment. The
    displacement is measured in that pose's heading frame, so the term is invariant
    to the environment's initial yaw. The reward is evaluated every step, while the
    displacement itself accumulates from the segment start. Turning commands are
    excluded because their lateral displacement is intentional. The Huber form is
    quadratic near zero for smooth shaping, then grows linearly after ``std`` so
    large offsets remain distinguishable instead of saturating.
    """
    if std <= 0:
        raise ValueError("std must be positive")

    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    straight_motion = torch.abs(command[:, 2]) <= max_yaw_command
    linear_speed = torch.linalg.norm(command[:, :2], dim=1)
    moving = linear_speed >= min_linear_command
    scaled_offset = torch.abs(command_term.lateral_offset_error) / std
    offset_penalty = torch.where(
        scaled_offset <= 1.0,
        0.5 * torch.square(scaled_offset),
        scaled_offset - 0.5,
    )
    return offset_penalty * (straight_motion & moving).float()


def turn_command_offset_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_yaw_command: float = 0.1,
    min_linear_command: float = 0.1,
) -> torch.Tensor:
    """Penalize planar drift during in-place yaw commands.

    A pure yaw segment has zero desired planar displacement. Measuring both
    heading-frame axes prevents an in-place turn from slowly walking away,
    while the command mask keeps this term separate from translation tracking.
    """
    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    turning = torch.abs(command[:, 2]) > max_yaw_command
    stationary = torch.linalg.norm(command[:, :2], dim=1) < min_linear_command
    return torch.sum(torch.square(command_term.planar_offset_error), dim=1) * (turning & stationary).float()


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - _as_torch(asset.data.root_ang_vel_b)[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-_as_torch(asset.data.projected_gravity_b)[:, 2], 0, 0.7) / 0.7
    return reward


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)



def asymmetric_pitch_roll_penalty(
    projected_gravity: torch.Tensor,
    forward_tolerance: float | torch.Tensor = 0.0,
    backward_scale: float = 16.0,
) -> torch.Tensor:
    """Penalize tilt with a soft forward-lean deadzone (parkour-style).

    Positive projected gravity on body x is forward lean. Forward tilt is
    penalized only beyond ``forward_tolerance`` (radians); backward tilt is
    scaled by ``backward_scale`` (parkour default 16); roll is unchanged.
    Inside the forward deadzone the forward term is zero.
    """
    tolerance = torch.as_tensor(
        forward_tolerance,
        dtype=projected_gravity.dtype,
        device=projected_gravity.device,
    )
    if torch.any((tolerance < 0.0) | (tolerance >= torch.pi / 2)):
        raise ValueError("forward_tolerance must be in [0, pi/2).")
    forward = torch.clamp(projected_gravity[..., 0], min=0.0)
    backward = torch.clamp(-projected_gravity[..., 0], min=0.0)
    forward_excess = torch.clamp(forward - torch.sin(tolerance), min=0.0)
    roll = projected_gravity[..., 1]
    return (
        torch.square(forward_excess)
        + torch.square(roll)
        + float(backward_scale) * torch.square(backward)
    )


def lean_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_tolerance: float = 0.0,
    backward_scale: float = 16.0,
) -> torch.Tensor:
    """Penalize configured body lean with a soft forward deadzone.

    Uses the configured bodies' projected gravity. Within ``forward_tolerance``
    (radians) forward lean is free; excess forward lean, roll, and especially
    backward lean (``backward_scale``) are squared. When multiple bodies are
    configured, the largest per-body penalty is returned so adding a body does not
    silently multiply the reward weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = _as_torch(asset.data.body_quat_w)[:, asset_cfg.body_ids]
    gravity_w = torch.zeros_like(body_quat_w[..., :3])
    gravity_w[..., 2] = -1.0
    projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_w)
    penalty = asymmetric_pitch_roll_penalty(
        projected_gravity,
        forward_tolerance=forward_tolerance,
        backward_scale=backward_scale,
    )
    return penalty.amax(dim=-1) if penalty.ndim > 1 else penalty




def shoulder_roll_excess_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_shoulder_roll_joint", "right_shoulder_roll_joint"]
    ),
    deadzone: float = 0.35,
) -> torch.Tensor:
    """Penalize |shoulder_roll| beyond a soft deadzone (demo ~+/-15-21 deg ~ 0.26-0.37 rad).

    For each shoulder roll joint:
        excess_i = relu(|q_i| - deadzone)
    Returns sum_i excess_i^2 (soft L2 outside the band). Inside the deadzone
    the term is zero so natural AMP-style arm carriage is not taxed. Reversible:
    set weight=0 or remove the RewTerm.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(q) - float(deadzone), min=0.0)
    return torch.sum(torch.square(excess), dim=1)




def high_speed_double_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    speed_gate: float = 1.5,
    max_flight_time: float = 0.30,
    max_forward_lean: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward double-flight (both feet airborne) at high commanded XY speed.

    If ``max_forward_lean`` (radians) is set, the reward is zeroed whenever the
    configured body's forward lean exceeds it. Forward lean uses the same
    convention as ``lean_orientation_l2`` / ``asymmetric_pitch_roll_penalty``:
    configured-body projected gravity ``[:, 0] > 0`` is forward lean, compared against
    ``sin(max_forward_lean)``. ``None`` disables the gate (C20 behaviour).

    When BOTH feet are airborne AND |lin_vel_cmd_xy| > speed_gate:
        reward = min(min(air_L, air_R), max_flight_time)
    else 0. ``min(air_L, air_R)`` is continuous double-air duration while both
    feet remain off the ground. Cap prevents unbounded flight credit.
    Does not change feet_air_time_positive_biped. Reversible: weight=0.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    both_air = torch.all(air_time > 0.0, dim=1)
    double_air = torch.min(air_time, dim=1).values
    reward = torch.where(
        both_air,
        torch.clamp(double_air, max=float(max_flight_time)),
        torch.zeros_like(double_air),
    )
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed > float(speed_gate)).float()
    # C22_LEAN_GATE_15: zero flight credit when forward lean > max_forward_lean (C22 default 15°).
    if max_forward_lean is not None:
        asset: Articulation = env.scene[asset_cfg.name]
        body_quat_w = _as_torch(asset.data.body_quat_w)[:, asset_cfg.body_ids]
        gravity_w = torch.zeros_like(body_quat_w[..., :3])
        gravity_w[..., 2] = -1.0
        projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_w)[:, 0]
        forward = torch.clamp(projected_gravity[:, 0], min=0.0)
        lean_ok = forward <= math.sin(float(max_forward_lean))
        reward = reward * lean_ok.float()
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        _as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
        - _as_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(angle), dim=1)


def joint_position_range_l2(
    env: ManagerBasedRLEnv,
    lower_bounds: tuple[float, ...],
    upper_bounds: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    min_linear_command: float = 1.5,
    angle_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize absolute angles outside bounds, optionally above a commanded XY speed."""
    if command_name is not None and (not math.isfinite(min_linear_command) or min_linear_command < 0):
        raise ValueError("min_linear_command must be finite and nonnegative")
    if not math.isfinite(angle_scale) or angle_scale <= 0.0:
        raise ValueError("angle_scale must be finite and positive")
    asset: Articulation = env.scene[asset_cfg.name]
    positions = _as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    if len(lower_bounds) != positions.shape[1] or len(upper_bounds) != positions.shape[1]:
        raise ValueError("Joint bounds must match the number of selected joints")
    if any(
        not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
        for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
    ):
        raise ValueError("Joint bounds must be finite with lower < upper")
    lower = positions.new_tensor(lower_bounds)
    upper = positions.new_tensor(upper_bounds)
    excess = (lower - positions).clamp_min(0.0) + (positions - upper).clamp_min(0.0)
    penalty = (excess / angle_scale).square().sum(dim=1)
    if command_name is not None:
        command = _as_torch(env.command_manager.get_command(command_name))
        active = torch.linalg.norm(command[:, :2], dim=1) > min_linear_command
        penalty *= active.to(penalty.dtype)
    return penalty


def paired_joints_mean_deviation_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the shared offset of paired joints from their default positions.

    This keeps symmetric paired joints centered around their nominal pose while still allowing
    anti-phase motion, such as normal left/right arm swing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(torch.mean(joint_deviation, dim=1))


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_torque_saturation_l2(
    env: ManagerBasedRLEnv,
    limit_ratio: float = 0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_substep_peak: bool = False,
) -> torch.Tensor:
    """Penalize applied torque only after it exceeds a fraction of its limit.

    Normalize excess by the remaining margin to the effort limit. The default
    averages the current joints; use_substep_peak instead takes the worst joint
    over all physics samples. A joint at its limit contributes one before
    reduction; at or below the threshold it contributes zero.
    """
    if not 0.0 < limit_ratio < 1.0:
        raise ValueError("limit_ratio must be in (0, 1).")
    if use_substep_peak:
        if asset_cfg.name != "robot":
            raise ValueError("Motor peak tracking is configured for robot only.")
        ratio = env.motor_load.torque_ratio_peak[:, asset_cfg.joint_ids]
        excess = torch.relu(ratio - limit_ratio) / (1.0 - limit_ratio)
        return excess.square().amax(dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    applied_torque = torch.abs(_as_torch(asset.data.applied_torque)[:, asset_cfg.joint_ids])
    effort_limits = _as_torch(asset.data.joint_effort_limits)[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    normalized_excess = torch.relu(applied_torque / effort_limits - float(limit_ratio)) / (1.0 - limit_ratio)
    return torch.mean(torch.square(normalized_excess), dim=1)


def _is_braking(env, command_name, min_speed, max_command):
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(_as_torch(env.scene["robot"].data.root_lin_vel_b)[:, :2], dim=1)
    return (torch.linalg.vector_norm(command[:, :2], dim=1) < max_command) & (speed > min_speed)


def braking_power_peak_l2(
    env: ManagerBasedRLEnv, command_name: str, power_threshold: float = 2500.0,
    min_speed: float = 0.2, max_command: float = 0.1,
) -> torch.Tensor:
    """Penalize excessive mechanical absorption during zero-command braking.

    Sum negative joint mechanical powers without cancelling them against
    motoring joints. This is a regeneration proxy, not a DC bus voltage model.
    """
    if not math.isfinite(power_threshold) or power_threshold <= 0:
        raise ValueError("power_threshold must be finite and positive.")
    excess = torch.relu(env.motor_load.braking_power_peak / power_threshold - 1.0)
    return excess.square() * _is_braking(env, command_name, min_speed, max_command)


def braking_torque_step_l2(
    env: ManagerBasedRLEnv, command_name: str, min_speed: float = 0.2, max_command: float = 0.1,
) -> torch.Tensor:
    """Penalize the largest normalized physics-step torque jump during braking."""
    return env.motor_load.torque_delta_ratio_peak.square() * _is_braking(env, command_name, min_speed, max_command)


def braking_load_metrics(
    env: ManagerBasedRLEnv, command_name: str, power_threshold: float = 2500.0,
    min_speed: float = 0.2, max_command: float = 0.1,
    power_weight: float = -0.02,
) -> dict[str, torch.Tensor]:
    """Retain braking samples so the logger weights conditional means by count.

    Power samples are control-interval peaks in watts, not electrical bus power.
    Empty conditional metrics are omitted rather than logged as zero or NaN.
    """
    braking = _is_braking(env, command_name, min_speed, max_command)
    metrics = {"Braking/active_fraction": braking.float()}
    power = env.motor_load.braking_power_peak[braking]
    if power.numel():
        excess = torch.relu(power / power_threshold - 1.0).square()
        metrics.update({
            "Braking/power_peak_w": power,
            "Braking/power_excess_fraction": (power > power_threshold).float(),
            "Braking/power_penalty": excess * power_weight,
            "Braking/torque_delta_ratio_peak": env.motor_load.torque_delta_ratio_peak[braking],
        })
    return metrics


def link_close_y_gauss(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
) -> torch.Tensor:
    """Parkour's minimum lateral link separation penalty, in the heading frame."""
    if threshold <= 0.0 or std <= 0.0:
        raise ValueError("threshold and std must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    positions = _as_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids]
    if positions.shape[1] != 2:
        raise ValueError("link_close_y_gauss requires exactly two bodies")
    heading = _as_torch(asset.data.heading_w)
    delta_xy = positions[:, 0, :2] - positions[:, 1, :2]
    distance_y = torch.abs(
        -torch.sin(heading) * delta_xy[:, 0] + torch.cos(heading) * delta_xy[:, 1]
    )
    return 1.0 - torch.exp(-torch.clamp(threshold - distance_y, min=0.0) / std**2)


def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = _as_torch(asset.data.root_quat_w).unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = _as_torch(asset.data.root_pos_w).unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Only reward air time exceeding the threshold (prevents negative penalties)
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(positive_air * first_contact.float(), dim=1)
    # no reward for zero command
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    turn_reward_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards air/contact mode time up to a threshold during single stance.
    Double support and double flight receive zero reward.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)
    linear_command = torch.linalg.norm(command[:, :2], dim=1) > 0.1
    active_command = torch.linalg.norm(command[:, :3], dim=1) > 0.1
    # Keep a small part of the stepping incentive for turns, while avoiding the
    # full single-support pressure that produced high-knee hopping turns.
    turn_command = active_command & ~linear_command
    command_scale = torch.where(turn_command, turn_reward_scale, active_command.float())
    reward *= command_scale
    return reward


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        _as_torch(asset.data.GRAVITY_VEC_W).unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_command_threshold: float = 0.1,
    velocity_threshold: float = 0.1,
    yaw_velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize default-pose offsets only under a stationary command and state.

    Planar command/speed and yaw command/speed must each be strictly below their
    thresholds, leaving braking and turning unconstrained until the robot stops.
    """
    for name, value in (
        ("command_threshold", command_threshold),
        ("yaw_command_threshold", yaw_command_threshold),
        ("velocity_threshold", velocity_threshold),
        ("yaw_velocity_threshold", yaw_velocity_threshold),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
        torch.abs(command[:, 2]) < yaw_command_threshold
    )
    linear_speed = torch.linalg.vector_norm(_as_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
    yaw_speed = torch.abs(_as_torch(asset.data.root_ang_vel_b)[:, 2])
    stopped = (linear_speed < velocity_threshold) & (yaw_speed < yaw_velocity_threshold)
    deviation = (
        _as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
        - _as_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(deviation), dim=1) * (standing & stopped)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Sum world-frame horizontal foot speeds while the feet are in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = _as_torch(contact_sensor.data.net_normal_forces_w)[:, sensor_cfg.body_ids, :]
    contacts = forces.norm(dim=-1) > contact_threshold
    foot_velocity = _as_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    return (foot_velocity.norm(dim=-1) * contacts).sum(dim=-1)

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Parkour's contact-foot vertical velocity/acceleration penalty.

    This is an impact proxy, not a sound-pressure measurement. Sensor and robot
    body indices are resolved separately. No command gate excludes running or
    stopping, and airborne-foot motion is not penalized.
    """
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be nonnegative")
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_z = _as_torch(asset.data.body_vel_w)[:, asset_cfg.body_ids, 2]
    acceleration_z = _as_torch(asset.data.body_acc_w)[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = _as_torch(contact_sensor.data.net_forces_w)[:, sensor_cfg.body_ids, 2]
    if velocity_z.shape != acceleration_z.shape or velocity_z.shape != force_z.shape:
        raise ValueError("sound_suppression requires matching robot and sensor body selections")
    in_contact = torch.abs(force_z) > contact_threshold
    motion = velocity_z.square() + acceleration_z.square()
    return torch.where(in_contact, motion, 0.0).sum(dim=1)


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Compute per-foot acceleration penalty for sound suppression.

    Penalize large vertical (z) accelerations when a foot is in contact with the ground.
    """

    asset = env.scene["robot"]

    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    acc_penalty = (foot_acc_z ** 2) * in_contact.float()
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)
