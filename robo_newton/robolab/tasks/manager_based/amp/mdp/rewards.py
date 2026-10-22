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
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


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


def link_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_tolerance: float = 0.0,
) -> torch.Tensor:
    """Penalize roll and pitch of one link with a forward-lean deadzone.

    The projected gravity x component is positive for forward lean in the
    standard x-forward, z-up link frame. Unlike parkour's version, this AMP
    variant intentionally has no terrain-specific tolerance or terrain lookup.
    """
    if not math.isfinite(forward_tolerance) or not 0.0 <= forward_tolerance < math.pi / 2:
        raise ValueError("forward_tolerance must be finite and in [0, pi/2).")
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("link_orientation requires exactly one body link.")

    asset: Articulation = env.scene[asset_cfg.name]
    link_quat = _as_torch(asset.data.body_quat_w)[:, asset_cfg.body_ids[0], :]
    gravity_w = _as_torch(asset.data.GRAVITY_VEC_W)
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.unsqueeze(0).expand(link_quat.shape[0], -1)
    projected_gravity = math_utils.quat_apply_inverse(link_quat, gravity_w)

    forward = torch.clamp(projected_gravity[:, 0], min=0.0)
    backward = torch.clamp(-projected_gravity[:, 0], min=0.0)
    forward_excess = torch.clamp(forward - math.sin(forward_tolerance), min=0.0)
    roll = projected_gravity[:, 1]
    return torch.square(forward_excess) + torch.square(roll) + 16.0 * torch.square(backward)


def link_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame xy angular velocity of the selected link(s)."""
    asset: Articulation = env.scene[asset_cfg.name]
    link_ang_vel_xy = _as_torch(asset.data.body_ang_vel_w)[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(link_ang_vel_xy), dim=(1, 2))



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
    convention used for the forward-lean gate:
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


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    ang_vel_gate: float | None = None,
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one.

    If ``command_name`` and ``ang_vel_gate`` are set, the penalty is zero when
    the commanded yaw rate magnitude is at least ``ang_vel_gate``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        _as_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
        - _as_torch(asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    )
    penalty = torch.sum(torch.abs(angle), dim=1)
    if command_name is None or ang_vel_gate is None:
        return penalty
    if ang_vel_gate < 0.0:
        raise ValueError("ang_vel_gate must be non-negative")
    yaw_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return penalty * (yaw_cmd < float(ang_vel_gate)).to(penalty.dtype)


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards air/contact mode time up to a threshold during single stance.
    Double support and double flight receive zero reward.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
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
    reward *= (torch.linalg.norm(command[:, :3], dim=1) > 0.1).float()
    return reward


def _in_place_yaw_gate(
    env: ManagerBasedRLEnv, command_name: str, lin_vel_max: float, ang_vel_min: float
) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd[:, :2], dim=1) < float(lin_vel_max)) & (
        torch.abs(cmd[:, 2]) >= float(ang_vel_min)
    )


def yaw_turn_foot_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    lin_vel_max: float = 0.3,
    ang_vel_min: float = 0.3,
    target_height: float = 0.06,
    target_air_time: float = 0.3,
    height_std: float = 0.03,
    air_std: float = 0.15,
) -> torch.Tensor:
    """In-place yaw: two-sided exp peaked at 6 cm and 0.3 s.

    ``exp(-((x-t)/s)^2)`` is 1 at the target. ``s = t/2`` makes the floor and
    the overshoot about 0.018 instead of 0.37, so shuffling cannot farm the
    term. Height is relative to the other foot. Double flight is zero.
    """
    if target_height <= 0.0 or target_air_time <= 0.0 or height_std <= 0.0 or air_std <= 0.0:
        raise ValueError("yaw-turn height/air targets and stds must be positive")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    foot_z = _as_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2]
    clearance = (foot_z - foot_z.flip(1)).clamp(min=0.0)
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    height_rew = torch.exp(-((clearance - target_height) / height_std).square())
    air_rew = torch.exp(-((air_time - target_air_time) / air_std).square())
    reward = (height_rew * air_rew).max(dim=1).values
    return reward * in_contact.any(dim=1).float() * _in_place_yaw_gate(
        env, command_name, lin_vel_max, ang_vel_min
    ).float()


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

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    joint_pos_weight: float = 0.6,
    joint_vel_weight: float = 0.05,
    body_vel_weight: float = 0.35,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene["robot"]
    cmd = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) + torch.abs(env.command_manager.get_command(command_name)[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = joint_pos_weight *torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    vel_reward = joint_vel_weight * torch.sum(torch.abs(asset.data.joint_vel), dim=1)
    body_vel_reward = body_vel_weight * body_vel
    reward = torch.where(
        cmd > 0.1,
        0.0,
        pos_reward + vel_reward + body_vel_reward,
    )
    return reward

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
    command_name: str | None = "base_velocity",
    speed_gate: float | None = 1.5,
) -> torch.Tensor:
    """Penalize contact-foot vertical velocity and acceleration.

    This is an impact proxy, not a sound-pressure measurement. Sensor and robot
    body indices are resolved separately. When ``speed_gate`` is set, the term
    is disabled at or above that commanded XY speed.
    """
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be nonnegative")
    if speed_gate is not None and (not math.isfinite(speed_gate) or speed_gate < 0.0):
        raise ValueError("speed_gate must be finite and nonnegative")
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_z = _as_torch(asset.data.body_vel_w)[:, asset_cfg.body_ids, 2]
    acceleration_z = _as_torch(asset.data.body_acc_w)[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = _as_torch(contact_sensor.data.net_forces_w)[:, sensor_cfg.body_ids, 2]
    if velocity_z.shape != acceleration_z.shape or velocity_z.shape != force_z.shape:
        raise ValueError("sound_suppression requires matching robot and sensor body selections")
    in_contact = torch.abs(force_z) > contact_threshold
    motion = velocity_z.square() + acceleration_z.square()
    penalty = torch.where(in_contact, motion, 0.0).sum(dim=1)
    if command_name is not None and speed_gate is not None:
        command = _as_torch(env.command_manager.get_command(command_name))
        commanded_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        penalty = penalty * (commanded_speed < speed_gate).to(penalty.dtype)
    return penalty


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)
