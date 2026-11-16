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
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
) -> torch.Tensor:
    """Sum roll/pitch penalties across links, with a stronger backward penalty.

    The projected gravity x component is positive for forward lean in the
    standard x-forward, z-up link frame. Unlike parkour's version, this AMP
    variant intentionally has no terrain-specific tolerance or terrain lookup.
    Unit gravity makes costs independent of gravity magnitude. Selected links
    contribute equally; backward tilt costs three times equal forward tilt.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    if link_quat.shape[1] == 0:
        raise ValueError("link_orientation requires at least one body link.")
    gravity_w = asset.data.GRAVITY_VEC_W.torch
    if gravity_w.ndim == 1:
        gravity_w = gravity_w.unsqueeze(0).expand(link_quat.shape[0], -1)
    gravity_w = gravity_w / gravity_w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gravity_w = gravity_w.unsqueeze(1).expand(-1, link_quat.shape[1], -1)
    projected_gravity = math_utils.quat_apply_inverse(link_quat, gravity_w)

    forward = torch.clamp(projected_gravity[..., 0], min=0.0)
    backward = torch.clamp(-projected_gravity[..., 0], min=0.0)
    roll = projected_gravity[..., 1]
    penalty = torch.square(forward) + torch.square(roll) + 3.0 * torch.square(backward)
    return penalty.sum(dim=1)


def link_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame xy angular velocity of the selected link(s)."""
    asset: Articulation = env.scene[asset_cfg.name]
    link_ang_vel_xy = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return link_ang_vel_xy.square().sum(dim=(1, 2))



def high_speed_double_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    speed_gate: float = 1.5,
    target_flight_time: float = 0.08,
    full_speed: float = 2.0,
    max_forward_lean: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Credit completed double flight once at the first foot's landing.

    Above ``max_forward_lean`` (radians), reward decreases linearly with forward
    gravity projection from sin(tolerance) to 1. Positive projected gravity X
    denotes forward lean; ``None`` disables the lean attenuation.

    Sensor timers reconstruct the overlap of the two latest air intervals.
    Normalize by target duration and cap at one; divide by control dt so the
    RewardManager integral credits one event, independently of control rate.
    """
    if not math.isfinite(target_flight_time) or target_flight_time <= 0:
        raise ValueError("target_flight_time must be finite and positive")
    if not math.isfinite(full_speed) or full_speed <= speed_gate:
        raise ValueError("full_speed must exceed speed_gate")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
    # The earliest landing is the foot with the longest current contact time.
    # Requiring it to be new prevents paying again when the second foot lands.
    since_takeoff = torch.where(contact_time > 0, last_air + contact_time, air_time)
    since_landing, first_foot = contact_time.max(dim=1)
    landed_now = first_contact.gather(1, first_foot.unsqueeze(1)).squeeze(1)
    duration = (since_takeoff.min(dim=1).values - since_landing).clamp_min(0.0)
    reward = (duration / target_flight_time).clamp(max=1.0) * landed_now / env.step_dt
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    reward = reward * ((cmd_speed - speed_gate) / (full_speed - speed_gate)).clamp(0.0, 1.0)
    if max_forward_lean is not None:
        if not math.isfinite(max_forward_lean) or not 0.0 <= max_forward_lean < math.pi / 2:
            raise ValueError("max_forward_lean must be in [0, pi/2)")
        asset: Articulation = env.scene[asset_cfg.name]
        body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        gravity_w = torch.zeros_like(body_quat_w[..., :3])
        gravity_w[..., 2] = -1.0
        projected_gravity = math_utils.quat_apply_inverse(body_quat_w, gravity_w)[:, 0]
        tolerance = math.sin(max_forward_lean)
        lean_scale = ((1.0 - projected_gravity[:, 0]) / (1.0 - tolerance)).clamp(0.0, 1.0)
        reward = reward * lean_scale
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
    yaw_reference_term: str | None = None,
    std: float | None = None,
) -> torch.Tensor:
    """Deviation from default pose; scale down when the yaw target is large."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    penalty = (
        (1. - torch.exp(-angle.square() / std**2)).sum(dim=1)
        if std is not None else angle.abs().sum(dim=1)
    )
    if yaw_reference_term is not None:
        yaw = env.reward_manager.get_term_cfg(yaw_reference_term).func.target_yaw_rate
    elif command_name is not None:
        yaw = env.command_manager.get_command(command_name)[:, 2]
    else:
        return penalty
    return penalty * torch.exp(-(yaw / ang_vel_gate).square())


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


class action_second_difference_l2(ManagerTermBase):
    """Penalize alternating targets using three valid policy actions per episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros_like(env.action_manager.action)
        self._previous_previous = torch.zeros_like(self._previous)
        self._history_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._cost = torch.zeros(self.num_envs, device=self.device)
        self._last_step = -1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = 0.0
        self._previous_previous[ids] = 0.0
        self._history_count[ids] = 0
        self._cost[ids] = 0.0

    def __call__(self, env):
        # Diagnostics may read the term again without advancing the policy step.
        if self._last_step != env.common_step_counter:
            action = env.action_manager.action
            second_difference = action - 2.0 * self._previous + self._previous_previous
            self._cost.copy_(torch.sum(second_difference.square(), dim=1))
            self._cost.masked_fill_(self._history_count < 2, 0.0)
            self._previous_previous.copy_(self._previous)
            self._previous.copy_(action)
            self._history_count.add_(1).clamp_(max=2)
            self._last_step = env.common_step_counter
        return self._cost.clone()


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
    applied_torque = torch.abs(asset.data.applied_torque.torch[:, asset_cfg.joint_ids])
    effort_limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
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
    positions = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    if positions.shape[1] != 2:
        raise ValueError("link_close_y_gauss requires exactly two bodies")
    heading = asset.data.heading_w.torch
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
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
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
    target_air_time: float = 0.3,
    walking_speed_range: tuple[float, float] = (0.05, 1.2),
    activity: torch.Tensor | None = None,
    linear_duration: bool = False,
    ) -> torch.Tensor:
    """Credit a completed walking or turning step while the other foot supports.

    ``threshold`` gates turns; ``walking_speed_range`` gates planar target speed.
    Speeds above the walking range receive no walking credit. ``target_air_time``
    caps the credited swing duration. The squared duration approximates the
    integral of the old time-ramp reward; divide by dt because RewardManager
    multiplies all terms by dt. Holding a foot aloft earns no ongoing credit.
    ``activity`` optionally replaces command gating with a per-environment load gate.
    ``linear_duration`` gives short completed steps linear credit, retaining the cap.
    """
    if not math.isfinite(target_air_time) or target_air_time <= 0:
        raise ValueError("target_air_time must be finite and positive")
    walk_min, walk_max = walking_speed_range
    if not all(math.isfinite(v) for v in walking_speed_range) or not 0 < walk_min < walk_max:
        raise ValueError("walking_speed_range must contain positive increasing bounds")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # Exclude simultaneous landings and require established contralateral support.
    support = (contact_time > env.step_dt) & ~first_contact
    continuous_support = contact_time.flip(dims=[1]) >= last_air_time + contact_time
    valid_landing = first_contact & support.flip(dims=[1]) & continuous_support
    duration = last_air_time.clamp(min=0.0, max=target_air_time)
    duration_credit = target_air_time * duration if linear_duration else duration.square()
    reward = (0.5 * duration_credit * valid_landing).sum(dim=1) / env.step_dt
    if activity is not None:
        return reward * activity
    cmd = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(cmd[:, :2], dim=1)
    walking = (speed >= walk_min) & (speed <= walk_max)
    turn_in_place = (speed < threshold) & (
        torch.abs(cmd[:, 2]) > threshold
    )
    return reward * (walking | turn_in_place).float()


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot tilt while in contact, using ||g_xy|| like Instinct feet_orientation_contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > contact_force_threshold
    feet_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    gravity = asset.data.GRAVITY_VEC_W.torch
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(feet_quat.shape[0], -1)
    gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gravity = gravity.unsqueeze(1).expand(-1, feet_quat.shape[1], -1)
    tilt = math_utils.quat_apply_inverse(feet_quat, gravity)[..., :2]
    return (tilt.norm(dim=-1) * in_contact.float()).sum(-1)

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
    """Sum world-frame foot XY linear speed + yaw angular speed while in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]
    contacts = forces.norm(dim=-1) > contact_threshold
    v_lin_xy = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    omega_yaw = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, 2]
    penalty = v_lin_xy.norm(dim=-1) + omega_yaw.abs()
    return (penalty * contacts).sum(dim=-1)

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
    """Penalize squared vertical foot-link speed while in contact."""
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be nonnegative")
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_z = asset.data.body_vel_w.torch[:, asset_cfg.body_ids, 2]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2]
    if velocity_z.shape != force_z.shape:
        raise ValueError("sound_suppression requires matching robot and sensor body selections")
    in_contact = torch.abs(force_z) > contact_threshold
    return torch.where(in_contact, velocity_z.square(), 0.0).sum(dim=1)


def feet_contact_force_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 2.0,
    command_name: str = "base_velocity",
    low_force_threshold: float = 1.2,
    low_speed: float = 1.2,
    high_speed: float = 2.0,
    peak_weight: float = 1.0,
    mean_weight: float = 0.0,
    rising_weight: float = 0.0,
    in_place_turn_weights: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Penalize vertical foot load above a body-weight threshold, sampled every physics substep.

    The threshold blends from ``low_force_threshold`` to ``force_threshold`` (body weights) over
    commanded XY speeds ``low_speed`` to ``high_speed``. The squared excess, summed over feet, is
    charged as its worst substep (peak), its interval mean (sustained overload) and its sharpest
    substep-to-substep increase (touchdown spike). ``in_place_turn_weights`` replaces the three
    weights while turning in place.
    """
    command = env.command_manager.get_command(command_name)
    speed = command[:, :2].norm(dim=1)
    blend = ((speed - low_speed) / (high_speed - low_speed)).clamp(0.0, 1.0)
    threshold = low_force_threshold + (force_threshold - low_force_threshold) * blend.square() * (3.0 - 2.0 * blend)
    asset: Articulation = env.scene[asset_cfg.name]
    body_weight = asset.data.body_mass.torch.sum(dim=1) * abs(env.cfg.sim.gravity[2])
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Newest first. The sample preceding this control interval is only the baseline of the first rise.
    substeps = env.cfg.decimation
    force_z = sensor.data.net_forces_w_history.torch[:, :substeps + 1, sensor_cfg.body_ids, 2].clamp_min(0.0)
    excess = (force_z / body_weight[:, None, None] - threshold[:, None, None]).clamp_min(0.0)
    overload = excess[:, :substeps].square().sum(dim=2)
    peak, mean = overload.amax(dim=1), overload.mean(dim=1)
    rising = (excess[:, :-1] - excess[:, 1:]).clamp_min(0.0).square().sum(dim=2).amax(dim=1)
    cost = peak_weight * peak + mean_weight * mean + rising_weight * rising
    if in_place_turn_weights is not None:
        turn_peak, turn_mean, turn_rising = in_place_turn_weights
        turning = (speed < 0.3) & (command[:, 2].abs() > 0.05)
        cost = torch.where(turning, turn_peak * peak + turn_mean * mean + turn_rising * rising, cost)
    return cost


def feet_impact_velocity_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    downward_speed_threshold: float = 0.15,
    per_event: bool = False,
) -> torch.Tensor:
    """Penalize residual downward speed on new foot contact in every motion mode.

    Velocity is read at the control boundary after physics, not before impact.
    The force-history penalty remains necessary to capture substep collisions.
    ``per_event`` divides by dt so each touchdown is charged once at full weight,
    matching the per-landing accounting of the air-time credit.
    """
    if not math.isfinite(downward_speed_threshold) or downward_speed_threshold < 0:
        raise ValueError("downward_speed_threshold must be finite and non-negative")
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
    downward_speed = (
        -asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, 2]
        - downward_speed_threshold
    ).clamp_min(0.0)
    penalty = (downward_speed.square() * first_contact).sum(dim=1)
    return penalty / env.step_dt if per_event else penalty


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)
