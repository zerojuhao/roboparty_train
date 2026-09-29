from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names
import isaaclab.utils.math as math_utils
from robolab.sensors.volume_points import VolumePoints
from robolab.sensors.volume_points.points_generator import grid3d_points_generator
from robolab.sensors.volume_points.points_generator_cfg import Grid3dPointsGeneratorCfg

from .contact_metrics import (
    bounded_flight_phase_reward,
    capped_swing_clearance_reward,
    directional_joint_lift_reward,
    horizontal_force_excess,
)
from .terrain_family import (
    STAIRS_DOWN_FAMILY_ID,
    STAIRS_UP_FAMILY_ID,
    asymmetric_pitch_roll_penalty,
    conservative_support_deficiency,
    get_terrain_family_ids,
    highest_plane_support_deficiency,
    terrain_foot_point_weights,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize action changes."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize second-order action changes."""
    action = env.action_buffer.buffer
    penalty = torch.sum(
        torch.square(action[:, -3, :] - 2 * action[:, -2, :] + action[:, -1, :]),
        dim=1,
    )
    return penalty

def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

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
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


# def stand_still(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     threshold: float = 0.15,
#     offset: float = 1.0,
# ) -> torch.Tensor:
#     """Penalize moving when there is no velocity command."""
#     asset = env.scene[asset_cfg.name]
#     dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
#     return (
#         (dof_error - offset)
#         * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
#         * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
#     )

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

def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviations from the default pose."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize squared joint deviation from default, optionally scaling the offset first."""
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(scale * angle), dim=1)



def body_distance_y(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def link_close_y_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when two links are too close together in the heading-frame y distance."""
    asset = env.scene[asset_cfg.name]
    left_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)
    left_y = -sin_heading * left_xy[:, 0] + cos_heading * left_xy[:, 1]
    right_y = -sin_heading * right_xy[:, 0] + cos_heading * right_xy[:, 1]
    distance_y = torch.abs(left_y - right_y)
    return 1 - torch.exp(-torch.clamp(threshold - distance_y, min=0.0) / std**2)


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize vertical foot velocity and acceleration during ground contact."""
    asset = env.scene["robot"]
    body_ids = sensor_cfg.body_ids
    foot_vel_z = asset.data.body_vel_w[:, body_ids, 2]
    foot_acc_z = asset.data.body_acc_w[:, body_ids, 2]
    contact_force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, body_ids, 2]
    in_contact = torch.abs(contact_force_z) > 0.0
    penalty = ((foot_vel_z.square() + foot_acc_z.square()) * in_contact.float()).sum(dim=1)
    return penalty

def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.05,
    command_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize moving too slowly when there is a velocity command in any direction."""
    asset: RigidObject = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    lin_vel_cmd = cmd[:, :2]
    ang_vel_cmd_z = cmd[:, 2]

    lin_vel = asset.data.root_lin_vel_b[:, :2]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]

    lin_vel_along_cmd = lin_vel * torch.sign(lin_vel_cmd)
    ang_vel_along_cmd = ang_vel_z * torch.sign(ang_vel_cmd_z)

    lin_penalty = (
        (lin_vel_along_cmd < speed_threshold).float()
        + (lin_vel_along_cmd < 0.0).float()
        + (lin_vel_along_cmd < -speed_threshold).float()
    ) * (torch.abs(lin_vel_cmd) > command_threshold).float()

    ang_penalty = (
        (ang_vel_along_cmd < speed_threshold).float()
        + (ang_vel_along_cmd < 0.0).float()
        + (ang_vel_along_cmd < -speed_threshold).float()
    ) * (torch.abs(ang_vel_cmd_z) > command_threshold).float()

    return lin_penalty.sum(dim=1) + ang_penalty


def feet_orientation_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    enabled_terrain_family_ids: tuple[int, ...] | None = None,
    deadzone: float = 0.0,
    quadratic_scale: float | None = None,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """Penalize contacting-foot tilt beyond a deadzone on selected terrains.

    With ``quadratic_scale``, the excess tilt is normalized and squared so rare,
    severe edge contacts receive substantially more weight than small tilts.
    """
    if quadratic_scale is not None and quadratic_scale <= 0.0:
        raise ValueError(f"quadratic_scale must be positive, got {quadratic_scale}.")
    if max_penalty <= 0.0:
        raise ValueError(f"max_penalty must be positive, got {max_penalty}.")
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W.torch)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W.torch)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    left_tilt = torch.atan2(
        torch.linalg.vector_norm(left_projected_gravity[:, :2], dim=-1),
        -left_projected_gravity[:, 2],
    )
    right_tilt = torch.atan2(
        torch.linalg.vector_norm(right_projected_gravity[:, :2], dim=-1),
        -right_projected_gravity[:, 2],
    )
    left_excess = torch.relu(left_tilt - float(deadzone))
    right_excess = torch.relu(right_tilt - float(deadzone))
    if quadratic_scale is not None:
        scale = float(quadratic_scale)
        left_excess = torch.square(left_excess / scale).clamp_max(max_penalty)
        right_excess = torch.square(right_excess / scale).clamp_max(max_penalty)
    penalty = left_excess * is_contact[:, 0] + right_excess * is_contact[:, 1]
    if enabled_terrain_family_ids is None:
        return penalty

    family_ids = get_terrain_family_ids(env)
    enabled = torch.zeros_like(family_ids, dtype=torch.bool)
    for family_id in enabled_terrain_family_ids:
        enabled |= family_ids == family_id
    return penalty * enabled


def _height_scanner_local_x(scanner, device: torch.device | str) -> torch.Tensor:
    """Foot-local x of each height-scanner ray (matches ``ray_hits_w`` order)."""
    pattern_cfg = scanner.cfg.pattern_cfg
    ray_starts, _ = pattern_cfg.func(pattern_cfg, device)
    return ray_starts[:, 0]


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.035,
    height_tolerance: float = 0.03,
    support_transition_width: float = 0.005,
    enable_terrain_foot_weights: bool = False,
    max_plane_terrain_family_ids: tuple[int, ...] | None = None,
    disabled_terrain_family_ids: tuple[int, ...] | None = None,
    stairs_weight_min: float = 0.0,
    stairs_weight_max: float = 1.0,
) -> torch.Tensor:
    """Penalize hanging sole mass while feet are in contact.

    Selected terrain families use the imagined-guidance highest-plane kernel,
    which prevents foot pitch from hiding a tread edge. Other families retain
    ankle-relative clearance unless explicitly disabled. Returns the sum of
    per-foot values in ``[0, 2]``; pair with a negative reward weight.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1
    )
    left_sensor = env.scene[left_height_scanner_cfg.name]
    right_sensor = env.scene[right_height_scanner_cfg.name]
    left_heights = left_sensor.data.ray_hits_w[..., 2]
    right_heights = right_sensor.data.ray_hits_w[..., 2]
    left_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    family_ids = None
    if (
        enable_terrain_foot_weights
        or max_plane_terrain_family_ids is not None
        or disabled_terrain_family_ids is not None
    ):
        family_ids = get_terrain_family_ids(env)

    left_weights = None
    right_weights = None
    if enable_terrain_foot_weights:
        assert family_ids is not None
        left_weights = terrain_foot_point_weights(
            _height_scanner_local_x(left_sensor, env.device),
            family_ids,
            stairs_weight_min,
            stairs_weight_max,
        )
        right_weights = terrain_foot_point_weights(
            _height_scanner_local_x(right_sensor, env.device),
            family_ids,
            stairs_weight_min,
            stairs_weight_max,
        )

    left_unsupported, _, _ = conservative_support_deficiency(
        left_foot_z.unsqueeze(-1),
        left_heights,
        height_offset=height_offset,
        height_tolerance=height_tolerance,
        transition_width=support_transition_width,
        point_weights=left_weights,
        unsupported_only=True,
    )
    right_unsupported, _, _ = conservative_support_deficiency(
        right_foot_z.unsqueeze(-1),
        right_heights,
        height_offset=height_offset,
        height_tolerance=height_tolerance,
        transition_width=support_transition_width,
        point_weights=right_weights,
        unsupported_only=True,
    )
    if max_plane_terrain_family_ids is not None:
        assert family_ids is not None
        use_max_plane = torch.zeros_like(family_ids, dtype=torch.bool)
        for family_id in max_plane_terrain_family_ids:
            use_max_plane |= family_ids == family_id
        left_max_plane, _, _ = highest_plane_support_deficiency(
            left_heights,
            height_tolerance=height_tolerance,
            transition_width=support_transition_width,
            point_weights=left_weights,
            unsupported_only=True,
        )
        right_max_plane, _, _ = highest_plane_support_deficiency(
            right_heights,
            height_tolerance=height_tolerance,
            transition_width=support_transition_width,
            point_weights=right_weights,
            unsupported_only=True,
        )
        left_unsupported = torch.where(use_max_plane, left_max_plane, left_unsupported)
        right_unsupported = torch.where(use_max_plane, right_max_plane, right_unsupported)
    penalty = left_unsupported * is_contact[:, 0].float() + right_unsupported * is_contact[
        :, 1
    ].float()
    if disabled_terrain_family_ids is None:
        return penalty

    assert family_ids is not None
    enabled = torch.ones_like(family_ids, dtype=torch.bool)
    for family_id in disabled_terrain_family_ids:
        enabled &= family_ids != family_id
    return penalty * enabled


def link_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_tolerance: float = 0.0,
    terrain_forward_tolerances: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize link tilt with optional terrain-specific forward deadzones.

    For the standard ``x``-forward, ``z``-up body frame, negative projected
    gravity on the x-axis corresponds to backward pitch.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W.torch)

    tolerance = torch.full(
        (env.num_envs,),
        float(forward_tolerance),
        dtype=link_projected_gravity.dtype,
        device=env.device,
    )
    if terrain_forward_tolerances:
        terrain = env.scene.terrain
        if terrain.cfg.terrain_type == "generator":
            sub_terrain_names = list(
                terrain.cfg.terrain_generator.sub_terrains.keys()
            )
            sub_terrain_indices = terrain.terrain_generator.get_subterrain_indices(
                terrain.terrain_levels,
                terrain.terrain_types,
                device=env.device,
            )
            for terrain_type, terrain_tolerance in terrain_forward_tolerances.items():
                if terrain_type not in sub_terrain_names:
                    raise ValueError(
                        f"Forward-tolerance terrain {terrain_type!r} is not configured. "
                        f"Available types: {sub_terrain_names}."
                    )
                tolerance[sub_terrain_indices == sub_terrain_names.index(terrain_type)] = float(
                    terrain_tolerance
                )
    return asymmetric_pitch_roll_penalty(link_projected_gravity, tolerance)


def link_pitch_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pitch_tolerance: float = 0.0,
    enabled_terrain_family_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Penalize link pitch beyond a symmetric deadzone; roll is ignored.

    Same projected-gravity-x convention as ``link_orientation``. Excess
    ``|g_x|`` above ``sin(pitch_tolerance)`` is squared and summed over links.
    ``enabled_terrain_family_ids`` zeros the penalty on other terrains.
    """
    if pitch_tolerance < 0.0 or pitch_tolerance >= math.pi / 2:
        raise ValueError("pitch_tolerance must be in [0, pi/2).")
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    gravity = asset.data.GRAVITY_VEC_W.torch.unsqueeze(1).expand(-1, link_quat.shape[1], -1)
    projected_gravity = quat_apply_inverse(link_quat, gravity.contiguous())
    excess = torch.clamp(
        torch.abs(projected_gravity[..., 0]) - math.sin(pitch_tolerance), min=0.0
    )
    penalty = torch.sum(torch.square(excess), dim=1)
    if enabled_terrain_family_ids is None:
        return penalty
    family_ids = get_terrain_family_ids(env)
    enabled = torch.zeros_like(family_ids, dtype=torch.bool)
    for family_id in enabled_terrain_family_ids:
        enabled |= family_ids == family_id
    return penalty * enabled


def link_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy angular velocity of specified link(s) in world frame using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    link_ang_vel_xy = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(link_ang_vel_xy), dim=(1, 2))

def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_ratio: float = 1.0,
    force_threshold: float = 20.0,
    force_scale: float = 100.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Continuously penalize horizontal-dominant foot impacts.

    The history maximum preserves short impacts between policy steps. The
    bounded output keeps the existing stumble reward curriculum meaningful.
    """
    if force_scale <= 0.0:
        raise ValueError(f"force_scale must be positive, got {force_scale}.")
    if max_penalty <= 0.0:
        raise ValueError(f"max_penalty must be positive, got {max_penalty}.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    excess = horizontal_force_excess(forces, horizontal_ratio, force_threshold)
    penalty = torch.clamp(excess / force_scale, max=max_penalty)
    return penalty.amax(dim=(1, 2))

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    return reward


def volume_points_penetration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)

    return torch.sum(velocity_times_penetration, dim=-1)


def volume_points_penetration_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    enable_terrain_foot_weights: bool = False,
    stairs_weight_min: float = 0.2,
    stairs_weight_max: float = 1.0,
    debug_print_terrain: bool = False,
) -> torch.Tensor:
    """Penalize volume-point penetration into virtual edge obstacles.

    With ``enable_terrain_foot_weights``, foot-local x maps heel (x_min) to toe (x_max).
    Up-stairs terrains (``pyramid_stairs_inv``) use toe-heavy weights; down-stairs
    (``pyramid_stairs``) use heel-heavy weights. Other terrains use mid-foot-heavy
    weights with heel and toe at ``stairs_weight_min`` and mid-foot at ``stairs_weight_max``.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    num_envs, num_bodies, num_points, _ = penetration.shape
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)

    if enable_terrain_foot_weights:
        terrain = env.scene.terrain
        if terrain.cfg.terrain_type == "generator":
            family_ids = get_terrain_family_ids(env)

            if debug_print_terrain:
                for env_id, family_id in enumerate(family_ids.tolist()):
                    if family_id == STAIRS_UP_FAMILY_ID:
                        label = "up"
                    elif family_id == STAIRS_DOWN_FAMILY_ID:
                        label = "down"
                    else:
                        label = "other"
                    print(f"env {env_id}: {label}, family_id={family_id}")

            points_cfg = volume_sensor.cfg.points_generator
            local_points = grid3d_points_generator(points_cfg).to(env.device)
            env_w = terrain_foot_point_weights(
                local_points[:, 0],
                family_ids,
                stairs_weight_min,
                stairs_weight_max,
            )

            velocity_times_penetration = velocity_times_penetration.view(
                num_envs, num_bodies, num_points
            )
            velocity_times_penetration = (
                velocity_times_penetration * env_w.unsqueeze(1)
            )
            velocity_times_penetration = velocity_times_penetration.flatten(1, 2)

    return torch.sum(velocity_times_penetration, dim=-1)


def step_safety(
    env: ManagerBasedRLEnv,
    volume_points_cfg: SceneEntityCfg,
    contact_forces_cfg: SceneEntityCfg,
    epsilon: float = 1e-5,
    once: bool = False,
) -> torch.Tensor:
    """A log based reward to encourage the robot to make contacts with no penetration to the virtual obstacles.
    Inspired by Deep Tracking Control and Robot Parkour Learning and Humanoid Parkour Learning.
    NOTE: make sure the contact forces sensor is selected for that the volume points sensors are interested in.
    aka. The number of selected bodies in the contact forces sensor should be the same as the number of selected bodies
    in all volume points sensors.
    NOTE: Be aware of the body order.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[volume_points_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_forces_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset  # (N, B_, P_, 3) where B_ and P_ varies in sensors
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_, P_)
    penetration_depth_max = torch.max(penetration_depth, dim=-1)[0]  # (N, B_)
    if once:
        contacts = contact_sensor.compute_first_contact(env.step_dt)[:, contact_forces_cfg.body_ids]  # (N, B_)
    else:
        contact_forces = contact_sensor.data.net_forces_w_history[:, :, contact_forces_cfg.body_ids, :]  # (N, T, B_, 3)
        contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0  # (N, B_)

    rewards = -torch.log(penetration_depth_max + epsilon) * contacts
    return torch.sum(rewards, dim=-1)

def contact_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_penalty: bool = True,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize body sliding.

    This function penalizes the agent for sliding its body on the ground. The reward is computed as the
    norm of the linear velocity of the body multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the body are in contact with the ground.
    """
    # Penalize body sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    if ang_vel_penalty:
        reward = reward + torch.sum(body_ang_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def motors_power_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_stiffness: bool = True,
    normalize_by_num_joints: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    power_j = asset.data.applied_torque * asset.data.joint_vel  # (batch_size, num_joints)
    if normalize_by_stiffness:
        stiffness = asset.data.joint_stiffness
        stiffness = stiffness.torch.clone() if hasattr(stiffness, "torch") else stiffness.clone()
        for actuator in asset.actuators.values():
            if hasattr(actuator, "joint_indices") and hasattr(actuator, "stiffness"):
                stiffness[:, actuator.joint_indices] = actuator.stiffness
        if torch.any(stiffness <= 0):
            for actuator_cfg in asset.cfg.actuators.values():
                joint_ids, joint_names = resolve_matching_names(actuator_cfg.joint_names_expr, asset.joint_names)
                configured = resolve_joint_parameter(
                    actuator_cfg.stiffness, None, joint_names, env.num_envs, env.device
                )
                stiffness[:, joint_ids] = torch.where(
                    stiffness[:, joint_ids] > 0, stiffness[:, joint_ids], configured
                )
        if torch.any(stiffness <= 0):
            raise ValueError("Motor power reward requires positive stiffness for every joint.")
        power_j /= stiffness
    power_j = power_j[:, asset_cfg.joint_ids]  # (batch_size, num_selected_joints)
    power = torch.sum(torch.square(power_j), dim=-1)  # (batch_size,)
    if normalize_by_num_joints:
        power /= power_j.shape[-1]
    return power
    
def applied_torque_limits_by_ratio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_ratio: float = 0.8,
):
    """Penalize when the applied torque excceed certain ratio of the joint torque limit."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_effort_limits = asset.data.joint_effort_limits  # (num_envs, num_joints)
    joint_effort_limits = joint_effort_limits[:, asset_cfg.joint_ids]

    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    applied_torque = torch.abs(applied_torque)

    out_of_limits = (applied_torque - joint_effort_limits * limit_ratio).clip(min=0)
    out_of_limits_err = torch.sum(torch.square(out_of_limits), dim=-1)  # (num_envs,)

    return out_of_limits_err

def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_pitch_asset_cfg: SceneEntityCfg | None = None,
    knee_asset_cfg: SceneEntityCfg | None = None,
    hip_pitch_flexion_signs: tuple[float, ...] = (-1.0, 1.0),
    knee_flexion_signs: tuple[float, ...] = (-1.0, 1.0),
    hip_pitch_target: float = math.radians(15.0),
    knee_flexion_target: float = math.radians(30.0),
    joint_lift_scale: float = 0.2,
    joint_lift_terrain_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.

    When the optional joint configurations are provided, a bounded flexion
    bonus is added for the swing leg. This is intended for in-place turning:
    it only applies during single support and can be restricted to selected
    terrain families.
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
    command_active = torch.norm(
        env.command_manager.get_command(command_name)[:, :3], dim=1
    ) > 0.1
    active = single_stance & command_active
    reward *= command_active

    if hip_pitch_asset_cfg is not None or knee_asset_cfg is not None:
        if hip_pitch_asset_cfg is None or knee_asset_cfg is None:
            raise ValueError(
                "hip_pitch_asset_cfg and knee_asset_cfg must be provided together."
            )
        if joint_lift_scale < 0.0:
            raise ValueError("joint_lift_scale must be non-negative.")
        hip_asset = env.scene[hip_pitch_asset_cfg.name]
        knee_asset = env.scene[knee_asset_cfg.name]
        swing_foot = ~in_contact
        hip_score = directional_joint_lift_reward(
            hip_asset.data.joint_pos[:, hip_pitch_asset_cfg.joint_ids],
            hip_asset.data.default_joint_pos[:, hip_pitch_asset_cfg.joint_ids],
            swing_foot,
            hip_pitch_flexion_signs,
            hip_pitch_target,
        )
        knee_score = directional_joint_lift_reward(
            knee_asset.data.joint_pos[:, knee_asset_cfg.joint_ids],
            knee_asset.data.default_joint_pos[:, knee_asset_cfg.joint_ids],
            swing_foot,
            knee_flexion_signs,
            knee_flexion_target,
        )
        joint_lift = 0.5 * (hip_score + knee_score)
        if joint_lift_terrain_names is not None:
            terrain = env.scene.terrain
            if terrain.cfg.terrain_type != "generator":
                active &= False
            else:
                sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains)
                sub_terrain_indices = terrain.terrain_generator.get_subterrain_indices(
                    terrain.terrain_levels,
                    terrain.terrain_types,
                    device=active.device,
                )
                terrain_active = torch.zeros_like(active)
                for terrain_name in joint_lift_terrain_names:
                    if terrain_name not in sub_terrain_names:
                        raise ValueError(
                            f"Unknown joint-lift terrain {terrain_name!r}; "
                            f"available terrains: {sub_terrain_names}."
                        )
                    terrain_active |= sub_terrain_indices == sub_terrain_names.index(
                        terrain_name
                    )
                active &= terrain_active
        reward += joint_lift_scale * joint_lift * active.float()
    return reward
