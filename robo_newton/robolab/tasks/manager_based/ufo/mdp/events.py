"""Reset / randomization events for the UFO task."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..motion.torch_utils import my_quat_rotate, quat_from_angle_axis, quat_mul
from robolab.utils.tensors import as_torch_tensor, resolve_env_ids

TERRAIN_RESET_CLEARANCE = 0.05


def resample_motion_for_reset(env: ManagerBasedRLEnv, env_ids, command_name: str):
    env.command_manager.get_term(command_name).prepare_reset(env_ids)


def _align_reset_root_to_terrain(env: ManagerBasedRLEnv, env_ids, root_states: torch.Tensor) -> None:
    """Place reset roots at the authored height above the local terrain.

    Motion clips contain world-space root heights for flat-ground playback.  They
    must not be used as the reset height on generated terrain: a low frame from a
    fall clip would put the robot inside the terrain.  The authored robot
    ``init_state`` height plus a small clearance is used above the highest
    sampled terrain point in the reset footprint.
    """
    sensor = env.scene.sensors.get("terrain_priv")
    robot = env.scene["robot"]
    init_height = float(robot.cfg.init_state.pos[2])
    terrain_base = env.scene.env_origins[env_ids, 2]
    fallback = terrain_base + init_height + TERRAIN_RESET_CLEARANCE

    if sensor is None:
        root_states[:, 2] = fallback
        robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
        return

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    sensor.update(dt=0.0, force_recompute=True)
    hit_heights = as_torch_tensor(sensor.data.ray_hits_w)[env_ids, ..., 2]
    finite = torch.isfinite(hit_heights)
    pattern = sensor.cfg.pattern_cfg
    ray_starts, _ = pattern.func(pattern, env.device)
    ray_xy = ray_starts[:, :2] + torch.tensor(sensor.cfg.offset.pos[:2], device=env.device)
    footprint = torch.linalg.vector_norm(ray_xy, dim=-1) <= 0.65
    footprint_heights = torch.where(
        finite[:, footprint],
        hit_heights[:, footprint],
        torch.full_like(hit_heights[:, footprint], -torch.inf),
    )
    terrain_height = footprint_heights.amax(dim=-1)
    valid = torch.isfinite(terrain_height)
    terrain_height = torch.where(valid, terrain_height, terrain_base)
    # Never lower a reset below the terrain patch origin when a ray misses or
    # the sampled footprint lies in a depression.
    terrain_height = torch.maximum(terrain_height, terrain_base)
    root_states[:, 2] = terrain_height + init_height + TERRAIN_RESET_CLEARANCE
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)


def reset_root_state_from_motion(env: ManagerBasedRLEnv, env_ids, command_name: str):
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    command = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    motion_res = command.get_motion_state()
    root_pos = motion_res["rg_pos_t"][env_ids, 0, :]
    root_quat = motion_res["rg_rot_t"][env_ids, 0, :]
    root_lin_vel = motion_res["body_vel_t"][env_ids, 0, :]
    root_ang_vel = motion_res["body_ang_vel_t"][env_ids, 0, :]

    terrain = env.scene.terrain
    nonflat = terrain is not None and terrain.cfg.terrain_type == "generator"
    if nonflat:
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(-torch.pi, torch.pi)
        axis = torch.zeros(len(env_ids), 3, device=env.device)
        axis[:, 2] = 1.0
        yaw_quat = quat_from_angle_axis(yaw, axis, w_last=True)
        root_quat = quat_mul(yaw_quat, root_quat, w_last=True)
        root_lin_vel = my_quat_rotate(yaw_quat, root_lin_vel)
        root_ang_vel = my_quat_rotate(yaw_quat, root_ang_vel)

    robot = env.scene["robot"]
    root_states = robot.data.default_root_state[env_ids].clone()
    root_states[:, 0:2] = root_pos[:, :2] + env.scene.env_origins[env_ids, :2]
    root_states[:, 2] = env.scene.env_origins[env_ids, 2] + float(robot.cfg.init_state.pos[2])
    if nonflat:
        root_states[:, :2] = env.scene.env_origins[env_ids, :2]
    # Motion NPZ and the local IsaacLab 3.0 Newton pose API both use XYZW.
    root_states[:, 3:7] = root_quat
    root_states[:, 7:10] = root_lin_vel
    root_states[:, 10:13] = root_ang_vel
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13], env_ids=env_ids)
    if nonflat:
        _align_reset_root_to_terrain(env, env_ids, root_states)


def randomize_camera_installation(
    env: ManagerBasedRLEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    position_error: float,
    angle_error_rpy: tuple[float, float, float],
):
    """Sample one fixed camera installation error per environment."""
    sensor = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    count = env_ids.numel()
    nominal_pos = torch.tensor(sensor.cfg.offset.pos, device=env.device).expand(count, -1)
    nominal_quat = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([sensor.cfg.offset.rot], device=env.device),
        origin=sensor.cfg.offset.convention,
        target="world",
    ).expand(count, -1)
    position = nominal_pos + torch.empty(count, 3, device=env.device).uniform_(
        -position_error, position_error
    )
    angle_error = torch.tensor(angle_error_rpy, device=env.device)
    angles = (2.0 * torch.rand(count, 3, device=env.device) - 1.0) * angle_error
    rotation = math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    orientation = math_utils.quat_mul(nominal_quat, rotation)
    wp.to_torch(sensor._offset_pos_wp)[env_ids] = position
    wp.to_torch(sensor._offset_quat_wp)[env_ids] = orientation


def reset_joint_state_from_motion(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str,
    joint_pos_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset to the motion frame with fresh position noise, keeping the reference zero fixed."""
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    command = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    motion_res = command.get_motion_state()
    dof_pos = motion_res["dof_pos"][env_ids]
    dof_vel = motion_res["dof_vel"][env_ids]
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, :] = dof_pos
    if joint_pos_range != (0.0, 0.0):
        joint_pos += torch.empty_like(joint_pos).uniform_(*joint_pos_range)
    joint_vel[:, :] = dof_vel
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    terrain = env.scene.terrain
    if terrain is not None and terrain.cfg.terrain_type == "generator":
        root_states = as_torch_tensor(robot.data.root_state_w)[env_ids].clone()
        _align_reset_root_to_terrain(env, env_ids, root_states)


def push_robot_by_setting_base_velocity(env: ManagerBasedRLEnv, env_ids, velocity_range: dict):
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    lin = as_torch_tensor(robot.data.root_lin_vel_w)[env_ids]
    vel_lin = torch.zeros_like(lin)
    vel_ang = torch.zeros_like(lin)
    for key, (lo, hi) in velocity_range.items():
        idx = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}[key]
        target = vel_ang if key in ("roll", "pitch", "yaw") else vel_lin
        target[:, idx] = torch.rand_like(target[:, idx]) * (hi - lo) + lo
    root_velocity = torch.cat([vel_lin, vel_ang], dim=-1)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def randomize_default_joint_pos(env: ManagerBasedRLEnv, env_ids, offset_range: tuple[float, float]):
    """Legacy entry point for serialized evaluation configs from already-running jobs.

    New training configs use position noise in ``reset_joint_state_from_motion`` instead.
    """
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    default_pos = as_torch_tensor(robot.data.default_joint_pos)
    offsets = torch.empty_like(default_pos[env_ids]).uniform_(*offset_range)
    default_pos[env_ids] += offsets
