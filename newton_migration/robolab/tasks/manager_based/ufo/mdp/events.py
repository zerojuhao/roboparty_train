"""Reset / randomization events for the UFO task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def _t(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def reset_root_state_from_motion(env: ManagerBasedRLEnv, env_ids, command_name: str):
    command = env.command_manager.get_term(command_name)
    if len(env_ids) == 0:
        return
    motion_res = command.get_motion_state()
    root_pos = motion_res["rg_pos_t"][env_ids, 0, :]
    root_quat = motion_res["rg_rot_t"][env_ids, 0, :]
    root_lin_vel = motion_res["body_vel_t"][env_ids, 0, :]
    root_ang_vel = motion_res["body_ang_vel_t"][env_ids, 0, :]

    robot = env.scene["robot"]
    root_states = robot.data.default_root_state[env_ids].clone()
    root_states[:, 0:3] = root_pos
    # Motion data and IsaacLab 3.0 sim poses share the XYZW (x, y, z, w)
    # quaternion convention, so the reference orientation is written as-is.
    root_states[:, 3:7] = root_quat
    root_states[:, 7:10] = root_lin_vel
    root_states[:, 10:13] = root_ang_vel
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13], env_ids=env_ids)


def reset_joint_state_from_motion(env: ManagerBasedRLEnv, env_ids, command_name: str):
    command = env.command_manager.get_term(command_name)
    if len(env_ids) == 0:
        return
    motion_res = command.get_motion_state()
    dof_pos = motion_res["dof_pos"][env_ids]
    dof_vel = motion_res["dof_vel"][env_ids]
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, :] = dof_pos
    joint_vel[:, :] = dof_vel
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def push_robot_by_setting_base_velocity(env: ManagerBasedRLEnv, env_ids, velocity_range: dict):
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    lin = _t(robot.data.root_lin_vel_w)[env_ids]
    vel_lin = torch.zeros_like(lin)
    vel_ang = torch.zeros_like(lin)
    for key, (lo, hi) in velocity_range.items():
        idx = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}[key]
        target = vel_ang if key in ("roll", "pitch", "yaw") else vel_lin
        target[:, idx] = torch.rand_like(target[:, idx]) * (hi - lo) + lo
    root_velocity = torch.cat([vel_lin, vel_ang], dim=-1)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)


def randomize_default_joint_pos(env: ManagerBasedRLEnv, env_ids, offset_range: tuple[float, float]):
    """Randomize the per-environment PD/action default joint-position offset."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    default_pos = _t(robot.data.default_joint_pos)
    offsets = torch.empty_like(default_pos[env_ids]).uniform_(*offset_range)
    default_pos[env_ids] += offsets
