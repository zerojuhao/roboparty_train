"""Reset / randomization events for the UFO task."""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Literal

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op

from ..motion.torch_utils import my_quat_rotate, quat_from_angle_axis, quat_mul
from robolab.utils.tensors import as_torch_tensor, resolve_env_ids
if TYPE_CHECKING:
    from isaaclab.sensors import Camera, RayCasterCamera
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCamera

TERRAIN_RESET_CLEARANCE = 0.05


def resample_motion_for_reset(env: ManagerBasedRLEnv, env_ids, command_name: str):
    env.command_manager.get_term(command_name).prepare_reset(env_ids)

def _view_world_poses(sensor, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
    positions, orientations = sensor._view.get_world_poses(indices)
    return positions.torch.clone(), orientations.torch.clone()

def randomize_joint_calibration(
    env: ManagerBasedRLEnv,
    env_ids,
    offset_range: tuple[float, float] = (-0.01, 0.01),
):
    """Sample one persistent joint zero/calibration offset per episode.

    The offset changes the joint setpoint zero used by the position action. It
    is kept outside the policy observation and is exposed to the optional
    world model as simulator context, matching SuperMimic's hidden
    per-episode calibration parameter.
    """
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    if env_ids.numel() == 0:
        return
    action = env.action_manager.get_term("joint_pos")
    if not hasattr(action, "_offset") or not torch.is_tensor(action._offset):
        raise RuntimeError("UFO joint calibration requires a tensor joint-position action offset")
    nominal = getattr(action, "_ufo_nominal_offset", None)
    if nominal is None:
        nominal = action._offset.detach().clone()
        action._ufo_nominal_offset = nominal
    offsets = torch.empty(
        (env_ids.numel(), nominal.shape[-1]), device=nominal.device, dtype=nominal.dtype
    ).uniform_(*offset_range)
    action._offset[env_ids] = nominal[env_ids] + offsets
    if not hasattr(env, "_ufo_joint_calibration_offset"):
        env._ufo_joint_calibration_offset = torch.zeros_like(nominal)
    env._ufo_joint_calibration_offset[env_ids] = offsets


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
    root_states[:, 2] = env.scene.env_origins[env_ids, 2] + root_pos[:, 2]
    if nonflat:
        root_states[:, :2] = env.scene.env_origins[env_ids, :2]
        root_states[:, 2] = env.scene.env_origins[env_ids, 2] + float(robot.cfg.init_state.pos[2])
    # Motion NPZ and the local IsaacLab 3.0 Newton pose API both use XYZW.
    root_states[:, 3:7] = root_quat
    root_states[:, 7:10] = root_lin_vel
    root_states[:, 10:13] = root_ang_vel
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=root_states[:, 7:13], env_ids=env_ids)
    if nonflat:
        _align_reset_root_to_terrain(env, env_ids, root_states)



def randomize_camera_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the camera offset pose to mimic the sensor installation errors, which lead to imperfect camera calibration.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    # extract the used quantities (to enable type-hinting), as well as all inherited classes
    sensor: Camera | RayCasterCamera | MultiMeshRayCasterCamera = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=sensor._device)

    # get the camera pose
    camera_offset_pos = torch.tensor(list(sensor.cfg.offset.pos), device=sensor._device).repeat(sensor._view.count, 1)
    camera_quat_w = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([sensor.cfg.offset.rot], device=sensor._device),
        origin=sensor.cfg.offset.convention,
        target="world",
    )
    camera_offset_quat = camera_quat_w.repeat(sensor._view.count, 1)
    camera_offset_pos = camera_offset_pos[env_ids]
    camera_offset_quat = camera_offset_quat[env_ids]

    # sample from given range
    camera_offset_pos[..., 0] = _randomize_prop_by_op(
        camera_offset_pos[..., 0].unsqueeze(-1),
        offset_pose_ranges.get("x", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 1] = _randomize_prop_by_op(
        camera_offset_pos[..., 1].unsqueeze(-1),
        offset_pose_ranges.get("y", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 2] = _randomize_prop_by_op(
        camera_offset_pos[..., 2].unsqueeze(-1),
        offset_pose_ranges.get("z", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_w = math_utils.euler_xyz_from_quat(camera_offset_quat)

    camera_euler_roll = _randomize_prop_by_op(
        camera_euler_w[0].unsqueeze(-1),
        offset_pose_ranges.get("roll", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_pitch = _randomize_prop_by_op(
        camera_euler_w[1].unsqueeze(-1),
        offset_pose_ranges.get("pitch", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_yaw = _randomize_prop_by_op(
        camera_euler_w[2].unsqueeze(-1),
        offset_pose_ranges.get("yaw", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_quat = math_utils.quat_from_euler_xyz(
        camera_euler_roll,
        camera_euler_pitch,
        camera_euler_yaw,
    )
    camera_pos_w, camera_quat_w = _view_world_poses(sensor, env_ids)
    camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
        camera_pos_w, camera_quat_w, camera_offset_pos, camera_offset_quat
    )

    # set the new camera pose
    # Note: the offset will be updated automatically,
    # and the attachment relation is kept.
    sensor.set_world_poses(
        camera_pos_w,
        camera_quat_w,
        env_ids=env_ids,
        convention=sensor.cfg.offset.convention,
    )


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
    else:
        from .pufo_reset import correct_terrain_penetration

        # Contacts must be recomputed after both the root and joints match the frame.
        root_states = as_torch_tensor(robot.data.root_link_state_w)[env_ids].clone()
        correct_terrain_penetration(env, env_ids, root_states)


def push_robot_by_setting_base_velocity(env: ManagerBasedRLEnv, env_ids, velocity_range: dict):
    """Apply a sampled base-velocity disturbance for tasks that explicitly use pushes."""
    robot = env.scene["robot"]
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    if env_ids.numel() == 0:
        return
    push_mask = getattr(env, "_ufo_wm_push_mask", None)
    if push_mask is not None:
        from .world_model_state import world_model_state

        before = world_model_state(env)[env_ids].clone()
        first_push = ~push_mask[env_ids]
        env._ufo_wm_pre_push_state[env_ids[first_push]] = before[first_push]
    lin = as_torch_tensor(robot.data.root_lin_vel_w)[env_ids]
    vel_lin = torch.zeros_like(lin)
    vel_ang = torch.zeros_like(lin)
    for key, (lo, hi) in velocity_range.items():
        idx = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}[key]
        target = vel_ang if key in ("roll", "pitch", "yaw") else vel_lin
        target[:, idx] = torch.rand_like(target[:, idx]) * (hi - lo) + lo
    target_velocity = torch.cat((vel_lin, vel_ang), dim=-1)
    robot.write_root_velocity_to_sim_index(root_velocity=target_velocity, env_ids=env_ids)
    if push_mask is not None:
        # A push replaces COM velocity, so the final push in a step takes precedence.
        env._ufo_wm_push_target[env_ids] = target_velocity
        push_mask[env_ids] = True
