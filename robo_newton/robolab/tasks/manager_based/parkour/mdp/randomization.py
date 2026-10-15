from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors.ray_caster import MultiMeshRayCaster

if TYPE_CHECKING:
    from isaaclab.sensors import Camera, RayCasterCamera
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCamera


def _view_world_poses(sensor, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
    positions, orientations = sensor._view.get_world_poses(indices)
    return positions.torch.clone(), orientations.torch.clone()


def randomize_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if offset_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, offset_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]  # type: ignore
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_ray_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the ray_starts and ray_directions of the sensor to mimic the sensor installation errors.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    num_env_ids = env.scene.num_envs if env_ids is None else len(env_ids)
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster | MultiMeshRayCaster = env.scene[asset_cfg.name]
    ray_starts = sensor.ray_starts[env_ids]  # (num_envs, num_rays, 3)
    ray_directions = sensor.ray_directions[env_ids]  # (num_envs, num_rays, 3)
    # sample from given range
    range_list = [offset_pose_ranges.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=ray_starts.device)  # (6, 2)
    rand_samples = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (num_env_ids, 6),
            device=ray_starts.device,
        )[..., None, :]
        .repeat(1, sensor.num_rays, 1)
        .flatten(0, 1)
    )
    position_samples = rand_samples[..., :3]  # (num_envs * num_rays, 3)
    rotation_samples = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3],
        rand_samples[..., 4],
        rand_samples[..., 5],
    )  # (num_envs * num_rays, 4) (w, x, y, z)
    # apply the randomization
    ray_starts += position_samples.reshape(ray_starts.shape)
    ray_directions = math_utils.quat_apply(rotation_samples.reshape(*ray_directions.shape[:-1], 4), ray_directions)

    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions


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


def push_by_setting_velocity_per_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    terrain_velocity_ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push robot with per-sub-terrain velocity sampling ranges (world frame)."""
    if len(env_ids) == 0:
        return

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device
    vel_axes = ("x", "y", "z", "roll", "pitch", "yaw")

    # --- build / reuse lookup table: (num_sub_terrains, 6, 2) ---
    cache_key = (id(velocity_range), id(terrain_velocity_ranges))
    if getattr(env, "_push_vel_range_cache_key", None) != cache_key:
        terrain = env.scene.terrain
        sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        num_sub = len(sub_names)
        table = torch.zeros(num_sub, 6, 2, device=device, dtype=torch.float32)

        for s, name in enumerate(sub_names):
            override = None
            if terrain_velocity_ranges:
                if name in terrain_velocity_ranges:
                    override = terrain_velocity_ranges[name]
                else:
                    best_key = None
                    for key in terrain_velocity_ranges:
                        if key in name and (best_key is None or len(key) > len(best_key)):
                            best_key = key
                    if best_key is not None:
                        override = terrain_velocity_ranges[best_key]
            for j, axis in enumerate(vel_axes):
                lo, hi = (override or {}).get(axis, velocity_range.get(axis, (0.0, 0.0)))
                table[s, j, 0] = lo
                table[s, j, 1] = hi

        env._push_vel_range_table = table
        env._push_vel_range_cache_key = cache_key

    table = env._push_vel_range_table  # (S, 6, 2)

    # --- vectorized per-env lookup ---
    terrain = env.scene.terrain
    terrain_gen = terrain.terrain_generator
    sub_idx = terrain_gen.get_subterrain_indices(
        terrain.terrain_levels[env_ids],
        terrain.terrain_types[env_ids],
        device=device,
    )
    ranges = table[sub_idx]          # (N, 6, 2)
    mins, maxs = ranges[..., 0], ranges[..., 1]

    vel_w = asset.data.root_vel_w[env_ids]
    add_vel = math_utils.sample_uniform(mins, maxs, vel_w.shape, device=device)
    asset.write_root_velocity_to_sim(vel_w + add_vel, env_ids=env_ids)
