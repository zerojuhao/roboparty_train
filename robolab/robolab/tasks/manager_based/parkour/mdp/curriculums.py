from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Convert manager-provided environment indices to a device-local tensor."""
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.tensor(list(env_ids), device=env.device, dtype=torch.long)


def _terrain_type_mask(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    terrain_types: Sequence[str],
) -> torch.Tensor:
    """Boolean mask over ``env_ids`` for named generated sub-terrains."""
    terrain: TerrainImporter = env.scene.terrain
    if getattr(terrain.cfg, "terrain_type", None) != "generator":
        return torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    indices = terrain.terrain_generator.get_subterrain_indices(
        terrain.terrain_levels, terrain.terrain_types, device=env.device
    )
    mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=env.device)
    for terrain_type in terrain_types:
        if terrain_type not in names:
            raise ValueError(
                f"Excluded terrain {terrain_type!r} is not configured. "
                f"Available types: {names}."
            )
        mask |= indices[env_ids] == names.index(terrain_type)
    return mask


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
    terrain_tracking_overrides: dict[str, dict] | None = None,
    exclude_terrain_types: Sequence[str] | None = None,
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
        exclude_terrain_types: Sub-terrains excluded from tracking-based level
            updates and from the logged mean terrain level.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    env_idx = _resolve_env_ids(env, env_ids)
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_idx].clone()
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_idx]
    eligible = torch.ones(len(env_idx), dtype=torch.bool, device=env.device)

    if terrain_tracking_overrides:
        sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        sub_terrain_indices = terrain.terrain_generator.get_subterrain_indices(
            terrain.terrain_levels, terrain.terrain_types, device=env.device
        )
        for terrain_type, override in terrain_tracking_overrides.items():
            if terrain_type not in sub_terrain_names:
                raise ValueError(f"Terrain tracking override references unknown terrain {terrain_type!r}.")
            metric_name = override["metric_name"]
            count_metric_name = override["count_metric_name"]
            min_valid_steps = int(override["min_valid_steps"])
            if metric_name not in command.metrics or count_metric_name not in command.metrics:
                raise ValueError(
                    f"Tracking override for {terrain_type!r} references unavailable command metrics."
                )
            target_index = sub_terrain_names.index(terrain_type)
            target_mask = sub_terrain_indices[env_idx] == target_index
            target_env_ids = env_idx[target_mask]
            tracking_exp_vel_xy[target_mask] = command.metrics[metric_name][target_env_ids]
            eligible[target_mask] = (
                command.metrics[count_metric_name][target_env_ids] >= min_valid_steps
            )

    if exclude_terrain_types:
        eligible &= ~_terrain_type_mask(env, env_idx, exclude_terrain_types)

    move_up = (
        eligible
        & (tracking_exp_vel_xy > lin_vel_threshold[1])
        & (tracking_exp_vel_yaw > ang_vel_threshold[1])
    )
    move_down = eligible & (tracking_exp_vel_xy < lin_vel_threshold[0]) & ~move_up
    # update terrain levels
    terrain.update_env_origins(env_idx, move_up, move_down)
    levels = terrain.terrain_levels.float()
    if exclude_terrain_types:
        keep = ~_terrain_type_mask(
            env,
            torch.arange(env.num_envs, device=env.device),
            exclude_terrain_types,
        )
        if keep.any():
            return levels[keep].mean()
    return levels.mean()


def modify_rewards_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str | Sequence[str],
    init_weight: float,
    final_weight: float,
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
    step_size: float = 0.02,
    exclude_terrain_types: Sequence[str] | None = None,
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to gradually adjust a reward term's weight from ``init_weight`` toward
    ``final_weight`` when the robot tracks its commanded velocity well (high score), and step
    it back toward ``init_weight`` when the robot tracks poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        term_name: Name of the reward term (or list of names updated together) whose weight
            will be modified.
        init_weight: Initial (easy) weight of the reward term.
        final_weight: Final (strict) weight to ramp toward when tracking is good.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear
            velocity tracking score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the weight is moved
            back toward ``init_weight``.
            If the score is above the upper threshold (good tracking), the weight is moved
            toward ``final_weight``.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular
            velocity tracking score (exponential kernel). Similar logic as ``lin_vel_threshold``.
        step_size: Fraction of the total weight range ``(final_weight - init_weight)`` applied
            as a fixed absolute step each time this curriculum is triggered
            (0.05 = move 5% of the full range per update).
        exclude_terrain_types: Sub-terrains whose tracking scores are ignored. Their
            per-env reward weights are left unchanged.

    Returns:
        The mean reward term weight as a tensor scalar, for logging. Excluded
        terrains are omitted from the mean when ``exclude_terrain_types`` is set.
    """
    command = env.command_manager.get_term("base_velocity")
    env_idx = _resolve_env_ids(env, env_ids)
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_idx]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_idx]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) & (
        tracking_exp_vel_yaw > ang_vel_threshold[1]
    )
    move_down = (tracking_exp_vel_xy < lin_vel_threshold[0]) & ~move_up
    if exclude_terrain_types:
        skip = _terrain_type_mask(env, env_idx, exclude_terrain_types)
        move_up = move_up & ~skip
        move_down = move_down & ~skip

    term_names = [term_name] if isinstance(term_name, str) else list(term_name)
    abs_step = (final_weight - init_weight) * step_size
    weight_min = min(init_weight, final_weight)
    weight_max = max(init_weight, final_weight)
    logged_mean = None
    keep_for_log = None
    if exclude_terrain_types:
        keep_for_log = ~_terrain_type_mask(
            env,
            torch.arange(env.num_envs, device=env.device),
            exclude_terrain_types,
        )
    for name in term_names:
        per_env_weights = env.reward_manager.get_per_env_term_weights(name)
        current = per_env_weights[env_idx]
        move_up_f = move_up.to(dtype=current.dtype, device=current.device)
        move_down_f = move_down.to(dtype=current.dtype, device=current.device)
        current = current + abs_step * move_up_f
        current = current - abs_step * move_down_f
        current = torch.clamp(current, weight_min, weight_max)
        env.reward_manager.set_term_weight_for_envs(name, env_idx, current)

        global_weights = env.reward_manager.get_per_env_term_weights(name)
        if keep_for_log is not None and keep_for_log.any():
            logged_mean = global_weights[keep_for_log].mean()
        else:
            logged_mean = global_weights.mean()
    return logged_mean
