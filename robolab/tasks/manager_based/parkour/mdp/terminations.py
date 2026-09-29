from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.terminations import bad_orientation as _bad_orientation
from isaaclab.envs.mdp.terminations import illegal_contact as _illegal_contact
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sub_terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the sub terrain.

    If the actor moves too close to the edge of the sub terrain, the termination is activated. The distance
    to the edge of the sub terrain is calculated based on the size of the sub terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 0] - env.scene.terrain.env_origins[:, 0])
            > 0.5 * grid_width - distance_buffer
        )
        y_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1])
            > 0.5 * grid_length - distance_buffer
        )
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def root_height_below_env_origin_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_pos_w[:, 2] - terrain_base_height < minimum_height


def _mask_done_on_terrains(
    env: ManagerBasedRLEnv,
    done: torch.Tensor,
    disabled_terrain_types: tuple[str, ...],
) -> torch.Tensor:
    if not disabled_terrain_types:
        return done
    terrain = env.scene.terrain
    if getattr(terrain.cfg, "terrain_type", None) != "generator":
        return done
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    unknown = set(disabled_terrain_types) - set(names)
    if unknown:
        raise ValueError(
            f"disabled_terrain_types not in generator: {sorted(unknown)}. "
            f"Available types: {names}."
        )
    sub_idx = terrain.terrain_generator.get_subterrain_indices(
        terrain.terrain_levels, terrain.terrain_types, device=env.device
    )
    disabled = torch.zeros_like(done)
    for name in disabled_terrain_types:
        disabled |= sub_idx == names.index(name)
    return done & ~disabled


def illegal_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    disabled_terrain_types: tuple[str, ...] = (),
) -> torch.Tensor:
    """Terminate on illegal contact, optionally skipping named generated terrains."""
    done = _illegal_contact(env, threshold=threshold, sensor_cfg=sensor_cfg)
    return _mask_done_on_terrains(env, done, disabled_terrain_types)


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    disabled_terrain_types: tuple[str, ...] = (),
) -> torch.Tensor:
    """Terminate on excessive tilt, optionally skipping named generated terrains."""
    done = _bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)
    return _mask_done_on_terrains(env, done, disabled_terrain_types)
