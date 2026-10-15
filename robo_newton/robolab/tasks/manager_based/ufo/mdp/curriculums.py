"""Training-clock curricula for PUFO terrain sampling."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_by_global_step(
    env: ManagerBasedRLEnv,
    env_ids,
    start_step: int,
    interval: int,
    initial_max_level: int,
    level_increment: int,
    easy_fraction: float,
):
    """Resample terrain rows at reset using the H200 global-step schedule."""
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return {"maximum_level": 0}
    if interval <= 0 or initial_max_level < 0 or level_increment <= 0:
        raise ValueError("Invalid terrain curriculum schedule.")
    if not 0.0 <= easy_fraction <= 1.0:
        raise ValueError("easy_fraction must be in [0, 1].")

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    global_step = max(0, int(getattr(env, "_ufo_global_step", 0)) - start_step)
    maximum = min(
        terrain.terrain_origins.shape[0] - 1,
        initial_max_level + level_increment * (global_step // interval),
    )
    levels = torch.randint(maximum + 1, (len(env_ids),), device=env.device)
    easy = torch.rand(len(env_ids), device=env.device) < easy_fraction
    easy_levels = torch.randint(min(2, maximum) + 1, (len(env_ids),), device=env.device)
    levels = torch.where(easy, easy_levels, levels)
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    return {"maximum_level": maximum}
