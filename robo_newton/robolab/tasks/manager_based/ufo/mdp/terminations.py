"""Termination terms for the UFO task.

The reference run disables all task terminations (time-out only), so
`motion_far` is provided but off by default.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from robolab.utils.tensors import as_torch_tensor


def terrain_grid_boundary_margin(root_xy, patch_size, *, num_rows, num_cols, border_width=0.0):
    """Distance to the outer edge of the connected, origin-centered terrain grid."""
    if border_width < 0 or num_rows < 1 or num_cols < 1:
        raise ValueError("Invalid terrain boundary dimensions")
    half_extent = 0.5 * root_xy.new_tensor(patch_size) * root_xy.new_tensor((num_rows, num_cols))
    return (half_extent + border_width - root_xy.abs()).amin(dim=-1)


def terrain_out_of_bounds(env: ManagerBasedRLEnv, safety_margin: float = 0.5) -> torch.Tensor:
    """Truncate before the root-centered terrain scanner leaves the global map."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if safety_margin < 0:
        raise ValueError("Terrain boundary safety margin must be non-negative")
    from .pufo_observations import (
        TERRAIN_PRIV_X_MIN, TERRAIN_PRIV_X_MAX, TERRAIN_PRIV_Y_MIN, TERRAIN_PRIV_Y_MAX,
    )
    radius = (max(abs(TERRAIN_PRIV_X_MIN), abs(TERRAIN_PRIV_X_MAX)) ** 2
              + max(abs(TERRAIN_PRIV_Y_MIN), abs(TERRAIN_PRIV_Y_MAX)) ** 2) ** 0.5
    cfg = terrain.cfg.terrain_generator
    root_xy = as_torch_tensor(env.scene["robot"].data.root_pos_w)[:, :2]
    margin = terrain_grid_boundary_margin(
        root_xy, cfg.size, num_rows=terrain.terrain_origins.shape[0],
        num_cols=terrain.terrain_origins.shape[1], border_width=cfg.border_width,
    )
    return margin < radius + safety_margin


def motion_far(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    motion_res = command.get_motion_state()
    root_pos_ref = motion_res["rg_pos_t"][:, 0, :]
    root_pos = env.scene["robot"].data.root_pos_w
    dist = torch.norm(root_pos[:, :2] - root_pos_ref[:, :2], dim=-1)
    return dist > threshold

from isaaclab.envs.mdp.terminations import time_out  # noqa: E402,F401
