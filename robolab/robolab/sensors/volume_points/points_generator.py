from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .points_generator_cfg import Grid3dPointsGeneratorCfg


def grid3d_points_generator(points_generator_cfg: Grid3dPointsGeneratorCfg) -> torch.Tensor:
    """Creates a grid of points in 3D space.

    Args:
        points_generator_cfg (Grid3dPointsGeneratorCfg): Configuration for the points generator.

    Returns:
        torch.Tensor shape (N, 3): A tensor containing the generated points in 3D space.
    """
    x = torch.linspace(
        points_generator_cfg.x_min,
        points_generator_cfg.x_max,
        points_generator_cfg.x_num,
    )
    y = torch.linspace(
        points_generator_cfg.y_min,
        points_generator_cfg.y_max,
        points_generator_cfg.y_num,
    )
    z = torch.linspace(
        points_generator_cfg.z_min,
        points_generator_cfg.z_max,
        points_generator_cfg.z_num,
    )

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
