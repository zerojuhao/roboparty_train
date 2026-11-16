from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from functools import wraps
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg, GuardedTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """Terrain generator that records per-cell sub-terrain configs and indices."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):
        # Per-cell sub-terrain index; filled during terrain generation (row, col) indexing.
        self.subterrain_index_grid: np.ndarray | None = None
        # Flat list: self._subterrain_specific_cfgs[row * num_cols + col]
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        self.collision_components = []
        if cfg.native_collision and cfg.use_cache:
            raise ValueError("Native collision requires use_cache=False; legacy OBJ caches lose components.")
        # One column per sub-terrain: num_cols follows the number of types.
        if cfg.one_col_per_subterrain:
            cfg.num_cols = len(cfg.sub_terrains)
        super().__init__(cfg, device)
        if cfg.native_collision:
            offset = (-cfg.size[0] * cfg.num_rows / 2, -cfg.size[1] * cfg.num_cols / 2, 0)
            for mesh, _ in self.collision_components:
                mesh.apply_translation(offset)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        """Record the specific config for each sub-terrain mesh generation."""
        original = cfg.function
        if self.cfg.native_collision:
            from robolab.terrains.native_collision import HEIGHT_DATA, flat_heightfield
            from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
            generated = []

            @wraps(original)
            def capture(difficulty, source):
                source._native_collision = True
                meshes, origin = original(difficulty, source)
                for index, part in enumerate(meshes):
                    flat = flat_heightfield(part)
                    part = flat if flat is not None else part
                    generated.append((part.copy(), HEIGHT_DATA in part.metadata or
                                      (isinstance(source, HfTerrainBaseCfg) and index == 0 and not part.is_watertight)))
                return [part for part, _ in generated], origin

            cfg = cfg.copy()
            cfg.function = capture
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        if self.cfg.native_collision:
            self._pending_components = generated
            cfg.function = original
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        self._subterrain_specific_cfgs.append(cfg)
        return mesh, origin

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if self.cfg.native_collision:
            for part, is_hf in self._pending_components:
                part.apply_translation((row * self.cfg.size[0], col * self.cfg.size[1], 0))
                self.collision_components.append((part, is_hf))
            self._pending_components = []

    def _add_terrain_border(self):
        previous = len(self.terrain_meshes)
        super()._add_terrain_border()
        if self.cfg.native_collision:
            for mesh in self.terrain_meshes[previous:]:
                self.collision_components.extend((m, False) for m in mesh.split(only_watertight=False, repair=False))

    def _generate_random_terrains(self):
        """Add terrains with random sub-terrain type per grid cell."""
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = int(self.np_rng.choice(len(proportions), p=proportions))
            self.subterrain_index_grid[sub_row, sub_col] = sub_index
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains with sub-terrain type fixed per column and difficulty along rows."""
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # One column per type, or map columns by cumulative proportion (Isaac Lab default).
        if self.cfg.one_col_per_subterrain:
            sub_indices = np.arange(self.cfg.num_cols, dtype=np.int32)
        else:
            sub_indices = []
            for index in range(self.cfg.num_cols):
                sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
                sub_indices.append(sub_index)
            sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        self.subterrain_index_grid = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                sub_index = int(sub_indices[sub_col])
                self.subterrain_index_grid[sub_row, sub_col] = sub_index
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def get_subterrain_indices(
        self, row_ids: torch.Tensor | int, col_ids: torch.Tensor | int, device: str | torch.device | None = None
    ) -> torch.Tensor:
        """Return sub-terrain dict indices for grid cells (row, col), aligned with terrain_levels/types."""
        if self.subterrain_index_grid is None:
            raise RuntimeError("subterrain_index_grid is not initialized. Terrain generation may have failed.")
        if device is None:
            if isinstance(row_ids, torch.Tensor):
                device = row_ids.device
            elif isinstance(col_ids, torch.Tensor):
                device = col_ids.device
        grid = torch.as_tensor(self.subterrain_index_grid, device=device, dtype=torch.long)
        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.to(grid.device)
        if isinstance(col_ids, torch.Tensor):
            col_ids = col_ids.to(grid.device)
        return grid[row_ids, col_ids]

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all sub-terrains."""
        return self._subterrain_specific_cfgs.copy()

    def get_subterrain_cfg(
        self, row_ids: int | torch.Tensor, col_ids: int | torch.Tensor
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a sub-terrain by its row and column index."""
        num_cols = self.cfg.num_cols
        if isinstance(row_ids, torch.Tensor):
            row_ids = row_ids.cpu().numpy()
        if isinstance(col_ids, torch.Tensor):
            col_ids = col_ids.cpu().numpy()
        idx = row_ids * num_cols + col_ids
        if isinstance(idx, np.ndarray):
            return [
                self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None for i in idx
            ]
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            return self._subterrain_specific_cfgs[i] if 0 <= i < len(self._subterrain_specific_cfgs) else None
        return None


class GuardedTerrainGenerator(FiledTerrainGenerator):
    """Terrain generator with surrounding terrain rings and an outer wall."""

    cfg: GuardedTerrainGeneratorCfg

    def __init__(self, cfg: GuardedTerrainGeneratorCfg, device: str = "cpu"):
        super().__init__(cfg, device)
        core_mesh_count = len(self.terrain_meshes)
        self._append_guard_rings()
        self._append_outer_walls()
        self.terrain_mesh = trimesh.util.concatenate(
            [self.terrain_mesh, *self.terrain_meshes[core_mesh_count:]]
        )

    def _append_guard_rings(self) -> None:
        tile_x, tile_y = self.cfg.size
        if self.cfg.guard_width <= 0.0:
            raise ValueError("guard_width must be positive.")
        if self.cfg.guard_difficulty_levels <= 0:
            raise ValueError("guard_difficulty_levels must be positive.")
        rings_x = round(self.cfg.guard_width / tile_x)
        rings_y = round(self.cfg.guard_width / tile_y)
        if not np.isclose(rings_x * tile_x, self.cfg.guard_width) or not np.isclose(
            rings_y * tile_y, self.cfg.guard_width
        ):
            raise ValueError("guard_width must be divisible by both terrain tile dimensions.")
        if self.cfg.border_width < self.cfg.guard_width:
            raise ValueError("border_width must be greater than or equal to guard_width.")
        if not self.cfg.guard_terrains:
            raise ValueError("guard_terrains must contain at least one sub-terrain.")

        guard_cfgs = [cfg.copy() for cfg in self.cfg.guard_terrains.values()]
        for cfg in guard_cfgs:
            cfg.size = self.cfg.size
            if hasattr(cfg, "horizontal_scale"):
                cfg.horizontal_scale = self.cfg.horizontal_scale
                cfg.vertical_scale = self.cfg.vertical_scale
                cfg.slope_threshold = self.cfg.slope_threshold

        core_x = self.cfg.num_rows * tile_x
        core_y = self.cfg.num_cols * tile_y
        tile_index = 0
        for row in range(-rings_x, self.cfg.num_rows + rings_x):
            for col in range(-rings_y, self.cfg.num_cols + rings_y):
                if 0 <= row < self.cfg.num_rows and 0 <= col < self.cfg.num_cols:
                    continue
                guard_cfg = guard_cfgs[tile_index % len(guard_cfgs)]
                level = tile_index % self.cfg.guard_difficulty_levels
                fraction = (level + 0.5) / self.cfg.guard_difficulty_levels
                lower, upper = self.cfg.difficulty_range
                difficulty = lower + (upper - lower) * fraction
                if self.cfg.native_collision:
                    mesh, _ = self._get_terrain_mesh(difficulty, guard_cfg)
                    self._subterrain_specific_cfgs.pop()  # guard tiles are not curriculum cells
                    for part, is_hf in self._pending_components:
                        part.apply_translation((row * tile_x - core_x / 2, col * tile_y - core_y / 2, 0))
                        self.collision_components.append((part, is_hf))
                    self._pending_components = []
                else:
                    mesh, _ = TerrainGenerator._get_terrain_mesh(self, difficulty, guard_cfg)
                transform = np.eye(4)
                transform[:2, -1] = (
                    (row + 0.5) * tile_x - core_x / 2.0,
                    (col + 0.5) * tile_y - core_y / 2.0,
                )
                mesh.apply_transform(transform)
                self.terrain_meshes.append(mesh)
                tile_index += 1

    def _append_outer_walls(self) -> None:
        if self.cfg.outer_wall_height <= 0.0:
            return
        if self.cfg.outer_wall_thickness <= 0.0:
            raise ValueError("outer_wall_thickness must be positive when walls are enabled.")

        outer_x = self.cfg.num_rows * self.cfg.size[0] + 2.0 * self.cfg.border_width
        outer_y = self.cfg.num_cols * self.cfg.size[1] + 2.0 * self.cfg.border_width
        half_height = self.cfg.outer_wall_height / 2.0
        half_thickness = self.cfg.outer_wall_thickness / 2.0
        wall_specs = (
            (
                (outer_x / 2.0 + half_thickness, 0.0, half_height),
                (self.cfg.outer_wall_thickness, outer_y + 2.0 * self.cfg.outer_wall_thickness),
            ),
            (
                (-outer_x / 2.0 - half_thickness, 0.0, half_height),
                (self.cfg.outer_wall_thickness, outer_y + 2.0 * self.cfg.outer_wall_thickness),
            ),
            (
                (0.0, outer_y / 2.0 + half_thickness, half_height),
                (outer_x, self.cfg.outer_wall_thickness),
            ),
            (
                (0.0, -outer_y / 2.0 - half_thickness, half_height),
                (outer_x, self.cfg.outer_wall_thickness),
            ),
        )
        for position, size_xy in wall_specs:
            transform = trimesh.transformations.translation_matrix(position)
            wall = trimesh.creation.box(
                extents=(*size_xy, self.cfg.outer_wall_height), transform=transform
            )
            wall.visual.face_colors = np.array([128, 128, 128, 255], dtype=np.uint8)
            self.terrain_meshes.append(wall)
            if self.cfg.native_collision:
                self.collision_components.append((wall.copy(), False))
