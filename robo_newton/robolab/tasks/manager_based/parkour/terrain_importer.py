"""Parkour terrain generation policy using the shared native importer."""

import isaaclab.sim as sim_utils

from robolab.terrains.terrain_generator import FiledTerrainGenerator
from robolab.terrains.native_importer import NativeTerrainImporter, collision_kind, merge_box_strips


class ParkourTerrainGenerator(FiledTerrainGenerator):
    """Retain authored components before concatenation and vertex welding."""

    def __init__(self, cfg, device="cpu"):
        cfg = cfg.copy()
        cfg.native_collision = True
        self.stair_collision_tiles = []
        super().__init__(cfg, device)

    def _get_terrain_mesh(self, difficulty, cfg):
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        # Inverted trapezoid stairs contain thousands of same-height infill
        # strips. Their exact union is a handful of boxes, avoiding quadratic
        # static-body collision-filter storage in Newton's builder.
        self._pending_components = merge_box_strips(self._pending_components)
        return mesh, origin

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        from isaaclab.terrains.height_field.hf_terrains_cfg import (
            HfPyramidStairsTerrainCfg, HfInvertedPyramidStairsTerrainCfg,
        )

        start = len(self.collision_components)
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if isinstance(sub_terrain_cfg, (HfPyramidStairsTerrainCfg, HfInvertedPyramidStairsTerrainCfg)):
            family = list(self.cfg.sub_terrains)[int(self.subterrain_index_grid[row, col])]
            # Keep references: the parent generator applies the final world
            # translation to these components after generating all tiles.
            self.stair_collision_tiles.append({
                "family": family, "row": row, "col": col,
                "components": self.collision_components[start:],
            })


class ParkourTerrainImporter(NativeTerrainImporter):
    def __init__(self, cfg):
        from isaaclab_newton.physics import NewtonCfg

        if isinstance(sim_utils.SimulationContext.instance().cfg.physics, NewtonCfg) and cfg.terrain_generator is not None:
            cfg = cfg.copy()
            cfg.terrain_generator.class_type = ParkourTerrainGenerator
        super().__init__(cfg)

    def validate_solver_geometry(self, model):
        from .stair_validation import validate_stair_edges

        actual = super().validate_solver_geometry(model)
        self.stair_edge_validation = validate_stair_edges(model, self.terrain_generator, self.device)
        print(f"[Parkour terrain] stair-edge ray check: {self.stair_edge_validation}", flush=True)
        return actual
