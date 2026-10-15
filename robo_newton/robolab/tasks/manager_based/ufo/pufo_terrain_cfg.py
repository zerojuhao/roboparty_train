"""RP1 PUFO guarded mixed terrain, with H200 primitives on 10 m tiles."""

from __future__ import annotations

from isaaclab.terrains.trimesh import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
)

from robolab.terrains import (
    BorderedMeshRandomGridTerrainCfg,
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinInvertedPyramidSlopedTerrainCfg,
    PerlinPlaneTerrainCfg,
)
from robolab.terrains.terrain_generator_cfg import GuardedTerrainGeneratorCfg


PUFO_TILE_SIZE = (10.0, 10.0)
PUFO_TILE_FLAT_BORDER_WIDTH = 0.5
PUFO_NONFLAT_GUARD_WIDTH = 10.0
PUFO_FLAT_SAFETY_WIDTH = 2.0
PUFO_TERRAIN_BORDER_WIDTH = PUFO_NONFLAT_GUARD_WIDTH + PUFO_FLAT_SAFETY_WIDTH


PUFO_TERRAIN_CFG = GuardedTerrainGeneratorCfg(
    seed=0,
    size=PUFO_TILE_SIZE,
    border_width=PUFO_TERRAIN_BORDER_WIDTH,
    num_rows=10,
    num_cols=7,
    horizontal_scale=0.2,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    one_col_per_subterrain=True,
    guard_width=PUFO_NONFLAT_GUARD_WIDTH,
    guard_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.5,
            noise_scale=[0.02, 0.06],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
            interior_height_offset=0.05,
        ),
        "boxes": PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.5,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=50,
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
            interior_height_offset=0.05,
        ),
    },
    outer_wall_height=5.0,
    outer_wall_thickness=0.05,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.20,
            convert_to_heightfield=True,
        ),
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.10,
            convert_to_heightfield=True,
            noise_scale=[0.02, 0.06],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
        ),
        "low_stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            convert_to_heightfield=False,
            step_height_range=(0.10, 0.15),
            step_width=0.30,
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
        ),
        "low_stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            convert_to_heightfield=False,
            step_height_range=(0.10, 0.15),
            step_width=0.30,
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
        ),
        "low_platforms": BorderedMeshRandomGridTerrainCfg(
            proportion=0.10,
            convert_to_heightfield=False,
            grid_height_range=(0.03, 0.15),
            grid_width=0.8,
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            merge_similar_heights=True,
            height_merge_threshold=0.025,
            max_merge_distance=3,
        ),
        "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            convert_to_heightfield=True,
            slope_range=(0.0, 0.2),
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
        ),
        "boxes": PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            convert_to_heightfield=True,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=50,
            platform_width=1.2,
            border_width=PUFO_TILE_FLAT_BORDER_WIDTH,
            exact_flat_border=True,
        ),
    },
)
