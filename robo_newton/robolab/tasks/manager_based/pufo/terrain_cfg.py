"""Compact mixed terrain used by the isolated PUFO visual task."""

from robolab.terrains import (
    PerlinPlaneTerrainCfg,
    PerlinPlatformsTerrainCfg,
    PerlinPyramidSlopedTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
)
from robolab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg


def _perlin(noise_scale):
    return PerlinPlaneTerrainCfg(
        noise_scale=noise_scale,
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
    )


PUFO_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    curriculum=True,
    one_col_per_subterrain=True,
    sub_terrains={
        "flat": PerlinPlaneTerrainCfg(
            proportion=0.20,
            noise_scale=[0.0, 0.01],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        "rough": PerlinPlaneTerrainCfg(
            proportion=0.20,
            noise_scale=[0.0, 0.05],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        "slope": PerlinPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.05, 0.14),
            platform_width=1.5,
            perlin_cfg=_perlin(0.02),
        ),
        "stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=1.5,
            perlin_cfg=_perlin(0.01),
        ),
        "platforms": PerlinPlatformsTerrainCfg(
            proportion=0.20,
            platform_height_range=(0.05, 0.18),
            band_width=0.5,
            gap_width=0.25,
            platform_width=1.5,
            perlin_cfg=_perlin(0.01),
        ),
    },
)
