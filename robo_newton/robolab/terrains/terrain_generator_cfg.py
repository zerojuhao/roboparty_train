from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils import configclass

from .terrain_generator import FiledTerrainGenerator, GuardedTerrainGenerator


@configclass
class FiledTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    class_type: type = FiledTerrainGenerator

    native_collision: bool = False
    """Retain authored Newton primitives and raw heights; bypass lossy mesh rasterization."""

    one_col_per_subterrain: bool = False
    """Whether to use one column per sub-terrain type. Defaults to False.

    If True, ``num_cols`` is set to ``len(sub_terrains)`` (one type per column), and
    environment counts follow each sub-terrain's ``proportion``. If False, columns are
    mapped by cumulative proportion and environments are split uniformly (Isaac Lab default).
    """


@configclass
class GuardedTerrainGeneratorCfg(FiledTerrainGeneratorCfg):
    """Terrain grid surrounded by generated guard rings and optional walls."""

    class_type: type = GuardedTerrainGenerator

    guard_width: float = MISSING
    """Width of generated terrain surrounding the core grid."""

    guard_terrains: dict[str, SubTerrainBaseCfg] = MISSING
    """Sub-terrains cycled through the guard rings."""

    guard_difficulty_levels: int = 10
    """Number of evenly spaced difficulty levels used by guard tiles."""

    outer_wall_height: float = 0.0
    """Outer wall height. A non-positive value disables walls."""

    outer_wall_thickness: float = 0.05
    """Outer wall thickness."""
