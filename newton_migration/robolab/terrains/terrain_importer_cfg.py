from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .virtual_obstacle import VirtualObstacleCfg


@configclass
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    """The inherited class to use for the terrain importer."""

    virtual_obstacles: dict[str, VirtualObstacleCfg] = {}
    """The virtual obstacles to use for the terrain importer."""

    terrain_type: Literal["generator", "plane", "usd", "hacked_generator"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane", "usd", and "generator".

    ## NOTE
    The TerrainImporter of this package has some dedicated hack to fit the self-defined tasks.
    We add a "hacked_generator" option to hack and run our own terrain generator implementation.
    """
