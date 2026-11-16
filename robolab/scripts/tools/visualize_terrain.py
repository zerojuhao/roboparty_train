# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize parkour terrains defined in terrain_generator_cfg.py.

Spawns one RP1 at each sub-terrain tile origin (terrain_origins[row, col]) in its
default standing pose and holds them static (no physics) as size/pose references.

Example usage:

.. code-block:: bash

    python scripts/tools/visualize_terrain.py
    python scripts/tools/visualize_terrain.py --sub_terrain pyramid_stairs
    python scripts/tools/visualize_terrain.py --color_scheme height --use_curriculum
    python scripts/tools/visualize_terrain.py --tick_spacing 0.5
    python scripts/tools/visualize_terrain.py --no_debug_overlay
    python scripts/tools/visualize_terrain.py --virtual_edges
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize parkour terrain_generator_cfg terrains.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme for terrain meshes.",
)
parser.add_argument("--use_curriculum", action="store_true", default=True, help="Enable terrain curriculum.")
parser.add_argument(
    "--sub_terrain",
    type=str,
    default=None,
    help="Visualize only this sub-terrain name (e.g. trapezoid_stairs).",
)
parser.add_argument("--num_rows", type=int, default=5, help="Number of terrain rows.")
parser.add_argument("--num_cols", type=int, default=10, help="Number of terrain columns.")
parser.add_argument("--tick_spacing", type=float, default=1.0, help="Grid tick spacing in meters (local frame).")
parser.add_argument(
    "--no_debug_overlay",
    action="store_true",
    default=False,
    help="Disable XY grid ticks and terrain-name overlay.",
)
parser.add_argument(
    "--virtual_edges",
    action="store_true",
    default=True,
    help="Enable virtual edge obstacles (GreedyconcatEdgeCylinder) for visualization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from robolab.assets.robots.roboparty import RP1_24DOF_CFG,RPO_CFG
from robolab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporter, TerrainImporterCfg
from robolab.terrains.terrain_generator import FiledTerrainGenerator

from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG

# Match rp1_parkour_env_cfg standing height above each terrain origin.
ROBOT_SPAWN_HEIGHT = 0.85

# Grid overlay colors: border, minor lines, +X axis, +Y axis.
_COLOR_BORDER = (0.85, 0.85, 0.85)
_COLOR_MINOR = (0.45, 0.45, 0.45)
_COLOR_AXIS_X = (0.95, 0.25, 0.25)
_COLOR_AXIS_Y = (0.25, 0.85, 0.25)


class _LineBatch:
    """Collects USD basis-curve segments and commits them as one prim."""

    def __init__(self) -> None:
        self.points: list[Gf.Vec3f] = []
        self.counts: list[int] = []
        self.colors: list[Gf.Vec3f] = []
        self.widths: list[float] = []

    def add(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        self.points.extend([Gf.Vec3f(*start), Gf.Vec3f(*end)])
        self.counts.append(2)
        self.colors.extend([Gf.Vec3f(*color), Gf.Vec3f(*color)])
        self.widths.extend([width, width])

    def commit(self, stage, prim_path: str) -> None:
        if not self.points:
            return
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateBasisAttr(UsdGeom.Tokens.bspline)
        curves.CreatePointsAttr(Vt.Vec3fArray(self.points))
        curves.CreateCurveVertexCountsAttr(self.counts)
        curves.CreateWidthsAttr(Vt.FloatArray(self.widths))
        curves.CreateDisplayColorAttr(Vt.Vec3fArray(self.colors))


def _tick_values(half_extent: float, spacing: float) -> np.ndarray:
    return np.arange(-half_extent, half_extent + spacing * 0.5, spacing)


def _global_grid_z(vertices: np.ndarray, *, wall_height: float = 5.0, clearance: float = 0.05) -> float:
    """Single overlay height for all tiles so adjacent grid lines stay coplanar."""
    z_vals = vertices[:, 2]
    walkable = z_vals[z_vals < wall_height - 0.25]
    if walkable.size == 0:
        walkable = z_vals
    return float(walkable.max()) + clearance


def _spawn_tile_grid(
    stage,
    prim_root: str,
    ox: float,
    oy: float,
    half_x: float,
    half_y: float,
    z: float,
    tick_spacing: float,
) -> None:
    """Draw border, minor grid lines, and axis highlights for one tile."""
    lines = _LineBatch()
    corners = [
        (ox - half_x, oy - half_y, z),
        (ox + half_x, oy - half_y, z),
        (ox + half_x, oy + half_y, z),
        (ox - half_x, oy + half_y, z),
    ]
    for i in range(4):
        lines.add(corners[i], corners[(i + 1) % 4], _COLOR_BORDER, 0.03)

    for local_x in _tick_values(half_x, tick_spacing):
        wx = ox + local_x
        on_axis = abs(local_x) < 1e-6
        color = _COLOR_AXIS_X if on_axis else _COLOR_MINOR
        width = 0.05 if on_axis else 0.02
        lines.add((wx, oy - half_y, z), (wx, oy + half_y, z), color, width)

    for local_y in _tick_values(half_y, tick_spacing):
        wy = oy + local_y
        on_axis = abs(local_y) < 1e-6
        color = _COLOR_AXIS_Y if on_axis else _COLOR_MINOR
        width = 0.05 if on_axis else 0.02
        lines.add((ox - half_x, wy, z), (ox + half_x, wy, z), color, width)

    lines.commit(stage, f"{prim_root}/grid")


def spawn_debug_overlay(terrain_importer, tick_spacing: float = 1.0) -> None:
    """Draw per-tile local XY grids. Local origin = terrain_origins[row, col]."""
    terrain_gen = terrain_importer.terrain_generator
    if not isinstance(terrain_gen, FiledTerrainGenerator) or terrain_gen.subterrain_index_grid is None:
        raise RuntimeError("FiledTerrainGenerator with subterrain_index_grid is required for debug overlay.")

    origins = terrain_importer.terrain_origins
    if origins is None:
        origins = terrain_gen.terrain_origins
    if origins is None:
        raise RuntimeError("terrain_origins is not available.")
    origins = origins.detach().cpu().numpy() if hasattr(origins, "detach") else np.asarray(origins)

    cfg = terrain_gen.cfg
    half_x, half_y = cfg.size[0] * 0.5, cfg.size[1] * 0.5
    sub_names = list(cfg.sub_terrains.keys())
    vertices = np.asarray(terrain_gen.terrain_mesh.vertices)
    stage = sim_utils.get_current_stage()
    grid_z = _global_grid_z(vertices)

    print("[INFO] Terrain debug overlay (local frame origin = terrain_origins[row, col]):")
    print(f"[INFO]   tile size = ({cfg.size[0]:.2f}, {cfg.size[1]:.2f}) m, tick_spacing = {tick_spacing:.2f} m")
    print(f"[INFO]   local x in [-{half_x:.2f}, {half_x:.2f}], local y in [-{half_y:.2f}, {half_y:.2f}]")
    print(f"[INFO]   grid z = {grid_z:.2f} m (shared across all tiles)")
    print("[INFO]   Red = local +X, Green = local +Y")

    for row in range(cfg.num_rows):
        for col in range(cfg.num_cols):
            ox, oy, oz = map(float, origins[row, col])

            sub_index = int(terrain_gen.subterrain_index_grid[row, col])
            terrain_name = sub_names[sub_index]
            print(
                f"[INFO]   ({row:02d},{col:02d}) {terrain_name:28s} "
                f"world_origin=({ox:7.2f}, {oy:7.2f}, {oz:6.2f})"
            )
            _spawn_tile_grid(
                stage,
                f"/Visuals/TerrainDebug/r{row:02d}_c{col:02d}",
                ox,
                oy,
                half_x,
                half_y,
                grid_z,
                tick_spacing,
            )


def design_scene():
    """Design the scene."""
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(
        curriculum=args_cli.use_curriculum,
        color_scheme=args_cli.color_scheme,
        num_rows=args_cli.num_rows,
        num_cols=args_cli.num_cols,
    )
    if args_cli.sub_terrain is not None:
        if args_cli.sub_terrain not in terrain_gen_cfg.sub_terrains:
            available = ", ".join(sorted(terrain_gen_cfg.sub_terrains.keys()))
            raise ValueError(f"Unknown sub-terrain '{args_cli.sub_terrain}'. Available: {available}")
        sub_cfg = terrain_gen_cfg.sub_terrains[args_cli.sub_terrain]
        terrain_gen_cfg.sub_terrains = {args_cli.sub_terrain: sub_cfg.replace(proportion=1.0)}
    # Disable border walls for clearer visualization (match play cfg).
    for sub_cfg in terrain_gen_cfg.sub_terrains.values():
        if hasattr(sub_cfg, "wall_prob"):
            sub_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]

    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=args_cli.num_rows * args_cli.num_cols,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
        virtual_obstacles=(
            {
                "edges": GreedyconcatEdgeCylinderCfg(
                    cylinder_radius=0.03,
                    min_points=2,
                ),
            }
            if args_cli.virtual_edges
            else {}
        ),
    )
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    else:
        terrain_importer_cfg.visual_material = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        )

    return TerrainImporter(terrain_importer_cfg)


def _terrain_origins_flat(terrain_importer, device: str) -> torch.Tensor:
    """Return (num_rows * num_cols, 3) world origins, row-major over the tile grid."""
    origins = terrain_importer.terrain_origins
    if origins is None:
        origins = terrain_importer.terrain_generator.terrain_origins
    if origins is None:
        raise RuntimeError("terrain_origins is not available.")
    if hasattr(origins, "detach"):
        origins = origins.to(device=device)
    else:
        origins = torch.as_tensor(origins, device=device, dtype=torch.float32)
    return origins.reshape(-1, 3)


@configclass
class Rp1VisSceneCfg(InteractiveSceneCfg):
    """One RP1 per env; env origins are overridden with terrain tile origins."""

    robot: ArticulationCfg = RPO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=RPO_CFG.init_state.replace(pos=(0.0, 0.0, ROBOT_SPAWN_HEIGHT)),
    )


def hold_robots_static(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Render loop: keep each RP1 at its terrain origin in default pose without physics."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] += scene.env_origins
    default_joint_pos = robot.data.default_joint_pos
    default_joint_vel = robot.data.default_joint_vel

    while simulation_app.is_running():
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        scene.write_data_to_sim()
        sim.forward()
        sim.render()
        scene.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[12.0, 12.0, 8.0], target=[0.0, 0.0, 0.0])

    terrain_importer = design_scene()
    num_envs = args_cli.num_rows * args_cli.num_cols
    scene = InteractiveScene(Rp1VisSceneCfg(num_envs=num_envs, env_spacing=2.0))
    sim.reset()

    tile_origins = _terrain_origins_flat(terrain_importer, sim.device)
    scene.env_origins[:] = tile_origins

    robot: Articulation = scene["robot"]
    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] += scene.env_origins
    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(sim.get_physics_dt())

    if not args_cli.no_debug_overlay:
        spawn_debug_overlay(terrain_importer, tick_spacing=args_cli.tick_spacing)
    print("[INFO]: Parkour terrain visualization ready.")
    print(f"[INFO]: Sub-terrains: {list(terrain_importer.terrain_generator.cfg.sub_terrains.keys())}")
    print(f"[INFO]: Virtual edge obstacles: {'enabled' if args_cli.virtual_edges else 'disabled'}")
    print(
        f"[INFO]: Spawned {num_envs} RP1 robots at each terrain origin "
        f"(standing height +{ROBOT_SPAWN_HEIGHT} m)."
    )

    hold_robots_static(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
