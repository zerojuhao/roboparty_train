# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Load and visualize a RoboLab robot, then print its Isaac Lab joint/link order.

Optionally visualize parkour feet ``FEET_VOLUME_POINTS_GRID`` from the selected robot cfg.

Example usage:

.. code-block:: bash

    python scripts/tools/visualize_robot.py --robot rp1
    python scripts/tools/visualize_robot.py --robot rpo --show-volume-points
    python scripts/tools/visualize_robot.py --robot rp1 --no-show-volume-points
    python scripts/tools/visualize_robot.py --robot rp1 --show-height-scanner-rays
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize RP1/RPO and optional parkour feet volume points.")
parser.add_argument(
    "--robot",
    type=str,
    choices=("rp1", "rpo"),
    default="rp1",
    help="Robot model to spawn (default: rp1).",
)
parser.add_argument(
    "--show-volume-points",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Visualize FEET_VOLUME_POINTS_GRID on ankle_roll_link (default: True).",
)
parser.add_argument(
    "--show-height-scanner-rays",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize parkour foot height-scanner ray hits (default: True).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RPO_CFG
from robolab.sensors import VOLUME_POINTS_VISUALIZER_CFG, Grid3dPointsGeneratorCfg, VolumePointsCfg
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import FEET_VOLUME_POINTS_GRID as RP1_FEET_VOLUME_POINTS_GRID
from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import FEET_VOLUME_POINTS_GRID as RPO_FEET_VOLUME_POINTS_GRID
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

ROBOT_SPAWN_HEIGHT = 0.8

FEET_VOLUME_POINTS_VISUALIZER_CFG = VOLUME_POINTS_VISUALIZER_CFG.replace(
    prim_path="/Visuals/feetVolumePoints",
)

# Matches parkour_env_cfg left/right_height_scanner (mesh path adapted for this scene's ground).
FOOT_HEIGHT_SCANNER_CFG = RayCasterCfg(
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.20, 0.03]),
    mesh_prim_paths=["/World/defaultGroundPlane"],
    update_period=0.02,
    visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/RayCaster",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=0.005,  # 改这里，单位米
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    ),
)


def get_robot_setup(robot_name: str) -> tuple[ArticulationCfg, Grid3dPointsGeneratorCfg, str]:
    """Return robot cfg, feet volume grid, and display label."""
    if robot_name == "rpo":
        robot_cfg = RPO_CFG
        feet_grid = RPO_FEET_VOLUME_POINTS_GRID
        label = "RPO (rpo_parkour_env_cfg.FEET_VOLUME_POINTS_GRID)"
    else:
        robot_cfg = RP1_24DOF_CFG
        feet_grid = RP1_FEET_VOLUME_POINTS_GRID
        label = "RP1 (rp1_parkour_env_cfg.FEET_VOLUME_POINTS_GRID)"
    robot_cfg.init_state.pos = (0.0, 0.0, ROBOT_SPAWN_HEIGHT)
    return robot_cfg, feet_grid, label


def make_robot_scene_cfg(
    robot_cfg: ArticulationCfg,
    feet_grid: Grid3dPointsGeneratorCfg,
    show_volume_points: bool,
    show_height_scanner_rays: bool,
):
    """Build scene cfg for the selected robot and volume-point options."""

    @configclass
    class RobotSceneCfg(InteractiveSceneCfg):
        """Scene with ground, lighting, and the selected robot."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if show_volume_points:
            feet_volume_points = VolumePointsCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
                points_generator=feet_grid,
                debug_vis=True,
                visualizer_cfg=FEET_VOLUME_POINTS_VISUALIZER_CFG,
            )

        if show_height_scanner_rays:
            left_height_scanner = FOOT_HEIGHT_SCANNER_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
                debug_vis=True,
            )
            right_height_scanner = FOOT_HEIGHT_SCANNER_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
                debug_vis=True,
            )

    return RobotSceneCfg


def print_joint_order(robot: Articulation) -> None:
    """Print Isaac Lab joint names."""
    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)

    print("\n" + "=" * 72)
    print(f"Isaac Lab joint order ({num_joints} DoF)")
    print("=" * 72)
    for name in joint_names:
        print(f"  - {name}")
    print("=" * 72)
    for idx, name in enumerate(joint_names):
        print(f"  [{idx:2d}] {name}")
    print("=" * 72 + "\n")


def print_link_order(robot: Articulation) -> None:
    """Print Isaac Lab body names."""
    body_names = list(robot.body_names)
    num_bodies = len(body_names)

    print("\n" + "=" * 72)
    print(f"Isaac Lab link order ({num_bodies} links)")
    print("=" * 72)
    for idx, name in enumerate(body_names):
        comma = "," if idx < num_bodies - 1 else ""
        print(f"    '{name}'{comma}")
    print("]")
    print("=" * 72)
    for idx, name in enumerate(body_names):
        print(f"  [{idx:2d}] {name}")
    print("=" * 72 + "\n")


def print_feet_volume_points_info(scene: InteractiveScene, grid: Grid3dPointsGeneratorCfg, label: str) -> None:
    """Print parkour feet volume-point grid info."""
    feet_sensor = scene["feet_volume_points"]

    print("\n" + "=" * 72)
    print(f"Parkour feet volume points: {label}")
    print("=" * 72)
    print(f"  grid x: [{grid.x_min}, {grid.x_max}] x{grid.x_num}")
    print(f"  grid y: [{grid.y_min}, {grid.y_max}] x{grid.y_num}")
    print(f"  grid z: [{grid.z_min}, {grid.z_max}] x{grid.z_num}")
    print(f"  bodies ({feet_sensor.data.point_num_each_body} pts/link): {list(feet_sensor.body_names)}")
    print("=" * 72 + "\n")


def print_height_scanner_info(scene: InteractiveScene) -> None:
    """Print parkour foot height-scanner layout (matches parkour_env_cfg)."""
    left = scene["left_height_scanner"]
    right = scene["right_height_scanner"]
    pattern = FOOT_HEIGHT_SCANNER_CFG.pattern_cfg

    print("\n" + "=" * 72)
    print("Parkour foot height scanners (parkour_env_cfg left/right_height_scanner)")
    print("=" * 72)
    print(f"  offset: {FOOT_HEIGHT_SCANNER_CFG.offset.pos}")
    print(f"  grid: resolution={pattern.resolution} m, size={pattern.size} -> {left.num_rays} rays/foot")
    print(f"  left prim:  {left.cfg.prim_path}")
    print(f"  right prim: {right.cfg.prim_path}")
    print("  debug markers show ray hit points on the ground")
    print("=" * 72 + "\n")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Render loop: hold the robot at its default pose without physics."""
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
    robot_cfg, feet_grid, grid_label = get_robot_setup(args_cli.robot)
    scene_cfg = make_robot_scene_cfg(
        robot_cfg, feet_grid, args_cli.show_volume_points, args_cli.show_height_scanner_rays
    )

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(scene_cfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    print_joint_order(robot)
    print_link_order(robot)

    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] += scene.env_origins
    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(sim.get_physics_dt())

    if args_cli.show_volume_points:
        print_feet_volume_points_info(scene, feet_grid, grid_label)

    if args_cli.show_height_scanner_rays:
        print_height_scanner_info(scene)

    if not args_cli.headless:
        root_pos = robot.data.default_root_state[0, :3].cpu().tolist()
        sim.set_camera_view(
            [root_pos[0] + 2.5, root_pos[1] + 2.5, root_pos[2] + 0.8],
            root_pos,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
