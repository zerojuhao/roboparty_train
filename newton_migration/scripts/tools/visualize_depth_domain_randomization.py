# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture one parkour-style ray-caster depth frame and visualize each domain-
randomization noise applied individually on the cropped image.

Pipeline mirrors ``parkour_env_cfg.SceneCfg.camera.noise_pipeline``:
  crop_and_resize -> [single noise] -> depth_normalization

Uses a lightweight InteractiveScene (robot + parkour terrain + camera) instead of
the full Parkour MDP env, so startup is fast and does not load motion/AMP managers.

Example usage:

.. code-block:: bash

    python scripts/tools/visualize_depth_domain_randomization.py --headless
    python scripts/tools/visualize_depth_domain_randomization.py --robot rp1 --sub_terrain pyramid_stairs --headless
    python scripts/tools/visualize_depth_domain_randomization.py --sub_terrain trapezoid_stairs --headless
    python scripts/tools/visualize_depth_domain_randomization.py --output_dir /tmp/depth_dr --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Visualize individual depth domain-randomization effects on cropped camera images."
)
parser.add_argument("--robot", type=str, choices=("rp1", "rpo"), default="rp1", help="Robot model.")
parser.add_argument(
    "--sub_terrain",
    type=str,
    default="pyramid_stairs",
    help="Parkour sub-terrain name from terrain_generator_cfg (default: pyramid_stairs).",
)
parser.add_argument(
    "--robot_offset",
    type=float,
    nargs=3,
    default=(1.5, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Extra XYZ offset from terrain tile origin so the cropped near-field sees structure "
    "(default: 1.5 0 0).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="logs/depth_domain_randomization",
    help="Directory to save images (cleared each run). Default: logs/depth_domain_randomization.",
)
parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible noise samples.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=10,
    help="Sim steps before capturing the depth frame.",
)
parser.add_argument(
    "--upsample",
    type=int,
    default=16,
    help="Nearest-neighbor upsample factor for saved PNGs (raw crop is ~18x32).",
)
parser.add_argument(
    "--force_apply",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Force apply_probability=1.0 so each noise always fires (default: True).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force line-buffered logs when redirected to a file.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from robolab.assets.robots.roboparty import PR1_LINKS, RP1_24DOF_CFG, RPO_CFG, RPO_LINKS
from robolab.sensors import NoisyGroupedRayCasterCameraCfg, get_link_prim_targets
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG
from robolab.terrains import TerrainImporterCfg
from robolab.utils.noise import (
    CropAndResizeCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    PerlinNoiseCfg,
    PixelFailureNoiseCfg,
    RandomConvNoiseCfg,
    ScaleRandomizationNoiseCfg,
    StereoFusionNoiseCfg,
)

ROBOT_SPAWN_HEIGHT = 0.85

# Keep in sync with parkour_env_cfg.SceneCfg.camera.noise_pipeline (crop + DR + normalize).
CROP_CFG = CropAndResizeCfg(crop_region=(18, 0, 16, 16))
DEPTH_NORM_CFG = DepthNormalizationCfg(
    depth_range=(0.0, 2.5),
    normalize=True,
    output_range=(0.0, 1.0),
)

DOMAIN_NOISES: dict[str, object] = {
    "scale_randomization": ScaleRandomizationNoiseCfg(
        apply_probability=0.7,
        scale_min=0.95,
        scale_max=1.05,
    ),
    "stereo_fusion": StereoFusionNoiseCfg(
        apply_probability=0.5,
        disparity_grad_threshold=0.09,
        texture_var_threshold=4e-4,
        hole_probability=0.02,
        hole_kernel_size=1,
        hole_value=2.5,
    ),
    "random_conv": RandomConvNoiseCfg(
        apply_probability=0.4,
        kernel_std=0.01,
        center_weight=1.0,
    ),
    "perlin_noise": PerlinNoiseCfg(
        apply_probability=0.6,
        octaves=4,
        base_frequency=8.0,
        lacunarity=2.0,
        persistence=0.5,
        amplitude=1.0,
        noise_std=0.025,
    ),
    "pixel_failures": PixelFailureNoiseCfg(
        apply_probability=0.7,
        dead_pixel_prob=2e-3,
        saturated_pixel_prob=2e-3,
        dead_value=0.0,
        saturated_value=2.5,
    ),
    "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
}


def _camera_cfg_for_robot(robot: str) -> NoisyGroupedRayCasterCameraCfg:
    """Parkour camera pattern/pose; noise applied offline in this script."""
    if robot == "rpo":
        prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        offset_pos = (0.0875, 0.01, 0.20568)
        link_targets = get_link_prim_targets(RPO_LINKS)
    else:
        prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
        offset_pos = (0.09175, 0.011, 0.3982)
        link_targets = get_link_prim_targets(PR1_LINKS)

    return NoisyGroupedRayCasterCameraCfg(
        prim_path=prim_path,
        mesh_prim_paths=["/World/ground", *link_targets],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=offset_pos,
            rot=(0.866, 0.0, 0.5, 0.0),
            convention="world",
        ),
        min_distance=0.1,
        # Capture clean metric depth; apply DR ourselves below.
        noise_pipeline={},
        data_histories={},
    )


def _make_terrain_cfg(sub_terrain: str, seed: int) -> TerrainImporterCfg:
    """Build a 1x1 parkour terrain tile (high curriculum difficulty)."""
    terrain_gen_cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrain_gen_cfg.seed = seed
    terrain_gen_cfg.curriculum = True
    # With curriculum, difficulty ≈ row/num_rows; a single hard row gives rich geometry.
    terrain_gen_cfg.num_rows = 1
    terrain_gen_cfg.num_cols = 1

    if sub_terrain not in terrain_gen_cfg.sub_terrains:
        available = ", ".join(sorted(terrain_gen_cfg.sub_terrains.keys()))
        raise ValueError(f"Unknown sub-terrain '{sub_terrain}'. Available: {available}")

    sub_cfg = terrain_gen_cfg.sub_terrains[sub_terrain]
    terrain_gen_cfg.sub_terrains = {sub_terrain: copy.deepcopy(sub_cfg).replace(proportion=1.0)}
    for cfg in terrain_gen_cfg.sub_terrains.values():
        if hasattr(cfg, "wall_prob"):
            cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={},
    )


def _make_scene_cfg(
    robot_cfg: ArticulationCfg,
    camera_cfg: NoisyGroupedRayCasterCameraCfg,
    terrain_cfg: TerrainImporterCfg,
):
    @configclass
    class DepthVizSceneCfg(InteractiveSceneCfg):
        terrain = terrain_cfg
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        camera = camera_cfg

    return DepthVizSceneCfg


def _force_apply(cfg):
    cfg = copy.deepcopy(cfg)
    if hasattr(cfg, "apply_probability"):
        cfg.apply_probability = 1.0
    return cfg


def _to_hwc_numpy(img: torch.Tensor) -> np.ndarray:
    if img.ndim == 4:
        img = img[0]
    arr = img.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _upsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def _save_depth_png(path: str, depth_01: np.ndarray, title: str, upsample: int) -> None:
    # Normalized depth in [0, 1] (0 m → 0, 2.5 m → 1). turbo is perceptually uniform
    # and keeps near/far stair bands easier to read than gray.
    vis = _upsample(np.clip(depth_01, 0.0, 1.0), upsample)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.imshow(vis, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _apply_cfg(data: torch.Tensor, cfg, env_ids: torch.Tensor) -> torch.Tensor:
    cfg = copy.deepcopy(cfg)
    cfg.device = data.device
    return cfg.func(data, cfg, env_ids)


def _normalize(data: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    return _apply_cfg(data, DEPTH_NORM_CFG, env_ids)


def main():
    output_dir = os.path.abspath(args_cli.output_dir)
    # Keep a single run folder: wipe previous outputs before writing.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    elif os.path.lexists(output_dir):
        os.remove(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    if args_cli.robot == "rpo":
        robot_cfg = RPO_CFG
    else:
        robot_cfg = RP1_24DOF_CFG
    robot_cfg.init_state.pos = (0.0, 0.0, ROBOT_SPAWN_HEIGHT)

    terrain_cfg = _make_terrain_cfg(args_cli.sub_terrain, args_cli.seed)
    camera_cfg = _camera_cfg_for_robot(args_cli.robot)
    scene_cfg_cls = _make_scene_cfg(robot_cfg, camera_cfg, terrain_cfg)

    print(f"[INFO] Creating SimulationContext on {args_cli.device}", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    print(
        f"[INFO] Building InteractiveScene (robot + {args_cli.sub_terrain} terrain + camera)...",
        flush=True,
    )
    scene = InteractiveScene(scene_cfg_cls(num_envs=1, env_spacing=2.0))
    sim.reset()

    # Place robot on the generated terrain tile origin (matches parkour spawn).
    terrain = scene["terrain"]
    origins = terrain.terrain_origins
    if origins is None and getattr(terrain, "terrain_generator", None) is not None:
        origins = terrain.terrain_generator.terrain_origins
    if origins is not None:
        tile_origin = origins.reshape(-1, 3)[0].to(device=sim.device)
        scene.env_origins[0] = tile_origin
        print(f"[INFO] Terrain tile origin: {tile_origin.tolist()}", flush=True)

    robot = scene["robot"]
    camera = scene["camera"]

    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] += scene.env_origins
    # Shift so cropped near-field (kept after top crop) sees stair/gap structure.
    offset = torch.tensor(args_cli.robot_offset, device=sim.device, dtype=root_states.dtype)
    root_states[:, :3] += offset
    print(f"[INFO] Robot offset from tile origin: {offset.tolist()}", flush=True)
    default_joint_pos = robot.data.default_joint_pos
    default_joint_vel = robot.data.default_joint_vel
    sim_dt = sim.get_physics_dt()

    print(f"[INFO] Warming up for {args_cli.warmup_steps} steps...", flush=True)
    for _ in range(args_cli.warmup_steps):
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        scene.write_data_to_sim()
        sim.forward()
        scene.update(sim_dt)

    raw = camera.data.output["distance_to_image_plane"].clone()
    env_ids = torch.arange(raw.shape[0], device=raw.device)
    cropped = _apply_cfg(raw, CROP_CFG, env_ids)

    print(f"[INFO] Raw depth shape: {tuple(raw.shape)}, cropped: {tuple(cropped.shape)}", flush=True)
    print(f"[INFO] Saving images to: {output_dir}", flush=True)

    panels: list[tuple[str, np.ndarray]] = []

    cropped_norm = _to_hwc_numpy(_normalize(cropped.clone(), env_ids))
    path = os.path.join(output_dir, "00_cropped_baseline.png")
    _save_depth_png(path, cropped_norm, "cropped (no DR)", args_cli.upsample)
    panels.append(("cropped", cropped_norm))
    print(f"  saved {path}", flush=True)

    raw_norm = _to_hwc_numpy(_normalize(raw.clone(), env_ids))
    path = os.path.join(output_dir, "00_raw_full.png")
    _save_depth_png(path, raw_norm, "raw full (pre-crop)", args_cli.upsample)
    print(f"  saved {path}", flush=True)

    for idx, (name, noise_cfg) in enumerate(DOMAIN_NOISES.items(), start=1):
        cfg = _force_apply(noise_cfg) if args_cli.force_apply else copy.deepcopy(noise_cfg)
        torch.manual_seed(args_cli.seed + idx)
        noised = _apply_cfg(cropped.clone(), cfg, env_ids)
        noised_norm = _to_hwc_numpy(_normalize(noised, env_ids))
        path = os.path.join(output_dir, f"{idx:02d}_{name}.png")
        title = name
        if args_cli.force_apply and hasattr(noise_cfg, "apply_probability"):
            title = f"{name} (forced)"
        _save_depth_png(path, noised_norm, title, args_cli.upsample)
        panels.append((name, noised_norm))
        print(f"  saved {path}", flush=True)

    torch.manual_seed(args_cli.seed)
    full = cropped.clone()
    for name, noise_cfg in DOMAIN_NOISES.items():
        cfg = _force_apply(noise_cfg) if args_cli.force_apply else copy.deepcopy(noise_cfg)
        full = _apply_cfg(full, cfg, env_ids)
    full_norm = _to_hwc_numpy(_normalize(full, env_ids))
    path = os.path.join(output_dir, f"{len(DOMAIN_NOISES) + 1:02d}_full_pipeline.png")
    _save_depth_png(path, full_norm, "full pipeline (all DR)", args_cli.upsample)
    panels.append(("full_pipeline", full_norm))
    print(f"  saved {path}", flush=True)

    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(
            _upsample(img, max(1, args_cli.upsample // 2)),
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        f"Depth DR on {args_cli.sub_terrain} (crop → single/all DR → normalize)",
        fontsize=11,
    )
    fig.tight_layout()
    grid_path = os.path.join(output_dir, "comparison_grid.png")
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {grid_path}", flush=True)

    np.savez_compressed(
        os.path.join(output_dir, "depth_tensors.npz"),
        raw_metric=_to_hwc_numpy(raw),
        cropped_metric=_to_hwc_numpy(cropped),
        cropped_normalized=cropped_norm,
        full_pipeline_normalized=full_norm,
    )
    print(f"[INFO] Done. Outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
