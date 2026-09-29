# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate HIGH_PLATFORM_45x60_CFG in Isaac Lab and save an inspection image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Screenshot the 45x60 high-platform terrain.")
parser.add_argument(
    "--output_file",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "data/motions/rp1_sim2sim_executed_lab/high_platforms_45x60.png"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from robolab.terrains import TerrainImporterCfg
from robolab.tasks.manager_based.parkour.terrain_generator_cfg import HIGH_PLATFORM_45x60_CFG


@configclass
class PreviewSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HIGH_PLATFORM_45x60_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.95)),
    )


def main() -> None:
    out_path = Path(args_cli.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    scene = InteractiveScene(PreviewSceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()
    print("[INFO] Terrain imported in Isaac Lab", flush=True)

    gen = scene.terrain.terrain_generator
    cfg = HIGH_PLATFORM_45x60_CFG.sub_terrains["high_platforms_45x60"]
    hx = HIGH_PLATFORM_45x60_CFG.horizontal_scale
    hy = HIGH_PLATFORM_45x60_CFG.vertical_scale
    size_x, size_y = HIGH_PLATFORM_45x60_CFG.size
    mesh = gen.terrain_mesh
    verts = np.asarray(mesh.vertices)
    z = verts[:, 2]
    print(
        f"[INFO] mesh vertices={len(verts)} xyz=[{verts.min(0)}, {verts.max(0)}] "
        f"height={cfg.platform_height_range[0]} band={cfg.band_width} gap={cfg.gap_width}",
        flush=True,
    )

    from robolab.terrains.height_field.hf_terrains import perlin_platforms_terrain

    fn = perlin_platforms_terrain
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    hf_cfg = cfg.copy()
    hf_cfg.size = HIGH_PLATFORM_45x60_CFG.size
    hf_cfg.horizontal_scale = hx
    hf_cfg.vertical_scale = hy
    hf_cfg.slope_threshold = HIGH_PLATFORM_45x60_CFG.slope_threshold
    hf = fn(1.0, hf_cfg).astype(np.float64) * hy
    xs = np.arange(hf.shape[0]) * hx - 0.5 * size_x
    ys = np.arange(hf.shape[1]) * hx - 0.5 * size_y
    mid = hf.shape[1] // 2
    slice_x = xs
    slice_z = hf[:, mid]
    print(f"[INFO] heightfield z[min,max]=[{hf.min():.3f},{hf.max():.3f}] shape={hf.shape}", flush=True)

    fig = plt.figure(figsize=(14, 6.2))
    fig.suptitle(
        "high_platforms_45x60  (height=0.45 m, deck=0.60 m, gap=1.00 m)",
        fontsize=13,
    )
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    step = 4
    xx, yy = np.meshgrid(xs[::step], ys[::step], indexing="ij")
    ax0.plot_surface(
        xx, yy, hf[::step, ::step],
        cmap=cm.terrain, linewidth=0, antialiased=True, vmin=0.0, vmax=0.5,
    )
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_zlabel("z (m)")
    ax0.set_zlim(0.0, 0.8)
    ax0.view_init(elev=28, azim=-55)
    ax0.set_title("Isaac Lab heightfield (3D)")

    ax1 = fig.add_subplot(1, 2, 2)
    im = ax1.imshow(
        hf.T,
        origin="lower",
        extent=(-0.5 * size_x, 0.5 * size_x, -0.5 * size_y, 0.5 * size_y),
        cmap=cm.terrain,
        vmin=0.0,
        vmax=0.5,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="height (m)")
    ax1.axhline(0.0, color="w", lw=0.6, ls="--", alpha=0.7)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("top-down height")
    ax1.text(
        0.02, 0.98,
        f"z max = {hf.max():.3f} m\nband = {cfg.band_width:.2f} m\ngap = {cfg.gap_width:.2f} m",
        transform=ax1.transAxes, va="top", ha="left",
        color="k", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    slice_path = out_path.with_name(out_path.stem + "_slice.png")
    fig2, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(slice_x, slice_z, color="tab:blue", lw=1.6)
    ax.axhline(0.45, color="tab:red", ls="--", lw=1.0, label="0.45 m")
    ax.set_xlabel("x (m), y=0 slice")
    ax.set_ylabel("height (m)")
    ax.set_ylim(-0.05, 0.6)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # mark a full raised ring (skip the outer ring clipped by border_width)
    raised = slice_z > 0.2
    runs = []
    i = 0
    while i < len(raised):
        if not raised[i]:
            i += 1
            continue
        j = i
        while j < len(raised) and raised[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    target = cfg.band_width
    if runs:
        i0, i1 = min(runs, key=lambda r: abs((slice_x[r[1]] - slice_x[r[0]]) - target))
        x0, x1 = float(slice_x[i0]), float(slice_x[i1])
        ax.annotate(
            f"band = {x1 - x0:.2f} m",
            xy=(0.5 * (x0 + x1), 0.45),
            xytext=(x1 + 0.15, 0.52),
            arrowprops=dict(arrowstyle="->", color="k"),
            fontsize=9,
        )
        ax.axvspan(x0, x1, color="tab:orange", alpha=0.2)
        pos_runs = [r for r in runs if slice_x[r[0]] > 0]
        if len(pos_runs) >= 2:
            g0 = float(slice_x[pos_runs[0][1]])
            g1 = float(slice_x[pos_runs[1][0]])
            ax.annotate(
                f"gap = {g1 - g0:.2f} m",
                xy=(0.5 * (g0 + g1), 0.0),
                xytext=(g1 + 0.1, 0.22),
                arrowprops=dict(arrowstyle="->", color="k"),
                fontsize=9,
            )
            ax.axvspan(g0, g1, color="tab:green", alpha=0.15)
    fig2.tight_layout()
    fig2.savefig(slice_path, dpi=140)
    plt.close(fig2)

    print(f"[INFO] Wrote {out_path}", flush=True)
    print(f"[INFO] Wrote {slice_path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
