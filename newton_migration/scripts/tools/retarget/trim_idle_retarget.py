# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trim idle frames from nested sim2sim CSVs, then retarget GMR order to Isaac Lab pkls.

Idle at the start/middle of a clip is dropped by keeping contiguous moving
segments (spawn-settle and end wobble shorter than --min_moving_s are ignored).
Output pkls match rp1_24dof_lab: fps, root_pos, root_rot (wxyz), dof_pos (lab
order), loop_mode, key_body_pos.

Example:
    python robolab/scripts/tools/retarget/trim_idle_retarget.py \
        --input_dir robolab/data/motions/rp1_sim2sim_executed \
        --output_dir robolab/data/motions/rp1_sim2sim_executed_lab \
        --headless --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trim idle GMR CSVs and retarget to Isaac Lab pkls.")
parser.add_argument("--robot", type=str, default="rp1_24dof", choices=["rpo", "rp1_24dof"])
parser.add_argument(
    "--input_dir",
    type=str,
    default="robolab/data/motions/rp1_sim2sim_executed",
    help="Directory of clip folders, each containing executed.csv.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="robolab/data/motions/rp1_sim2sim_executed_lab",
    help="Directory for converted .pkl files (same layout as rp1_24dof_lab, but a separate folder).",
)
parser.add_argument(
    "--config_file",
    type=str,
    default="robolab/scripts/tools/retarget/config/rp1_24dof.yaml",
)
parser.add_argument("--loop", type=str, choices=["wrap", "clamp"], default="clamp")
parser.add_argument("--csv_fps", type=float, default=50.0, help="Fallback FPS if CSV has no sample_rate comment.")
parser.add_argument("--speed_thresh", type=float, default=0.2, help="Smoothed root+dof speed below this is idle.")
parser.add_argument("--smooth_s", type=float, default=0.2, help="Moving-average window for speed, seconds.")
parser.add_argument("--min_moving_s", type=float, default=1.0, help="Drop moving bursts shorter than this.")
parser.add_argument("--pad_s", type=float, default=0.1, help="Keep this much idle on each side of a moving span.")
parser.add_argument("--no_trim", action="store_true", help="Retarget full clips without dropping idle.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pickle
import warnings

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

if args_cli.robot == "rpo":
    from robolab.assets.robots.roboparty import RPO_CFG as ROBOT_CFG
elif args_cli.robot == "rp1_24dof":
    from robolab.assets.robots.roboparty import RP1_24DOF_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

sys.path.insert(0, str(Path(__file__).parent))
from gmr_to_lab import LoopMode, ReplayMotionsSceneCfg, load_gmr_motion_data, remap_gmr_to_lab, run_simulator


def list_executed_csvs(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted(root.rglob("executed.csv"))
    if files:
        return files
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def moving_spans(
    root_pos: np.ndarray,
    dof_pos: np.ndarray,
    fps: float,
    speed_thresh: float,
    smooth_s: float,
    min_moving_s: float,
    pad_s: float,
) -> list[tuple[int, int]]:
    n = int(root_pos.shape[0])
    dt = 1.0 / float(fps)
    v_root = np.linalg.norm(np.diff(root_pos, axis=0, prepend=root_pos[:1]), axis=1) / dt
    v_dof = np.linalg.norm(np.diff(dof_pos, axis=0, prepend=dof_pos[:1]), axis=1) / dt
    speed = v_root + 0.25 * v_dof
    win = max(1, int(round(smooth_s / dt)))
    smooth = np.convolve(speed, np.ones(win) / win, mode="same")
    moving = smooth > speed_thresh

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and bool(moving[j]) == bool(moving[i]):
            j += 1
        if moving[i]:
            runs.append((i, j))
        i = j

    pad = int(round(pad_s / dt))
    min_frames = max(2, int(round(min_moving_s / dt)))
    spans = []
    for start, end in runs:
        if end - start < min_frames:
            continue
        spans.append((max(0, start - pad), min(n, end + pad)))
    if not spans:
        return [(0, n)]
    return spans


def main() -> None:
    with open(args_cli.config_file, "r") as f:
        config = yaml.safe_load(f)
    gmr_dof_names = config["gmr_dof_names"]
    lab_dof_names = config["lab_dof_names"]
    lab_key_body_names = config["lab_key_body_names"]
    loop_mode = LoopMode.CLAMP if args_cli.loop == "clamp" else LoopMode.WRAP

    csv_files = list_executed_csvs(args_cli.input_dir)
    if not csv_files:
        print(f"No executed.csv files found in {args_cli.input_dir}")
        return

    Path(args_cli.output_dir).mkdir(parents=True, exist_ok=True)
    motion_data_dicts = []
    output_names = []
    fps_values = []

    print(f"Found {len(csv_files)} CSV files.")
    for csv_path in csv_files:
        clip_name = csv_path.parent.name if csv_path.name == "executed.csv" else csv_path.stem
        gmr = load_gmr_motion_data(str(csv_path), gmr_dof_names, csv_fps=args_cli.csv_fps)
        fps = float(gmr["fps"])
        n = int(gmr["dof_pos"].shape[0])
        if args_cli.no_trim:
            spans = [(0, n)]
        else:
            spans = moving_spans(
                gmr["root_pos"],
                gmr["dof_pos"],
                fps,
                speed_thresh=args_cli.speed_thresh,
                smooth_s=args_cli.smooth_s,
                min_moving_s=args_cli.min_moving_s,
                pad_s=args_cli.pad_s,
            )
        print(
            f"[TRIM] {clip_name}: {n} frames ({n / fps:.2f}s) -> "
            + ", ".join(f"[{s}:{e}] {(e - s) / fps:.2f}s" for s, e in spans)
        )
        for i, (start, end) in enumerate(spans):
            motion = remap_gmr_to_lab(
                gmr,
                gmr_dof_names=gmr_dof_names,
                lab_dof_names=lab_dof_names,
                loop_mode=loop_mode,
                start_frame=start,
                end_frame=end,
            )
            name = clip_name if len(spans) == 1 else f"{clip_name}_{i:02d}"
            motion_data_dicts.append(motion)
            output_names.append(f"{name}.pkl")
            fps_values.append(motion["fps"])

    if not all(f == fps_values[0] for f in fps_values):
        warnings.warn("Motions have different fps. Using fps from first motion.")
    fps = float(fps_values[0])
    dt = 1.0 / fps

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motion_data_dicts),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    sim.reset()
    print("Simulation starting ...")
    motion_data_dicts = run_simulator(simulation_app, sim, scene, motion_data_dicts, lab_key_body_names)

    print("Saving converted motions...")
    for name, motion in zip(output_names, motion_data_dicts):
        out_path = Path(args_cli.output_dir) / name
        with open(out_path, "wb") as f:
            pickle.dump(motion, f)
        print(f"Saved: {out_path}  frames={motion['dof_pos'].shape[0]}  fps={motion['fps']}")
    print("Done.")
    os._exit(0)


if __name__ == "__main__":
    main()
