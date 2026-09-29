# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Isaac Lab motion pkls and record a video the same way Isaac Lab does.

Isaac Lab does not use TiledCamera for training/play videos. It:
  1. launches with --headless --enable_cameras
  2. points a CPU rgb annotator at the viewport camera /OmniverseKit_Persp
  3. calls sim.render() then annotator.get_data()
  4. wraps frames with gymnasium RecordVideo (we write mp4 directly)

See IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py::render
and IsaacLab/docs/source/how-to/record_video.rst.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Isaac Lab motion pkls and record a video.")
parser.add_argument("--robot", type=str, default="rp1_24dof", choices=["rpo", "rp1_24dof"])
parser.add_argument("--input_dir", type=str, required=True, help="Directory of converted .pkl motions.")
parser.add_argument("--output_file", type=str, default=None, help="Output mp4 path.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.output_file is None:
    args_cli.output_file = str(Path(args_cli.input_dir) / "isaaclab_preview.mp4")
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene

sys.path.insert(0, str(Path(__file__).parent))

if args_cli.robot == "rpo":
    from robolab.assets.robots.roboparty import RPO_CFG as ROBOT_CFG
elif args_cli.robot == "rp1_24dof":
    from robolab.assets.robots.roboparty import RP1_24DOF_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

from gmr_to_lab import ReplayMotionsSceneCfg

# Official ViewerCfg.cam_prim_path
CAM_PATH = "/OmniverseKit_Persp"


def load_lab_pkls(input_dir: str) -> list[tuple[str, dict]]:
    files = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() == ".pkl")
    if not files:
        raise ValueError(f"No .pkl files in {input_dir}")
    motions = []
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        motions.append((path.stem, data))
    return motions


def grab_rgb(annotator, width: int, height: int) -> np.ndarray | None:
    """Match ManagerBasedRLEnv.render(): empty warmup frames are discarded, not written."""
    rgb_data = annotator.get_data()
    if rgb_data is None:
        return None
    rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
    if rgb_data.size == 0:
        return None
    rgb = rgb_data[:, :, :3]
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb)


def main():
    motions = load_lab_pkls(args_cli.input_dir)
    fps = float(motions[0][1]["fps"])
    dt = 1.0 / fps
    resolution = (args_cli.width, args_cli.height)
    print(f"[INFO] Loaded {len(motions)} motions, fps={fps}", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        sky_light=AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.9, 1.0)),
        ),
    )
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]

    distant = sim_utils.DistantLightCfg(intensity=4000.0, color=(1.0, 1.0, 1.0), angle=0.8)
    distant.func("/World/VideoLight", distant)

    print(f"[INFO] render_mode={sim.render_mode}", flush=True)
    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    # Viewport camera pose, same helper ViewportCameraController uses.
    sim.set_camera_view(eye=(2.5, 2.5, 1.8), target=(0.0, 0.0, 0.5), camera_prim_path=CAM_PATH)

    persp = omni.usd.get_context().get_stage().GetPrimAtPath(CAM_PATH)
    print(f"[INFO] {CAM_PATH} valid={persp.IsValid()} type={persp.GetTypeName() if persp.IsValid() else None}", flush=True)

    # Official env.render() attaches only after the sim is playing. Under GPU
    # contention the first attach can fail; retry with a fresh render product.
    annotator = None
    last_err = None
    for attempt in range(12):
        for _ in range(8):
            sim.render()
            scene.update(dt)
        render_product = rep.create.render_product(CAM_PATH, resolution)
        for _ in range(4):
            sim.render()
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        try:
            annotator.attach([render_product])
            print(f"[INFO] rgb annotator attached after {attempt + 1} attempts", flush=True)
            break
        except TypeError as err:
            last_err = err
            print(f"[WARN] annotator attach failed ({attempt + 1}/12): {err}", flush=True)
            try:
                annotator.detach()
            except Exception:
                pass
            annotator = None
    if annotator is None:
        raise RuntimeError(f"Failed to attach rgb annotator: {last_err}") from last_err

    rgb0 = None
    for i in range(60):
        sim.render()
        rgb0 = grab_rgb(annotator, args_cli.width, args_cli.height)
        mean = None if rgb0 is None else float(rgb0.mean())
        print(f"[INFO] warmup frame {i + 1}: {'empty' if rgb0 is None else f'rgb mean={mean:.2f}'}", flush=True)
        if rgb0 is not None and mean > 8.0:
            break
    else:
        raise RuntimeError(
            f"Camera frames stayed black (last mean={0.0 if rgb0 is None else rgb0.mean():.2f})."
        )

    debug_png = Path(args_cli.output_file).with_name("isaaclab_preview_frame0.png")
    debug_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_png), cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Wrote debug frame -> {debug_png}", flush=True)

    out_path = Path(args_cli.output_file)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args_cli.width, args_cli.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    origin = scene.env_origins[0]
    hold = max(1, int(round(fps * 0.4)))
    total_frames = 0
    skipped_black = 0

    for name, motion in motions:
        root_pos = torch.as_tensor(motion["root_pos"], device=scene.device, dtype=torch.float32)
        root_rot = torch.as_tensor(motion["root_rot"], device=scene.device, dtype=torch.float32)
        dof_pos = torch.as_tensor(motion["dof_pos"], device=scene.device, dtype=torch.float32)
        num_frames = int(dof_pos.shape[0])
        print(f"[INFO] Replaying {name}: {num_frames} frames", flush=True)

        frame_ids = list(range(num_frames)) + [num_frames - 1] * hold
        for frame_idx in frame_ids:
            root_states = robot.data.default_root_state.clone()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone() * 0.0

            pos = root_pos[frame_idx] + origin
            root_states[0, :3] = pos
            root_states[0, 3:7] = root_rot[frame_idx]
            root_states[0, 7:13] = 0.0
            joint_pos[0] = dof_pos[frame_idx]
            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            eye = pos.detach().cpu().numpy() + np.array([2.4, 2.0, 1.5])
            target = pos.detach().cpu().numpy() + np.array([0.0, 0.0, 0.45])
            sim.set_camera_view(eye=tuple(eye.tolist()), target=tuple(target.tolist()), camera_prim_path=CAM_PATH)

            sim.render()
            scene.update(dt)
            rgb = grab_rgb(annotator, args_cli.width, args_cli.height)
            if rgb is None or rgb.mean() < 3.0:
                skipped_black += 1
                continue

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"{name}  {frame_idx + 1}/{num_frames}",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(bgr)
            total_frames += 1

            if not simulation_app.is_running():
                break
        if not simulation_app.is_running():
            break

    writer.release()
    print(f"[INFO] Wrote {total_frames} frames (skipped {skipped_black} black) -> {out_path}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
