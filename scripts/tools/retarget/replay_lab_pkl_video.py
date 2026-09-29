#!/usr/bin/env python3
"""Kinematic MuJoCo replay of Isaac Lab motion pkls. Writes one mp4 per clip."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import yaml

from robolab.assets import ISAAC_DATA_DIR

_FFMPEG_BIN = (
    "/data/conda/envs/zym_isaaclab/lib/python3.11/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def encode_h264(src: Path) -> None:
    ffmpeg = _FFMPEG_BIN if os.path.isfile(_FFMPEG_BIN) else "ffmpeg"
    tmp = src.with_suffix(".h264.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            "-movflags", "+faststart", str(tmp),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.replace(src)


def load_lab_pkls(input_dir: Path) -> list[tuple[str, dict]]:
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".pkl")
    if not files:
        raise ValueError(f"No .pkl files in {input_dir}")
    motions = []
    for path in files:
        with open(path, "rb") as f:
            motions.append((path.stem, pickle.load(f)))
    return motions


def lab_dof_to_mj(dof_lab: np.ndarray, lab_names: list[str], mj_names: list[str]) -> np.ndarray:
    idx = [lab_names.index(name) for name in mj_names]
    return dof_lab[:, idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay lab pkls to mp4.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="robolab/scripts/tools/retarget/config/rp1_24dof.yaml")
    parser.add_argument("--xml", type=str, default=f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/mjcf/rp1_flat.xml")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    lab_names = list(config["lab_dof_names"])
    gmr_names = list(config["gmr_dof_names"])

    input_dir = Path(args.input_dir)
    motions = load_lab_pkls(input_dir)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mj_joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name != "base_link_freejoint"]
    if mj_joint_names != gmr_names:
        raise ValueError(f"MJCF joint order != gmr_dof_names:\n{mj_joint_names}\n{gmr_names}")

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.distance = 4.0
    cam.azimuth = 135.0
    cam.elevation = -20.0

    for name, motion in motions:
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
        dof_mj = lab_dof_to_mj(np.asarray(motion["dof_pos"], dtype=np.float32), lab_names, gmr_names)
        fps = float(motion["fps"])
        n = int(dof_mj.shape[0])
        out_path = input_dir / f"{name}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
        print(f"[INFO] {name}: {n} frames -> {out_path}")
        for i in range(n):
            data.qpos[:3] = root_pos[i]
            data.qpos[3:7] = root_rot[i]
            data.qpos[7:] = dof_mj[i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = (root_pos[i, 0], root_pos[i, 1], root_pos[i, 2] + 0.4)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"{name}  {i + 1}/{n}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(bgr)
        writer.release()
        encode_h264(out_path)
        print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
