#!/usr/bin/env python3
"""Kinematic MuJoCo replay of Isaac Lab motion pkls. Writes one mp4 per clip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio_ffmpeg
import joblib
import mujoco
import numpy as np
import yaml

from robolab.assets import ISAAC_DATA_DIR

def load_lab_pkls(input_dir: Path, names: list[str] | None = None) -> list[tuple[str, dict]]:
    if names:
        files = [input_dir / f"{name}.pkl" for name in names]
    else:
        files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise ValueError(f"No .pkl files in {input_dir}")
    return [(path.stem, joblib.load(path)) for path in files]


def lab_dof_to_mj(dof_lab: np.ndarray, lab_names: list[str], mj_names: list[str]) -> np.ndarray:
    idx = [lab_names.index(name) for name in mj_names]
    return dof_lab[:, idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay lab pkls to mp4.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--motions", nargs="+", help="Clip names without the .pkl suffix")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--config_file", type=Path, default=Path(__file__).parent / "config/rp1_24dof.yaml")
    parser.add_argument("--xml", type=str, default=f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/mjcf/rp1_flat.xml")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    lab_names = list(config["lab_dof_names"])
    gmr_names = list(config["gmr_dof_names"])

    input_dir = Path(args.input_dir)
    motions = load_lab_pkls(input_dir, args.motions)
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mj_joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name != "base_link_freejoint"]
    if mj_joint_names != gmr_names:
        raise ValueError(f"MJCF joint order != gmr_dof_names:\n{mj_joint_names}\n{gmr_names}")

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    # Use the same full-body framing as sim2sim_rp1_amp.py.
    cam.distance = 2.15
    cam.azimuth = 62.0
    cam.elevation = -12.0
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0

    for name, motion in motions:
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
        dof_mj = lab_dof_to_mj(np.asarray(motion["dof_pos"], dtype=np.float32), lab_names, gmr_names)
        fps = float(motion["fps"])
        n = int(dof_mj.shape[0])
        out_path = output_dir / f"{name}.mp4"
        if out_path.exists():
            raise FileExistsError(f"Recording already exists: {out_path}")
        writer = imageio_ffmpeg.write_frames(
            str(out_path), (args.width, args.height), fps=fps, codec="libx264",
            quality=None, macro_block_size=1,
            output_params=["-crf", "18", "-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"],
        )
        writer.send(None)
        print(f"[INFO] {name}: {n} frames -> {out_path}")
        try:
            for i in range(n):
                data.qpos[:3] = root_pos[i]
                data.qpos[3:7] = root_rot[i]
                data.qpos[7:] = dof_mj[i]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = (root_pos[i, 0] + 0.15, root_pos[i, 1], max(0.58, root_pos[i, 2]))
                renderer.update_scene(data, camera=cam, scene_option=option)
                rgb = renderer.render()
                cv2.rectangle(rgb, (20, 15), (850, 110), (18, 23, 29), -1)
                cv2.putText(rgb, f"REFERENCE (kinematic) | {name}", (35, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(rgb, f"{i / fps:.2f} / {(n - 1) / fps:.2f} s", (35, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
                writer.send(rgb)
        finally:
            writer.close()
        print(f"[INFO] Wrote {out_path}")
    renderer.close()


if __name__ == "__main__":
    main()
