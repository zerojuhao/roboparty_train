#!/usr/bin/env python3
"""Record 16 LAFAN motions as a synchronized 4x4 MuJoCo grid.

The clips are kinematic references from the BeyondMimic NPZ export.  Each
panel follows its own root position, while the camera keeps the close,
front-facing direction measured from that clip's first frame.  The original
MuJoCo sky and checkerboard floor remain visible in every panel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path("/data/zym/roboparty_train/ufo_data/motions/rp1_lafan")
DEFAULT_XML = ROOT / "robo_newton/data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"
DEFAULT_OUTPUT = Path("/data/zym/roboparty_train/videos/lafan_16_grid_fixed_front_ground_no_labels_30s.mp4")

# One clip per distinct LAFAN style/category.  All are present in rp1_lafan.
DEFAULT_MOTIONS = (
    "dance1_subject1", "dance2_subject1", "fallAndGetUp1_subject1", "fallAndGetUp2_subject2",
    "fight1_subject2", "fightAndSports1_subject1", "ground1_subject1", "ground2_subject2",
    "jumps1_subject1", "multipleActions1_subject1", "push1_subject2", "pushAndFall1_subject1",
    "run1_subject2", "run2_subject1", "sprint1_subject2", "walk1_subject1",
)


def load_motions(dataset: Path, names: tuple[str, ...]) -> list[tuple[str, dict[str, np.ndarray]]]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    records = {str(item["name"]): item for item in manifest.get("clips", [])}
    result = []
    for name in names:
        record = records.get(name, {})
        source = Path(record.get("npz", dataset / "npz" / f"{name}.npz"))
        if not source.is_absolute():
            source = dataset / source
        if not source.is_file():
            raise FileNotFoundError(f"Missing LAFAN motion {name}: {source}")
        with np.load(source, allow_pickle=False) as archive:
            required = ("joint_pos", "body_pos_w", "body_quat_w", "fps")
            missing = [key for key in required if key not in archive]
            if missing:
                raise ValueError(f"{source} is missing {missing}")
            motion = {key: archive[key].copy() for key in archive.files}
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if abs(fps - 50.0) > 1e-3:
            raise ValueError(f"{name} has fps={fps}, expected 50 Hz")
        result.append((name, motion))
    return result


def scene_model(xml_path: Path) -> mujoco.MjModel:
    """Load the checked-in MuJoCo model with its original environment."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    try:
        model.vis.quality.shadowsize = 2048
    except (AttributeError, ValueError):
        pass
    return model


def set_motion(model: mujoco.MjModel, data: mujoco.MjData, motion: dict[str, np.ndarray], frame: int,
               joint_ids: np.ndarray, qpos_ids: np.ndarray) -> np.ndarray:
    root = motion["body_pos_w"][frame, 0].astype(np.float64, copy=False)
    data.qpos[:3] = root
    # Dataset quaternions are XYZW; MuJoCo free joints use WXYZ.
    quat = motion["body_quat_w"][frame, 0]
    data.qpos[3:7] = quat[[3, 0, 1, 2]]
    data.qpos[qpos_ids] = motion["joint_pos"][frame]
    mujoco.mj_forward(model, data)
    return root


def forward_heading(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return the robot's local +X axis projected onto the world XY plane."""
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    heading = np.array((1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y + w * z)), dtype=np.float64)
    norm = np.linalg.norm(heading)
    return heading / norm if norm > 1.0e-8 else np.array((0.0, -1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=int, choices=(25, 30, 50), default=30)
    parser.add_argument("--tile-size", type=int, default=480, help="Pixels per square panel (4x4 output).")
    parser.add_argument("--camera-distance", type=float, default=2.35)
    args = parser.parse_args()
    if len(DEFAULT_MOTIONS) != 16:
        raise AssertionError("The grid must contain exactly 16 motions")
    if args.duration <= 0 or args.tile_size < 160 or args.camera_distance <= 0:
        parser.error("duration, tile distance and tile size must be positive")

    motions = load_motions(args.dataset, DEFAULT_MOTIONS)
    model = scene_model(args.xml)
    data = [mujoco.MjData(model) for _ in motions]
    joint_names = json.loads((args.dataset / "manifest.json").read_text())["npz"]["joint_order"]
    joint_ids = np.asarray([model.joint(name).id for name in joint_names], dtype=np.int32)
    qpos_ids = model.jnt_qposadr[joint_ids]
    renderer = mujoco.Renderer(model, height=args.tile_size, width=args.tile_size)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.sitegroup[:] = 0
    cameras = [mujoco.MjvCamera() for _ in motions]
    for camera, (_, motion) in zip(cameras, motions):
        camera.distance = args.camera_distance
        camera.elevation = -12.0
        initial_heading = forward_heading(motion["body_quat_w"][0, 0])
        # Fix the view in front of the initial pose.  During playback only the
        # look-at position moves, so turns in the motion do not rotate the view.
        camera.azimuth = float(np.degrees(np.arctan2(-initial_heading[1], -initial_heading[0])))

    total_frames = int(round(args.duration * args.fps))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.tile_size * 4}x{args.tile_size * 4}",
               "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
               "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for output_frame in range(total_frames):
            # Keep visible gutters distinct from the original MuJoCo scene.
            canvas = np.full((args.tile_size * 4, args.tile_size * 4, 3), (28, 32, 38), dtype=np.uint8)
            source_frame = int(round(output_frame * 50.0 / args.fps))
            for index, ((_, motion), mj_data, camera) in enumerate(zip(motions, data, cameras)):
                frame = source_frame % int(motion["joint_pos"].shape[0])
                root = set_motion(model, mj_data, motion, frame, joint_ids, qpos_ids)
                camera.lookat[:] = (root[0], root[1], max(0.72, root[2] + 0.05))
                renderer.update_scene(mj_data, camera=camera, scene_option=option)
                tile = renderer.render().copy()
                row, col = divmod(index, 4)
                y0, x0 = row * args.tile_size, col * args.tile_size
                canvas[y0:y0 + args.tile_size, x0:x0 + args.tile_size] = tile
            process.stdin.write(canvas.tobytes())
            if output_frame % max(1, args.fps * 5) == 0:
                print(f"[INFO] {output_frame / args.fps:.1f}/{args.duration:.1f}s", flush=True)
    finally:
        if process.stdin:
            process.stdin.close()
        code = process.wait()
        renderer.close()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    print(f"[INFO] Wrote {total_frames} frames -> {output}")


if __name__ == "__main__":
    main()
