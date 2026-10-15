"""Measure depth-only action changes at matched states and render saved rollouts."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sim2sim_rp1_pufo import Policy, Simulation, make_model, place_on_stairs, read_config
from diagnose_pufo_depth_stability import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, help="Also render baseline / true-velocity saved trajectories")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest_path = ROOT / "robo_newton/data/motions/rp1_lafan/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    item = next(x for x in manifest["clips"] if x["name"] == "dance1_subject2")
    with np.load(manifest_path.parent / item["npz"]) as data:
        motion = {k: data[k].copy() for k in data.files}
    cfg = read_config(args.checkpoint.parent / "params/env.yaml")
    policy = Policy(args.checkpoint)
    flat = Simulation(make_model(False), manifest["npz"]["joint_order"], cfg)
    stair = Simulation(make_model(True), manifest["npz"]["joint_order"], cfg)
    contexts, headings, velocities = policy.contexts(motion, flat.default)
    actor, reports = policy.actor, []
    try:
        with torch.inference_mode():
            for frame in (0, 45, 50, 100, 200):
                for sim in (flat, stair):
                    sim.set_reference(motion, frame)
                    sim.depth_history.clear()
                place_on_stairs(stair)
                obs, _ = flat.observe(headings[frame], velocities[frame])
                norm = normalize(policy, obs)
                latent = actor.depth_encoder(obs["depth_image"])
                z = contexts[frame:frame+1]
                base_action, base_estimate = actor._mean_from_depth_latent(norm, z, latent, latent)
                images = {"flat": obs["depth_image"].clone(),
                          "stairs": stair.observe(headings[frame], velocities[frame])[0]["depth_image"],
                          "zero": torch.zeros_like(obs["depth_image"])}
                original_rotation = flat.camera_rot.copy()
                for angle in (-2, 2):
                    flat.camera_rot = original_rotation @ Rotation.from_euler("y", angle, degrees=True).as_matrix()
                    flat.depth_history.clear()
                    images[f"camera_pitch_{angle}"] = flat.observe(headings[frame], velocities[frame])[0]["depth_image"]
                flat.camera_rot = original_rotation
                results = {}
                for name, depth in images.items():
                    changed = actor.depth_encoder(depth)
                    branches = {}
                    for branch, direct, estimator in (("direct", changed, latent),
                                                      ("estimator", latent, changed),
                                                      ("both", changed, changed)):
                        action, estimate = actor._mean_from_depth_latent(norm, z, direct, estimator)
                        delta = (action-base_action)[0].numpy() * flat.action_scale * policy.action_rescale
                        branches[branch] = dict(target_delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
                                                max_target_delta_rad=float(abs(delta).max()),
                                                velocity_estimate_delta_m_s=(estimate-base_estimate)[0].tolist())
                    results[name] = dict(pixel_mae_uint8=float((depth.float()-obs["depth_image"].float()).abs().mean()),
                                         branches=branches)
                reports.append(dict(frame=frame, inputs=results))
                np.savez_compressed(args.output / f"images_frame{frame}.npz",
                                    **{k: v.numpy() for k, v in images.items()})
        (args.output / "report.json").write_text(json.dumps(dict(
            checkpoint=str(args.checkpoint), scope="Same proprioceptive state and context; only depth changes",
            frames=reports), indent=2) + "\n")
        if args.rollouts:
            with np.load(args.rollouts) as data:
                modes = data["modes"].tolist()
                root, quat, joints = data["root_pos"], data["root_quat"], data["joint_pos"]
            columns = [("BASELINE | ESTIMATED VELOCITY", modes.index("baseline")),
                       ("CONTROL | TRUE VELOCITY", modes.index("true_velocity"))]
            video = args.output / "default50_baseline_vs_true_velocity.mp4"
            encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "1280x690", "-r", "25", "-i", "-", "-an",
                "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
            try:
                for step in range(0, len(root), 2):
                    canvas = np.full((690, 1280, 3), 24, dtype=np.uint8)
                    for column, (label, index) in enumerate(columns):
                        flat.data.qpos[:3] = root[step, index]
                        flat.data.qpos[3:7] = quat[step, index, [3, 0, 1, 2]]
                        flat.data.qpos[flat.qids] = joints[step, index]
                        mujoco.mj_forward(flat.model, flat.data)
                        x = column*640
                        canvas[40:680, x:x+640] = flat.render()
                        cv2.putText(canvas, f"{label}  {step/50:.2f}s", (x+12, 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (240,240,240), 1, cv2.LINE_AA)
                    encoder.stdin.write(canvas.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Encoder failed")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i",
                            str(video), "-f", "null", "-"], check=True)
    finally:
        flat.close()
        stair.close()


if __name__ == "__main__":
    main()
