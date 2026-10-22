#!/usr/bin/env python3
"""Run an exported UFO actor in MuJoCo and record a tracking video.

This is the non-visual counterpart of ``sim2sim_rp1_pufo.py``.  It consumes
``export_ufo_deploy.py`` output, so no IsaacLab, simulator camera, or GPU is
needed for deployment validation.  The video contains only the kinematic
motion reference and a flat MuJoCo policy rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[3]

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from rsl_rl.runners.fbcpr_runner import make_ufo_history
from robolab.utils.mujoco_actuators import RoboPartyMujocoActuator
from robolab.utils.mujoco_physics import NewtonPhysicsAsset
from robolab.utils.motion_npz import FULL_BODY_FIELDS, load_motion, motion_clip_paths
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
    heading_observation,
    heading_xy,
    velocity_observation,
    yaw_to_world,
)

# The terrain/MJCF construction is shared with the PUFO recorder.  It contains
# no policy or depth-specific logic and keeps native Newton assets identical.
from sim2sim_rp1_pufo import (  # noqa: E402
    DATA_ROOT,
    MJCF,
    NATIVE_PHYSICS,
    TERRAIN_XML,
    base_ang_vel_obs_scale,
    default_motion_directory,
    joint_values,
    make_model,
)


def read_motion(path: Path, joint_names: list[str], body_names: list[str]) -> dict:
    data = load_motion(path, joint_names=joint_names, body_names=body_names)
    return {key: data[key] for key in ("fps", "joint_pos", "joint_vel", *FULL_BODY_FIELDS)}


def weighted_context_window(contexts: torch.Tensor, frame: int, gamma: float, window_size: int) -> torch.Tensor:
    if contexts.ndim != 2 or not 0 <= frame < len(contexts):
        raise ValueError(f"Invalid context frame {frame} for shape {tuple(contexts.shape)}")
    count = min(window_size, len(contexts) - frame)
    weights = torch.pow(
        torch.as_tensor(gamma, dtype=contexts.dtype),
        torch.arange(count, dtype=contexts.dtype),
    )
    value = (contexts[frame : frame + count] * weights[:, None]).sum(dim=0) / weights.sum()
    first_norm = torch.linalg.vector_norm(contexts[0])
    value_norm = torch.linalg.vector_norm(value)
    return value * (first_norm / value_norm) if first_norm > 0 and value_norm > 0 else value


class Policy:
    def __init__(self, export_dir: Path):
        self.export_dir = export_dir
        self.meta = json.loads((export_dir / "export_meta.json").read_text())
        self.session = ort.InferenceSession(
            str(export_dir / self.meta["actor_file"]),
            providers=["CPUExecutionProvider"],
        )
        expected = [*self.meta["actor_inputs"].keys()]
        actual = [item.name for item in self.session.get_inputs()]
        if actual != expected or [item.name for item in self.session.get_outputs()] != ["actions"]:
            raise ValueError(f"UFO actor ONNX contract mismatch: expected {expected}, got {actual}")
        self.action_rescale = float(self.meta["action_rescale"])
        self.action_scale = float(self.meta["action_scale"])
        self.step = int(self.meta.get("global_step", 0))

    def act(self, observations: dict[str, np.ndarray], z: torch.Tensor) -> np.ndarray:
        feeds = {key: np.asarray(observations[key], dtype=np.float32) for key in self.meta["actor_inputs"] if key != "z"}
        feeds["z"] = z.detach().cpu().numpy().astype(np.float32, copy=False)
        action = self.session.run(["actions"], feeds)[0]
        if not np.isfinite(action).all():
            raise FloatingPointError("UFO ONNX actor returned non-finite actions")
        return action


class Simulation:
    def __init__(self, model, names: list[str], meta: dict, physics_asset: NewtonPhysicsAsset):
        self.model, self.data = model, mujoco.MjData(model)
        self.names = list(names)
        self.jids = np.asarray([model.joint(name).id for name in names])
        self.qids = model.jnt_qposadr[self.jids]
        self.vids = model.jnt_dofadr[self.jids]
        self.joint_limits = model.jnt_range[self.jids].copy()
        self.joint_limits[~model.jnt_limited[self.jids].astype(bool)] = [-np.inf, np.inf]
        self.default = np.asarray(meta["default_joint_pos"], dtype=np.float64)
        if self.default.shape != (len(names),):
            raise ValueError("Export metadata default_joint_pos does not match joint_order")
        actuator_cfg = meta["actuator_cfg"]
        self.motor = RoboPartyMujocoActuator(MJCF.parent.parent / "urdf/rp1_24dof.urdf", names, actuator_cfg)
        self.aids = self.motor.bind(model)
        self.action_scale = float(meta["action_scale"])
        self.base_ang_vel_scale = float(meta["base_ang_vel_scale"])
        self.base = model.body("base_link").id
        self.history = make_ufo_history(1, "cpu")
        self.last_action = np.zeros(len(names), dtype=np.float32)
        self.renderer = None
        self.camera_distance, self.camera_azimuth, self.camera_elevation = 3.4, 55.0, -25.0
        if not physics_asset.is_native_model(model):
            raise ValueError("UFO sim2sim requires the native Newton physics asset")

    def set_reference(self, motion: dict, frame: int) -> None:
        self.data.qpos[:3] = motion["body_pos_w"][frame, 0]
        quat = motion["body_quat_w"][frame, 0]
        self.data.qpos[3:7] = quat[[3, 0, 1, 2]]
        self.data.qpos[self.qids] = motion["joint_pos"][frame]
        self.data.qvel[:3] = motion["body_lin_vel_w"][frame, 0]
        self.data.qvel[3:6] = Rotation.from_quat(quat).as_matrix().T @ motion["body_ang_vel_w"][frame, 0]
        self.data.qvel[self.vids] = motion["joint_vel"][frame]
        mujoco.mj_forward(self.model, self.data)

    def observe(self, target_heading: np.ndarray, target_velocity: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
        rot = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        tensor = lambda value: torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)[None]
        values = {
            "joint_pos": tensor(self.data.qpos[self.qids] - self.default),
            "joint_vel": tensor(self.data.qvel[self.vids]),
            "projected_gravity": tensor(-rot[2]),
            "base_ang_vel": tensor(rot.T @ velocity[:3] * self.base_ang_vel_scale),
            "actions": tensor(self.last_action),
        }
        current_heading = F.normalize(tensor(rot[:2, 0]), dim=-1)
        target_heading_t = tensor(target_heading)
        target_velocity_t = tensor(target_velocity)
        obs_t = {
            "state": torch.cat([values[key] for key in ("joint_pos", "joint_vel", "projected_gravity", "base_ang_vel")], dim=-1),
            "last_action": values["actions"],
            "history_actor": self.history.flat(),
            "heading": heading_observation(current_heading, target_heading_t, torch.ones(1, 1, dtype=torch.bool)),
            "velocity_command": velocity_observation(target_velocity_t, current_heading, torch.ones(1, 1, dtype=torch.bool)),
        }
        return {key: value.numpy() for key, value in obs_t.items()}, values

    def advance(self, action: np.ndarray, action_rescale: float) -> None:
        self.last_action = np.clip(np.asarray(action, dtype=np.float32) * action_rescale, -action_rescale, action_rescale)
        target = self.default + self.action_scale * self.last_action
        target = np.clip(target, *self.joint_limits.T)
        for _ in range(4):
            self.data.ctrl[self.aids] = self.motor.compute(target, self.data.qpos[self.qids], self.data.qvel[self.vids])
            mujoco.mj_step(self.model, self.data, nstep=2)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced non-finite state")

    def render(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=640, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.lookat[2] = max(0.55, camera.lookat[2])
        camera.distance = self.camera_distance
        camera.azimuth, camera.elevation = self.camera_azimuth, self.camera_elevation
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def compose(simulations: list[Simulation], seconds: float, total: float, errors: np.ndarray, motion_name: str, model_step: int) -> np.ndarray:
    frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
    labels = ("MOTION REFERENCE", "FLAT | UFO POLICY")
    for index, (sim, label) in enumerate(zip(simulations, labels)):
        x = index * 640
        frame[50:690, x : x + 640] = sim.render()
        cv2.putText(frame, label, (x + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        info = f"{motion_name}  {seconds:06.2f}/{total:.2f}s" if index == 0 else f"Joint L2: {errors[index - 1]:.3f} rad | model {model_step:,}"
        cv2.putText(frame, info, (x + 16, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--motion-dir", "--manifest", dest="motion_dir", type=Path, default=default_motion_directory())
    parser.add_argument("--motion", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--physics-profile", choices=("native", "legacy"), default="native")
    parser.add_argument("--physics-asset", type=Path, default=NATIVE_PHYSICS.path)
    parser.add_argument("--window-size", type=int, choices=(3, 5), default=5)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--fps", type=int, choices=(25, 50), default=25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional short smoke-test limit.")
    args = parser.parse_args()
    if args.window_size <= 0 or not math.isfinite(args.gamma) or args.gamma <= 0:
        parser.error("Invalid context smoothing parameters")
    export_dir = args.export_dir.resolve()
    policy = Policy(export_dir)
    meta = policy.meta
    motion_name = args.motion or meta["motion"]
    if motion_name != meta["motion"]:
        raise ValueError(f"Exported context is for {meta['motion']!r}, not {motion_name!r}")
    context = np.load(export_dir / meta["context_file"], allow_pickle=False)
    contexts = torch.from_numpy(np.asarray(context["z"], dtype=np.float32))
    start_frame = int(np.asarray(context["start_frame"]).item())
    heading_relative = np.asarray(context["heading_relative_xy"], dtype=np.float32)
    velocity_relative = np.asarray(context["reference_velocity_heading0_xy"], dtype=np.float32)
    if len(contexts) != len(heading_relative) or len(contexts) != len(velocity_relative):
        raise ValueError("Context sidecar fields have inconsistent frame counts")

    source = motion_clip_paths(args.motion_dir, [motion_name])[0]
    body_order = list(meta["body_order"])
    joint_order = list(meta["joint_order"])
    full_motion = read_motion(source, joint_order, body_order)
    if float(full_motion["fps"].item()) != 50:
        raise ValueError("Expected 50 Hz motion with XYZW quaternions")
    first = start_frame + 1
    last = min(len(full_motion["joint_pos"]), first + len(contexts))
    motion_indices = np.arange(first, last, dtype=np.int64)
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if args.max_frames is not None:
        motion_indices = motion_indices[: args.max_frames]
    if len(motion_indices) < 2:
        raise ValueError("Exported context does not contain enough source frames")
    motion = {key: value[motion_indices].copy() if key != "fps" else value for key, value in full_motion.items()}
    motion["body_pos_w"][..., :2] -= motion["body_pos_w"][0, 0, :2].copy()
    motion_frames = len(motion["joint_pos"])
    contexts = contexts[:motion_frames]
    heading_relative = heading_relative[:motion_frames]
    velocity_relative = velocity_relative[:motion_frames]
    anchor = torch.from_numpy(full_motion["body_quat_w"][start_frame, 0][None]).float()
    anchor_heading = heading_xy(anchor)
    target_heading = yaw_to_world(torch.from_numpy(heading_relative), anchor_heading.expand(len(heading_relative), -1)).numpy()
    target_velocity = yaw_to_world(torch.from_numpy(velocity_relative), anchor_heading.expand(len(velocity_relative), -1)).numpy()

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    physics_asset = NewtonPhysicsAsset(args.physics_asset.resolve(), data_root=DATA_ROOT)
    sims = [
        Simulation(make_model(False, args.physics_profile, physics_asset=physics_asset), joint_order, meta, physics_asset),
        Simulation(make_model(False, args.physics_profile, physics_asset=physics_asset), joint_order, meta, physics_asset),
    ]
    output = args.output or export_dir / "videos" / f"{motion_name}_{meta.get('global_step', 0)}_sim2sim_ufo.mp4"
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = None
    temporary_video = None
    try:
        for sim in sims:
            sim.set_reference(motion, 0)
        body_ids = [sims[0].model.body(name).id for name in body_order]
        fk_error = float(np.max(np.linalg.norm(sims[0].data.xpos[body_ids] - motion["body_pos_w"][0], axis=-1)))
        if fk_error > 0.025:
            raise ValueError(f"Motion/MJCF kinematic mismatch: {fk_error:.4f} m")
        n = motion_frames - 1
        policy_count = len(sims) - 1
        trajectory = np.empty((n + 1, policy_count, len(joint_order)), dtype=np.float32)
        root_positions = np.empty((n + 1, policy_count, 3), dtype=np.float32)
        actions = np.empty((n, policy_count, len(joint_order)), dtype=np.float32)
        with tempfile.NamedTemporaryFile(prefix=".ufo-export-", suffix=".mp4", dir=output.parent, delete=False) as stream:
            temporary_video = Path(stream.name)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_video)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        started = time.monotonic()
        for step in range(n + 1):
            sims[0].set_reference(motion, step)
            for index, sim in enumerate(sims[1:]):
                trajectory[step, index] = sim.data.qpos[sim.qids]
                root_positions[step, index] = sim.data.qpos[:3]
            errors = np.linalg.norm(trajectory[step] - motion["joint_pos"][step], axis=-1)
            if step % (50 // args.fps) == 0:
                process.stdin.write(compose(sims, step / 50, n / 50, errors, motion_name, policy.step).tobytes())
            if step == n:
                break
            pairs = [sim.observe(target_heading[step], target_velocity[step]) for sim in sims[1:]]
            obs = {key: np.concatenate([pair[0][key] for pair in pairs], axis=0) for key in pairs[0][0]}
            z = weighted_context_window(contexts, step, args.gamma, args.window_size).repeat(policy_count, 1)
            actions[step] = policy.act(obs, z)
            for index, (sim, (_, values)) in enumerate(zip(sims[1:], pairs)):
                sim.advance(actions[step, index], policy.action_rescale)
                sim.history.add(values)
            if step % 250 == 0:
                print(f"t={step / 50:.1f}s/{n / 50:.1f}s joint_l2={errors.round(3).tolist()} elapsed={time.monotonic() - started:.1f}s", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoder failed")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i", str(temporary_video), "-f", "null", "-"], check=True)
        np.savez_compressed(output.with_suffix(".npz"), joint_pos=trajectory, root_pos=root_positions, actions=actions, reference_joint_pos=motion["joint_pos"], fps=50)
        report = {
            "format": "ufo_sim2sim_report_v1",
            "export_dir": str(export_dir),
            "motion": motion_name,
            "duration_s": n / 50,
            "video_fps": args.fps,
            "joint_order": joint_order,
            "physics_asset": str(physics_asset.path),
            "initial_fk_error_m": fk_error,
            "context_window_size": args.window_size,
            "context_gamma": args.gamma,
            "inference_device": "cpu_onnxruntime",
            "elapsed_seconds": time.monotonic() - started,
        }
        for index, name in enumerate(("flat",)):
            error = np.linalg.norm(trajectory[:, index] - motion["joint_pos"], axis=-1)
            report[name] = {
                "mean_joint_l2_rad": float(error.mean()),
                "max_joint_l2_rad": float(error.max()),
                "final_joint_l2_rad": float(error[-1]),
                "root_displacement_xy_m": (root_positions[-1, index] - root_positions[0, index]).tolist(),
            }
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        temporary_video.replace(output)
        print(json.dumps(report, indent=2), flush=True)
        print(f"VIDEO={output}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.stdin.close()
            process.wait(timeout=30)
        for sim in sims:
            sim.close()
        if temporary_video is not None and temporary_video.exists():
            temporary_video.unlink()


if __name__ == "__main__":
    main()
