"""Record reference / flat / RP1 simple-terrain PUFO tracking in a synchronized MuJoCo video.

Uses the RP1 MJCF and ray-camera approach of sim2sim_rp1_parkour.py, but
loads PUFO networks and observation/control contracts from the training run.
CPU inference is intentional: a recording must not evict the training GPUs.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "rsl_rl"), str(ROOT / "robo_newton")]
DATA_ROOT = ROOT / "robo_newton/robolab/data"

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import yaml

from robolab.utils.mujoco_physics import NewtonPhysicsAsset
from robolab.utils.mujoco_actuators import RoboPartyMujocoActuator
from robolab.utils.motion_npz import FULL_BODY_FIELDS, load_motion, motion_clip_paths
from robolab.utils.data_paths import motion_data_dir

from rsl_rl.algorithms.fb_networks import BackwardMap, ObsNormalizer, VisualResidualActor
from rsl_rl.runners.fbcpr_runner import make_ufo_history
from robolab.tasks.manager_based.ufo.mdp.motion_observations import compute_humanoid_observations_max
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
    heading_observation, heading_xy, velocity_observation, yaw_to_world,
)
from robolab.tasks.manager_based.ufo.motion.torch_utils import quat_rotate_inverse

LOG_ROOT = Path("/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo")
MJCF = DATA_ROOT / "robots/rp1/mjcf/rp1_flat.xml"
NATIVE_PHYSICS = NewtonPhysicsAsset(MJCF.with_name("robot_24dof_newton_physics.json"),
                                    data_root=DATA_ROOT)
TERRAIN_DIR = MJCF.parent / "rp1_simple_terrain"
TERRAIN_XML = TERRAIN_DIR / "rp1_simple_terrain.xml"


def terrain_metadata():
    return json.loads(TERRAIN_XML.with_name("meta.json").read_text())


def default_motion_directory():
    return Path(os.environ.get("UFO_MOTION_DIR", os.environ.get("UFO_MOTION_MANIFEST", motion_data_dir("rp1_lafan"))))


def read_motion(path, joint_names, body_names):
    data = load_motion(path, joint_names=joint_names, body_names=body_names)
    # Metadata describes the clip; it must not be sliced or passed to the policy.
    return {key: data[key] for key in ("fps", "joint_pos", "joint_vel", *FULL_BODY_FIELDS)}


class ConfigLoader(yaml.SafeLoader):
    pass

# Read serialized config data without instantiating arbitrary Python objects.
for _tag in ("python/tuple", "python/object/apply:builtins.slice"):
    ConfigLoader.add_constructor("tag:yaml.org,2002:" + _tag,
                                 lambda loader, node: loader.construct_sequence(node))

def read_config(path):
    return yaml.load(path.read_text(), Loader=ConfigLoader)

def latest_checkpoint(run_dir=None):
    runs = [Path(run_dir)] if run_dir else sorted(LOG_ROOT.iterdir(), reverse=True)
    for run in runs:
        files = [p for p in run.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
        if files:
            return max(files, key=lambda p: int(p.stem.split("_")[-1]))
    raise FileNotFoundError("No published model checkpoint found")

def joint_values(value, names, default=None):
    if isinstance(value, (int, float)):
        return np.full(len(names), value, dtype=np.float64)
    result = []
    for name in names:
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1 or (not matches and default is None):
            raise ValueError(f"Ambiguous or missing config for {name}: {matches}")
        result.append(matches[0] if matches else default)
    return np.asarray(result, dtype=np.float64)


def base_ang_vel_obs_scale(env_cfg, default: float = 1.0) -> float:
    """Read observations.policy.base_ang_vel.scale from env yaml/dict."""
    try:
        term = env_cfg["observations"]["policy"]["base_ang_vel"]
        scale = term.get("scale") if isinstance(term, dict) else None
    except Exception:
        scale = None
    if scale is None:
        return default
    if isinstance(scale, (tuple, list)):
        return float(scale[0])
    return float(scale)


def weighted_context_window(
    contexts: torch.Tensor,
    frame: int,
    gamma: float,
    window_size: int,
) -> torch.Tensor:
    """Match deployment's causal-in-playback, future-window latent smoothing.

    The deployment context sidecar stores one raw z per frame.  At playback
    frame ``t`` it combines z[t:t+window_size] with weights 1, gamma, ... and
    rescales the result to the norm of the first stored z, matching
    ``weighted_latent_window`` in the real-robot runtime.
    """
    if contexts.ndim != 2 or not 0 <= frame < len(contexts):
        raise ValueError(f"Invalid context frame {frame} for shape {tuple(contexts.shape)}")
    if window_size <= 0 or not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"Invalid context smoothing parameters: gamma={gamma}, window_size={window_size}")
    count = min(window_size, len(contexts) - frame)
    weights = torch.pow(
        torch.as_tensor(gamma, dtype=contexts.dtype, device=contexts.device),
        torch.arange(count, dtype=contexts.dtype, device=contexts.device),
    )
    smoothed = (contexts[frame:frame + count] * weights[:, None]).sum(dim=0) / weights.sum()
    first_norm = torch.linalg.vector_norm(contexts[0])
    smoothed_norm = torch.linalg.vector_norm(smoothed)
    if first_norm > 0.0 and smoothed_norm > 0.0:
        smoothed = smoothed * (first_norm / smoothed_norm)
    return smoothed


class Policy:
    def __init__(self, checkpoint):
        self.cfg = cfg = read_config(checkpoint.parent / "params/agent.yaml")
        env_cfg = read_config(checkpoint.parent / "params/env.yaml")
        self.base_ang_vel_scale = base_ang_vel_obs_scale(env_cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.step = int(state["env_iterations"] * state["num_envs"] * state["world_size"])
        self.action_rescale = float(state["action_rescale"])
        weights = state["model_state"]
        suffix = ".running_mean"
        dims = {key[len("normalizers."):-len(suffix)]: value.numel()
                for key, value in weights["obs_normalizer"].items() if key.endswith(suffix)}
        dims.update(heading=2, velocity_command=3, velocity_target_yaw=3)
        if (dims["state"], dims["last_action"], dims["history_actor"]) != (54, 24, 312):
            raise ValueError(f"Unsupported PUFO observation contract: {dims}")
        encoder = dict(cfg["depth_encoder"])
        shape = tuple(encoder.pop("input_shape"))
        if shape not in ((8, 18, 36), (8, 18, 32)):
            raise ValueError(f"Unsupported depth shape: {shape}")
        self.normalizer = ObsNormalizer(dims, identity_keys=("heading", "velocity_command", "velocity_target_yaw"))
        self.actor = VisualResidualActor(
            dims, cfg["actor_obs_keys"], cfg["z_dim"], 24, depth_shape=shape,
            encoder_cfg=encoder, estimator_cfg=cfg["velocity_estimator"],
            estimate_velocity=cfg["estimated_velocity_enabled"], **cfg["actor"],
        )
        self.backward = BackwardMap(sum(dims[k] for k in cfg["privileged_obs_keys"]), cfg["z_dim"],
                                    norm=cfg["norm_z"], **cfg["backward_map"])
        for module, key in ((self.actor, "actor"), (self.backward, "backward_map"),
                            (self.normalizer, "obs_normalizer")):
            module.load_state_dict(weights[key], strict=True)
            module.eval().requires_grad_(False)

    @torch.inference_mode()
    def contexts(self, motion, default):
        arrays = {k: torch.from_numpy(v).float() for k, v in motion.items() if k != "fps"}
        quat = arrays["body_quat_w"][:, 0]
        gravity = torch.tensor([0., 0., -1.]).expand(len(quat), -1)
        state = torch.cat((arrays["joint_pos"] - torch.tensor(default, dtype=torch.float32),
                           arrays["joint_vel"], quat_rotate_inverse(quat, gravity, w_last=True),
                           quat_rotate_inverse(quat, arrays["body_ang_vel_w"][:, 0], w_last=True)
                           * self.base_ang_vel_scale), -1)
        privileged = torch.cat(list(compute_humanoid_observations_max(
            arrays["body_pos_w"], arrays["body_quat_w"], arrays["body_lin_vel_w"],
            arrays["body_ang_vel_w"], True, True,
        ).values()), -1)
        norm = self.normalizer.normalize_dict({"state": state, "privileged_state": privileged})
        z = torch.cat([self.backward(batch) for batch in torch.cat(
            [norm[k] for k in self.cfg["privileged_obs_keys"]], -1).split(256)])
        if self.cfg["norm_z"]:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z, heading_xy(quat), arrays["body_lin_vel_w"][:, 0, :2]

    @torch.inference_mode()
    def exported_contexts(self, context_path, full_motion, motion_indices, default, start_frame):
        """Load the deployment sidecar and reconstruct its actor-facing targets.

        The sidecar stores z plus heading/velocity vectors in the yaw frame of
        ``start_frame``.  It does not store the actor's current-relative
        heading or velocity command, so those are reconstructed here exactly
        as they are in the training observation functions.
        """
        with np.load(context_path, allow_pickle=False) as archive:
            required = {
                "z", "heading_relative_xy", "heading_valid",
                "reference_velocity_heading0_xy", "velocity_valid",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Context sidecar is missing fields: {sorted(missing)}")
            z = torch.from_numpy(np.asarray(archive["z"], dtype=np.float32))
            heading_relative = torch.from_numpy(
                np.asarray(archive["heading_relative_xy"], dtype=np.float32)
            )
            heading_valid = torch.from_numpy(np.asarray(archive["heading_valid"])).bool()
            velocity_relative = torch.from_numpy(
                np.asarray(archive["reference_velocity_heading0_xy"], dtype=np.float32)
            )
            velocity_valid = torch.from_numpy(np.asarray(archive["velocity_valid"])).bool()
            sidecar_start = int(np.asarray(archive["start_frame"]).item()) if "start_frame" in archive.files else start_frame

        if z.ndim != 2 or z.shape[-1] != self.cfg["z_dim"]:
            raise ValueError(f"Expected context z [frames, {self.cfg['z_dim']}], got {tuple(z.shape)}")
        if not (heading_relative.shape == (len(z), 2)
                and velocity_relative.shape == (len(z), 2)
                and heading_valid.reshape(-1, 1).shape == (len(z), 1)
                and velocity_valid.reshape(-1, 1).shape == (len(z), 1)):
            raise ValueError("Context sidecar fields do not have matching frame dimensions")
        if sidecar_start < 0 or sidecar_start >= len(full_motion["body_quat_w"]):
            raise ValueError(f"Context start frame {sidecar_start} is outside the source motion")

        motion_indices = np.asarray(motion_indices, dtype=np.int64)
        context_indices = motion_indices - sidecar_start
        if int(context_indices.min()) < 0 or int(context_indices.max()) >= len(z):
            raise ValueError(
                f"Context covers source frames [{sidecar_start}, {sidecar_start + len(z) - 1}], "
                f"but requested frames [{motion_indices.min()}, {motion_indices.max()}]"
            )

        anchor_quat = torch.from_numpy(
            np.asarray(full_motion["body_quat_w"][sidecar_start, 0], dtype=np.float32)
        ).reshape(1, 4)
        anchor_heading = heading_xy(anchor_quat)
        target_heading_world = yaw_to_world(heading_relative, anchor_heading.expand_as(heading_relative))
        target_velocity_world = yaw_to_world(velocity_relative, anchor_heading.expand_as(velocity_relative))
        selected = torch.from_numpy(context_indices)
        return (
            z[selected],
            target_heading_world[selected],
            target_velocity_world[selected],
            heading_valid.reshape(-1, 1)[selected],
            velocity_valid.reshape(-1, 1)[selected],
        )

    @torch.inference_mode()
    def act(self, observations, context):
        normalized = self.normalizer.normalize_dict(observations)
        normalized["depth_image"] = observations["depth_image"]
        return self.actor(normalized, context, self.cfg["actor_std"]).mean

def make_model(stairs, physics_profile="native", *, physics_asset=None):
    physics_asset = physics_asset or NATIVE_PHYSICS
    if physics_profile not in {"native", "legacy"}:
        raise ValueError(f"Unknown PUFO physics profile: {physics_profile}")
    xml = ET.parse(MJCF).getroot()
    xml.remove(xml.find("include"))
    robot = ET.parse(MJCF.parent / "robot_24dof.xml").getroot()
    robot.find("compiler").set("meshdir", str(MJCF.parent.parent))
    if physics_profile == "native":
        physics_asset.align_robot_xml(robot)
    visual = xml.find("visual")
    visual.find("global").set("offwidth", "640")
    visual.find("global").set("offheight", "640")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ground = xml.find("worldbody/geom")
    ground.set("friction", "0.9 0.02 0.002")
    if stairs:
        if not TERRAIN_XML.is_file():
            raise FileNotFoundError(
                f"Missing RP1 simple terrain asset {TERRAIN_XML}; install {TERRAIN_XML.name} and meta.json "
                f"from the robot asset bundle into {TERRAIN_DIR}"
            )
        xml.find("worldbody").remove(ground)
    xml.extend(list(robot))
    if not stairs:
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    else:
        # Attach the full mjlab rp1_simple map (all terrain families), not a stairs island.
        robot_spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        terrain_spec = mujoco.MjSpec.from_file(str(TERRAIN_XML))
        robot_spec.worldbody.add_frame().attach_body(terrain_spec.body("terrain"), "terrain_", "")
        model = robot_spec.compile()
    model.opt.timestep = .0025
    model.opt.iterations = 600
    model.opt.ls_iterations = 80
    model.opt.tolerance = .01
    physics_asset.align_compiled_model(model)
    return model

class Simulation:
    def __init__(self, model, names, env_cfg, *, clip_joint_targets=True, physics_asset=None):
        physics_asset = physics_asset or NATIVE_PHYSICS
        self.model, self.data = model, mujoco.MjData(model)
        self.jids = np.array([model.joint(name).id for name in names])
        self.clip_joint_targets = clip_joint_targets
        self.joint_limits = model.jnt_range[self.jids].copy()
        self.joint_limits[~model.jnt_limited[self.jids].astype(bool)] = [-np.inf, np.inf]
        self.qids = model.jnt_qposadr[self.jids]
        self.vids = model.jnt_dofadr[self.jids]
        self.aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in self.jids])
        if not np.allclose(model.actuator_gear[self.aids, 0], 1):
            raise ValueError("Expected unit torque actuator gears")
        robot = env_cfg["scene"]["robot"]
        actuator = robot["actuators"]["all_joints"]
        self.default = joint_values(robot["init_state"]["joint_pos"], names, 0.)
        self.motor = RoboPartyMujocoActuator(MJCF.parent.parent / "urdf/rp1_24dof.urdf", names, actuator)
        self.limit = self.motor.effort_limit
        self.speed = self.motor.velocity_limit
        self.keep = self.motor.keep_velocity
        self.physics_profile = "native" if physics_asset.is_native_model(model) else "legacy"
        if self.physics_profile == "native":
            physics_asset.align_compiled_model(model)
        self.aids = self.motor.bind(model)
        self.action_scale = float(env_cfg["actions"]["joint_pos"]["scale"])
        self.base_ang_vel_scale = base_ang_vel_obs_scale(env_cfg)
        self.base = model.body("base_link").id
        self.waist = model.body("waist_yaw_link").id
        self.history = make_ufo_history(1, "cpu")
        self.last_action = np.zeros(24)
        self.last_depth = None
        camera = env_cfg["scene"]["camera"]
        self.depth_history_length = camera["data_histories"]["distance_to_image_plane_noised"]
        self.depth_history = deque(maxlen=self.depth_history_length)
        self.depth_crop = tuple(camera["noise_pipeline"]["crop_and_resize"]["crop_region"])
        self.aligned_depth = camera["noise_pipeline"]["sanitize_depth"].get("invalid_depth", 0.) == 2.5
        self.camera_pos = np.array(camera["offset"]["pos"])
        self.camera_rot = Rotation.from_quat(camera["offset"]["rot"]).as_matrix()
        self.depth_delay = 1  # Training Uniform(0,1) followed by floor gives one frame.
        u, v = np.meshgrid((np.arange(64) + .5) / 64 * 2 - 1,
                           (np.arange(36) + .5) / 36 * 2 - 1)
        rays = np.stack((np.ones_like(u), -u * math.tan(math.radians(89.51) / 2),
                         -v * math.tan(math.radians(58.29) / 2)), -1)
        self.rays = (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3)
        self.renderer = None
        self.camera_distance, self.camera_azimuth, self.camera_elevation = 3.4, 55., -25.

    def set_reference(self, motion, frame):
        d = self.data
        d.qpos[:3] = motion["body_pos_w"][frame, 0]
        d.qpos[3:7] = motion["body_quat_w"][frame, 0, [3, 0, 1, 2]]
        d.qpos[self.qids] = motion["joint_pos"][frame]
        d.qvel[:3] = motion["body_lin_vel_w"][frame, 0]
        rot = Rotation.from_quat(motion["body_quat_w"][frame, 0]).as_matrix()
        d.qvel[3:6] = rot.T @ motion["body_ang_vel_w"][frame, 0]
        d.qvel[self.vids] = motion["joint_vel"][frame]
        mujoco.mj_forward(self.model, d)

    def depth(self):
        rot = self.data.xmat[self.waist].reshape(3, 3)
        eye = self.data.xpos[self.waist] + rot @ self.camera_pos
        rays = np.ascontiguousarray(self.rays @ (rot @ self.camera_rot).T)
        distance = np.empty(len(rays))
        geomids = np.empty(len(rays), dtype=np.int32)
        # Training rays hit terrain and robot meshes, not duplicate collision shapes.
        groups = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        mujoco.mj_multiRay(self.model, self.data, eye, rays.ravel(), groups, True, -1,
                          geomids, distance, None, len(rays), 10.)
        depth = np.where(distance >= 0, distance * self.rays[:, 0], 2.5 if self.aligned_depth else 0.).reshape(36, 64)
        if self.aligned_depth:
            depth = depth.clip(0, 2.5)
        top, bottom, left, right = self.depth_crop
        depth = cv2.GaussianBlur(depth[top:36-bottom, left:64-right].astype(np.float32), (3, 3), 1,
                                 borderType=cv2.BORDER_REFLECT_101)
        depth = ((depth.clip(.1, 2.5) - .1) / 2.4 * 255).round().astype(np.uint8)
        if not self.depth_history:
            self.depth_history.extend([depth] * self.depth_history_length)
        else:
            self.depth_history.append(depth)
        self.last_depth = depth
        return torch.from_numpy(np.stack([self.depth_history[self.depth_history_length - 1 - offset - self.depth_delay]
                                         for offset in range(35, -1, -5)]))[None]

    def observe(self, target_heading, target_velocity, heading_valid=None, velocity_valid=None):
        d = self.data
        rot = d.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        # BODY local velocity uses the inertial principal axes, not the link axes.
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        angular_velocity_body = rot.T @ velocity[:3]
        tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32)[None]
        values = dict(joint_pos=tensor(d.qpos[self.qids] - self.default),
                      joint_vel=tensor(d.qvel[self.vids]), projected_gravity=tensor(-rot[2]),
                      base_ang_vel=tensor(angular_velocity_body * self.base_ang_vel_scale),
                      actions=tensor(self.last_action))
        heading = F.normalize(tensor(rot[:2, 0]), dim=-1)
        heading_valid = (torch.ones(1, 1, dtype=torch.bool)
                         if heading_valid is None else heading_valid.reshape(1, 1).bool())
        velocity_valid = (torch.ones(1, 1, dtype=torch.bool)
                          if velocity_valid is None else velocity_valid.reshape(1, 1).bool())
        obs = dict(state=torch.cat([values[k] for k in ("joint_pos", "joint_vel", "projected_gravity", "base_ang_vel")], -1),
                   last_action=values["actions"], history_actor=self.history.flat(),
                   heading=heading_observation(heading, target_heading[None], heading_valid),
                   velocity_command=velocity_observation(target_velocity[None], heading, velocity_valid),
                   depth_image=self.depth())
        return obs, values

    def prepare_action_target(self, action, rescale):
        # Keep the policy's action history unchanged; clip only the PD target.
        self.last_action = np.clip(action * rescale, -rescale, rescale)
        self.raw_joint_target = self.default + self.action_scale * self.last_action
        self.joint_target = (np.clip(self.raw_joint_target, *self.joint_limits.T)
                             if self.clip_joint_targets else self.raw_joint_target.copy())
        return self.joint_target

    def advance(self, action, rescale):
        target = self.prepare_action_target(action, rescale)
        for _ in range(4):
            torque = self.motor.compute(target, self.data.qpos[self.qids], self.data.qvel[self.vids])
            self.data.ctrl[self.aids] = torque
            mujoco.mj_step(self.model, self.data, nstep=2)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced non-finite state")

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=640, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.lookat[2] = max(.55, camera.lookat[2])
        camera.distance = self.camera_distance
        camera.azimuth, camera.elevation = self.camera_azimuth, self.camera_elevation
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

def place_on_stairs(environment):
    """Move onto the low_stairs_down level-5 origin; never correct pose after reset."""
    origin = np.asarray(terrain_metadata()["stairs_reset"]["origin"], dtype=np.float64)
    environment.data.qpos[0] = origin[0]
    environment.data.qpos[1] = origin[1]
    mujoco.mj_forward(environment.model, environment.data)
    point = environment.data.qpos[:3].copy()
    point[2] = 5.
    geomid = np.empty(1, dtype=np.int32)
    distance = mujoco.mj_ray(environment.model, environment.data, point, np.array([0., 0., -1.]),
                             np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, geomid)
    name = environment.model.geom(int(geomid[0])).name if distance >= 0 else ""
    if distance < 0 or not name.startswith("terrain_"):
        raise ValueError("The stairs reset must be supported by the training terrain")
    height = float(point[2] - distance)
    environment.data.qpos[2] += height
    mujoco.mj_forward(environment.model, environment.data)
    return height

def compose(simulations, seconds, total, errors, checkpoint_step, motion_name):
    frame = np.full((720, 1920, 3), 24, dtype=np.uint8)
    labels = ("MOTION REFERENCE", "FLAT | POLICY TRACKING", "INV STAIRS | POLICY TRACKING")
    for index, (sim, label) in enumerate(zip(simulations, labels)):
        x = index * 640
        frame[50:690, x:x + 640] = sim.render()
        cv2.putText(frame, label, (x + 18, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2, cv2.LINE_AA)
        info = f"{motion_name}  {seconds:06.2f}/{total:.2f}s" if index == 0 else f"Joint L2: {errors[index-1]:.3f} rad | model {checkpoint_step:,}"
        cv2.putText(frame, info, (x + 16, 710), cv2.FONT_HERSHEY_SIMPLEX, .46, (230, 230, 230), 1, cv2.LINE_AA)
        if index and sim.last_depth is not None:
            preview = cv2.resize(sim.last_depth, (144, 72), interpolation=cv2.INTER_NEAREST)
            frame[604:676, x + 480:x + 624] = np.repeat(preview[..., None], 3, axis=-1)
    return frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--physics-profile", choices=("native", "legacy"), default="native",
                        help="Native Newton collision/limit asset, or legacy MJCF geometry; both use the training actuator")
    parser.add_argument("--physics-asset", type=Path, default=NATIVE_PHYSICS.path,
                        help="Robot physics asset exported from a nominal Newton model; independent of policy weights")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--motion-dir", "--manifest", dest="motion_dir",
        type=Path,
        default=default_motion_directory(),
    )
    parser.add_argument("--motion", default="dance1_subject2")
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument(
        "--context", type=Path,
        help="Deployment context_1.npz; use with --context-start-frame (default: 50)",
    )
    parser.add_argument(
        "--context-start-frame", type=int, default=50,
        help="Source-motion frame represented by context[0] when the sidecar has no start_frame field",
    )
    parser.add_argument(
        "--window-size", type=int, choices=(3, 5), default=5,
        help="Runtime z smoothing window; matches deployment's supported 3/5-frame setting (default: 5)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.8,
        help="Future-frame z weight decay used by deployment (default: 0.8)",
    )
    parser.add_argument("--duration", type=float, help="Default: the entire remaining motion")
    parser.add_argument("--hold-frame", type=int,
                        help="Freeze the reference, heading, and velocity at this zero-based motion frame")
    parser.add_argument(
        "--start-default", action="store_true",
        help="Start policy robots at configured default joints with zero velocity while keeping the motion target",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, choices=(25, 50), default=25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--camera-distance", type=float, default=3.4)
    parser.add_argument("--camera-azimuth", type=float, default=55.)
    parser.add_argument("--camera-elevation", type=float, default=-25.)
    args = parser.parse_args()
    if args.start < 0 or (args.duration is not None and args.duration <= 0):
        parser.error("Start must be nonnegative and duration positive")
    if args.context_start_frame < 0:
        parser.error("Context start frame must be nonnegative")
    if not math.isfinite(args.gamma) or args.gamma <= 0.0:
        parser.error("Gamma must be finite and positive")
    if args.hold_frame is not None and args.hold_frame < 0:
        parser.error("Hold frame must be nonnegative")
    if args.camera_distance <= 0:
        parser.error("Camera distance must be positive")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    checkpoint = (args.checkpoint or latest_checkpoint(args.run_dir)).resolve()
    mode_suffix = f"_HOLD_frame{args.hold_frame}" if args.hold_frame is not None else ""
    output = args.output or checkpoint.parent / "videos" / f"{args.motion}_{checkpoint.stem}{mode_suffix}_sim2sim_linkframe_rp1terrain.mp4"
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing recording: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg = read_config(checkpoint.parent / "params/env.yaml")
    source = motion_clip_paths(args.motion_dir, [args.motion])[0]
    loaded = load_motion(source)
    names = loaded["joint_names"].tolist()
    body_names = loaded["body_names"].tolist()
    full_motion = {key: loaded[key] for key in ("fps", "joint_pos", "joint_vel", *FULL_BODY_FIELDS)}
    if float(full_motion["fps"].item()) != 50:
        raise ValueError("Expected 50 Hz motion with XYZW quaternions")
    if args.hold_frame is not None:
        if args.hold_frame >= len(full_motion["joint_pos"]):
            raise ValueError(f"Hold frame {args.hold_frame} is outside the {len(full_motion['joint_pos'])}-frame motion")
        hold_steps = round((args.duration if args.duration is not None else 15.) * 50)
        motion_indices = np.full(hold_steps + 1, args.hold_frame, dtype=np.int64)
        motion = {
            k: np.repeat(v[args.hold_frame:args.hold_frame + 1], hold_steps + 1, axis=0)
            if k != "fps" else v
            for k, v in full_motion.items()
        }
    else:
        start = round(args.start * 50)
        end = (len(full_motion["joint_pos"]) if args.duration is None
               else min(len(full_motion["joint_pos"]), start + round(args.duration * 50) + 1))
        if end - start < 2:
            raise ValueError("Requested motion interval is empty")
        motion_indices = np.arange(start, end, dtype=np.int64)
        motion = {k: v[start:end].copy() if k != "fps" else v for k, v in full_motion.items()}
    motion["body_pos_w"][..., :2] -= motion["body_pos_w"][0, 0, :2].copy()
    terrain = terrain_metadata()
    physics_asset = NewtonPhysicsAsset(args.physics_asset, data_root=DATA_ROOT)
    sims = [Simulation(make_model(stairs, args.physics_profile, physics_asset=physics_asset), names, env_cfg,
                       physics_asset=physics_asset)
            for stairs in (False, False, True)]
    process = None
    temporary_video = None
    try:
        for sim in sims:
            sim.camera_distance = args.camera_distance
            sim.camera_azimuth = args.camera_azimuth
            sim.camera_elevation = args.camera_elevation
            sim.set_reference(motion, 0)
        if args.start_default:
            # Deployment starts after the motor reset pose, while the first
            # sidecar context still targets the motion reference. Preserve
            # the reference yaw so heading encoding remains comparable.
            yaw = Rotation.from_quat(motion["body_quat_w"][0, 0]).as_euler("xyz")[2]
            default_quat = Rotation.from_euler("z", yaw).as_quat()
            for sim in sims[1:]:
                sim.data.qpos[3:7] = default_quat[[3, 0, 1, 2]]
                sim.data.qpos[sim.qids] = sim.default
                sim.data.qvel[:] = 0.0
                mujoco.mj_forward(sim.model, sim.data)
        stairs_reset_height = place_on_stairs(sims[2])
        body_ids = [sims[0].model.body(name).id for name in body_names]
        fk_error = float(np.max(np.linalg.norm(sims[0].data.xpos[body_ids] - motion["body_pos_w"][0], axis=-1)))
        if fk_error > .025:
            raise ValueError(f"Motion/MJCF kinematic mismatch: {fk_error:.4f} m")
        policy = Policy(checkpoint)
        exported_context = args.context is not None
        if not exported_context:
            contexts, headings, velocities = policy.contexts(motion, sims[0].default)
            heading_valid = velocity_valid = torch.ones(len(contexts), 1, dtype=torch.bool)
            context_source = "checkpoint backward map"
        else:
            context_path = args.context.resolve()
            if not context_path.is_file():
                raise FileNotFoundError(f"Context sidecar does not exist: {context_path}")
            contexts, headings, velocities, heading_valid, velocity_valid = policy.exported_contexts(
                context_path, full_motion, motion_indices, sims[0].default, args.context_start_frame,
            )
            context_source = str(context_path)
        n = len(contexts) - 1
        hold_info = f"; hold frame: {args.hold_frame}" if args.hold_frame is not None else ""
        print(f"Checkpoint: {checkpoint}\nFrames: {n}; duration: {n / 50:.2f}s{hold_info}; "
              f"FK error: {fk_error:.6f}m\nContext: {context_source}; "
              f"z_window={args.window_size}, gamma={args.gamma}", flush=True)
        trajectory = np.empty((n + 1, 2, 24), dtype=np.float32)
        root_positions = np.empty((n + 1, 2, 3), dtype=np.float32)
        actions = np.empty((n, 2, 24), dtype=np.float32)
        depth_range = [255, 0]
        started = time.monotonic()
        with tempfile.NamedTemporaryFile(prefix=".pufo-export-", suffix=".mp4", dir=output.parent,
                                         delete=False) as stream:
            temporary_video = Path(stream.name)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                   "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x720", "-r", str(args.fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_video)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for step in range(n + 1):
            sims[0].set_reference(motion, step)
            for index, sim in enumerate(sims[1:]):
                trajectory[step, index] = sim.data.qpos[sim.qids]
                root_positions[step, index] = sim.data.qpos[:3]
            errors = np.linalg.norm(trajectory[step] - motion["joint_pos"][step], axis=-1)
            if step % (50 // args.fps) == 0:
                process.stdin.write(compose(sims, step / 50, n / 50, errors, policy.step, args.motion).tobytes())
            if step == n:
                break
            pairs = [sim.observe(headings[step], velocities[step], heading_valid[step], velocity_valid[step])
                     for sim in sims[1:]]
            obs = {key: torch.cat([pair[0][key] for pair in pairs]) for key in pairs[0][0]}
            # Both paths contain one raw z per frame.  The real deployment
            # applies this same future-window weighted smoothing at inference.
            context_step = step if exported_context else step + 1
            context = weighted_context_window(
                contexts, context_step, args.gamma, args.window_size,
            ).unsqueeze(0)
            action = policy.act(obs, context.expand(2, -1)).numpy()
            if not np.isfinite(action).all():
                raise FloatingPointError("Policy returned non-finite actions")
            actions[step] = action
            for index, (sim, (_, values)) in enumerate(zip(sims[1:], pairs)):
                sim.advance(action[index], policy.action_rescale)
                sim.history.add(values)
                depth_range = [min(depth_range[0], int(sim.last_depth.min())), max(depth_range[1], int(sim.last_depth.max()))]
            if step % 250 == 0:
                print(f"t={step / 50:.1f}s/{n / 50:.1f}s joint_l2={errors.round(3).tolist()} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoder failed")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i",
                        str(temporary_video), "-f", "null", "-"], check=True)
        np.savez_compressed(output.with_suffix(".npz"), joint_pos=trajectory, root_pos=root_positions,
                            actions=actions, reference_joint_pos=motion["joint_pos"], fps=50)
        report = dict(checkpoint=str(checkpoint), model_step=policy.step, source_motion=str(source.resolve()),
                      physics_profile=args.physics_profile,
                      physics_asset=str(physics_asset.path),
                      physics_profile_sha256=(hashlib.sha256(physics_asset.path.read_bytes()).hexdigest()
                                              if args.physics_profile == "native" else None),
                      motor_effort_limits_nm=sims[0].limit.tolist(),
                      motor_velocity_limits_rad_s=sims[0].speed.tolist(),
                      joint_order=names,
                      motion=args.motion, start_s=args.start, hold_frame=args.hold_frame,
                      start_default=args.start_default,
                      context_source=context_source,
                      duration_s=n / 50, video_fps=args.fps,
                      panels=["kinematic_reference", "flat_policy", "stairs_policy"],
                      terrain_asset=str(TERRAIN_XML), terrain_meta=str(TERRAIN_XML.with_name("meta.json")),
                      terrain_generator=terrain["generator"], terrain_source=terrain["source"],
                      terrain_types=terrain["terrain_types"],
                      terrain_level=terrain["stairs_reset"]["terrain_level"],
                      terrain_type=terrain["stairs_reset"]["terrain_type"],
                      terrain_tile_size_m=terrain["tile_size"],
                      terrain_grid=[terrain["num_rows"], terrain["num_cols"]],
                      terrain_outer_extent_m=terrain["outer_extent_m"],
                      terrain_ngeom=terrain["ngeom"], terrain_nhfield=terrain["nhfield"],
                      stairs_reset_origin_m=terrain["stairs_reset"]["origin"],
                      terrain_surroundings="full mjlab rp1_simple terrain map",
                      stairs_reset_height_m=stairs_reset_height,
                      control_hz=50, actuator_hz=200, physics_hz=400, inference_device="cpu",
                      observation_noise=False, actuator_delay_ticks=0, depth_delay_frames=1,
                      force_perturbations=False, automatic_resets=False,
                      context_window_size=args.window_size, context_gamma=args.gamma,
                      angular_velocity_frame="base_link (world angular velocity rotated by link orientation)",
                      camera_distance=args.camera_distance, camera_azimuth=args.camera_azimuth,
                      camera_elevation=args.camera_elevation,
                      action_target="clamp(default_joint_pos + action_scale * clamp(action_rescale * actor_mean, -action_rescale, action_rescale), joint_min, joint_max)",
                      clip_joint_targets=True,
                      initial_fk_error_m=fk_error, depth_uint8_range=depth_range,
                      elapsed_seconds=time.monotonic() - started)
        initial_heading = Rotation.from_quat(motion["body_quat_w"][0, 0]).as_matrix()[:2, 0]
        for index, name in enumerate(("flat", "stairs")):
            errors = np.linalg.norm(trajectory[:, index] - motion["joint_pos"], axis=-1)
            low = np.flatnonzero(root_positions[:, index, 2] < .35)
            displacement = root_positions[:, index, :2] - root_positions[0, index, :2]
            forward = displacement @ initial_heading
            report[name] = dict(mean_joint_l2_rad=float(errors.mean()), max_joint_l2_rad=float(errors.max()),
                                final_joint_l2_rad=float(errors[-1]),
                                first_root_below_035m_s=float(low[0] / 50) if len(low) else None,
                                root_displacement_xy_m=displacement[-1].tolist(),
                                forward_displacement_m=float(forward[-1]),
                                max_backward_excursion_m=float(max(0., -forward.min())))
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
