"""Run an RP1 AMP checkpoint in MuJoCo and record ramp, fixed speed, or yaw commands.

The observation, joint mapping, PD gains, and 50 Hz control contract match
``sim2sim_rp1_parkour.py``. AMP checkpoints are loaded directly, so Isaac Sim
and a separately exported TorchScript policy are not required.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robo_newton"))
from robolab.utils.command_path import command_path_features

DEFAULT_RUN = Path(
    "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp/"
    "2026-10-09_02-24-22_rp1_amp_lsgan_disc4_currsync_20261009"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN / "model_9999.pt"
DEFAULT_MJCF = ROOT / "robo_newton/data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"

# MuJoCo/URDF traversal order used by robot_24dof.xml.
MJ_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_joint",
)

# The AMP run resolved ``joint_names=[".*"]`` to this exact MuJoCo/URDF
# traversal order. The interleaved usd2urdf mapping hard-coded in the parkour
# sim2sim script does not match this checkpoint and must not be used here.
POLICY_TO_MJ = np.arange(len(MJ_JOINT_NAMES), dtype=np.int64)
FOOT_CONTACT_FORCE_THRESHOLD_N = 5.0
POLICY_TERM_WIDTHS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": len(MJ_JOINT_NAMES),
    "joint_vel": len(MJ_JOINT_NAMES),
    "actions": len(MJ_JOINT_NAMES),
    "command_path_state": 4,
}


class ConfigLoader(yaml.SafeLoader):
    pass


for _tag in ("python/tuple", "python/object/apply:builtins.slice"):
    ConfigLoader.add_constructor(
        "tag:yaml.org,2002:" + _tag,
        lambda loader, node: loader.construct_sequence(node),
    )


def read_config(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=ConfigLoader)


def values_by_regex(value, names: tuple[str, ...], default: float | None = None) -> np.ndarray:
    import re

    if isinstance(value, (int, float)):
        return np.full(len(names), value, dtype=np.float64)
    result = []
    for name in names:
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1 or (not matches and default is None):
            raise ValueError(f"Ambiguous or missing config for {name}: {matches}")
        result.append(matches[0] if matches else default)
    return np.asarray(result, dtype=np.float64)


def activation(name: str, value: torch.Tensor) -> torch.Tensor:
    functions = {"elu": F.elu, "relu": F.relu, "tanh": torch.tanh}
    return functions[name.lower()](value)


class AMPPolicy:
    """Minimal deterministic AMP actor reconstructed from an RSL-RL checkpoint."""

    def __init__(self, checkpoint: Path):
        config = read_config(checkpoint.parent / "params/agent.yaml")
        self.activation_name = str(config["actor"]["activation"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        weights = state["actor_state_dict"]
        self.mean = weights["obs_normalizer._mean"]
        self.std = weights["obs_normalizer._std"]
        self.estimator = self._layers(weights, "velocity_estimator")
        self.actor = self._layers(weights, "mlp")
        self.iteration = int(state["iter"])
        self.observation_size = self.mean.shape[-1]
        if self.mean.ndim != 2 or self.mean.shape[0] != 1 or self.actor[-1][0].shape[0] != 24:
            raise ValueError(
                f"Unsupported AMP checkpoint contract: obs={tuple(self.mean.shape)}, "
                f"actions={self.actor[-1][0].shape[0]}"
            )

    @staticmethod
    def _layers(weights: dict, prefix: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        indices = sorted(
            int(key.split(".")[1])
            for key in weights
            if key.startswith(prefix + ".") and key.endswith(".weight")
        )
        return [(weights[f"{prefix}.{index}.weight"], weights[f"{prefix}.{index}.bias"]) for index in indices]

    def _forward_layers(self, value: torch.Tensor, layers) -> torch.Tensor:
        for index, (weight, bias) in enumerate(layers):
            value = F.linear(value, weight, bias)
            if index + 1 < len(layers):
                value = activation(self.activation_name, value)
        return value

    @torch.inference_mode()
    def __call__(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if observation.shape != (self.observation_size,):
            raise ValueError(f"Checkpoint expects {self.observation_size} observations, got {observation.shape}")
        value = torch.from_numpy(observation.astype(np.float32, copy=False))[None]
        normalized = (value - self.mean) / (self.std + 1.0e-2)
        estimated_velocity = self._forward_layers(normalized, self.estimator)
        action = self._forward_layers(torch.cat((normalized, estimated_velocity), dim=-1), self.actor)
        return action[0].numpy(), estimated_velocity[0].numpy()


class TermHistory:
    """Isaac Lab CircularBuffer semantics, flattened oldest to newest."""

    def __init__(self, length: int, width: int):
        self.length = length
        self.values: deque[np.ndarray] = deque(maxlen=length)
        self.width = width

    def append(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(self.width)
        if not self.values:
            self.values.extend([value.copy() for _ in range(self.length)])
        else:
            self.values.append(value.copy())

    def flat(self) -> np.ndarray:
        return np.concatenate(self.values)


def policy_histories_from_config(env_config: dict) -> dict[str, TermHistory]:
    """Match ObservationManager's group override and per-term history semantics."""
    policy_config = env_config["observations"]["policy"]
    if not policy_config["concatenate_terms"]:
        raise ValueError("MuJoCo AMP inference requires concatenated policy observations")
    group_history = policy_config.get("history_length")
    histories = {}
    for name, term_config in policy_config.items():
        if not isinstance(term_config, dict) or "func" not in term_config:
            continue
        history_config = policy_config if group_history is not None else term_config
        history_length = history_config.get("history_length", 0)
        if history_length > 0 and not history_config.get("flatten_history_dim", True):
            raise ValueError("MuJoCo AMP inference requires flattened policy histories")
        histories[name] = TermHistory(max(1, history_length), POLICY_TERM_WIDTHS[name])
    return histories


def foot_contact_metrics(model, data, foot_ids):
    """Read loaded point slip against this recorder's static world-body ground.

    Query the last physics substep without forwarding or changing simulator
    state. Averaging speed magnitudes preserves sliding during sole rotation.
    """
    normal_force = np.zeros(len(foot_ids))
    weighted_speed = np.zeros(len(foot_ids))
    wrench = np.zeros(6)
    jacobian = np.zeros((3, model.nv))
    for contact_index, contact in enumerate(data.contact):
        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if 0 not in bodies:
            continue
        for foot_index, foot_id in enumerate(foot_ids):
            if foot_id not in bodies:
                continue
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            load = max(0.0, wrench[0])
            normal = contact.frame[:3]
            # MuJoCo stores the midpoint; recover the foot's surface point as Newton does.
            side = -1.0 if bodies[0] == foot_id else 1.0
            point = contact.pos + side * 0.5 * contact.dist * normal
            mujoco.mj_jac(model, data, jacobian, None, point, foot_id)
            velocity = jacobian @ data.qvel
            tangent = velocity - np.dot(velocity, normal) * normal
            normal_force[foot_index] += load
            weighted_speed[foot_index] += load * np.linalg.norm(tangent)
    return {
        "foot_normal_force": normal_force,
        "foot_contact_slip_speed": weighted_speed / np.maximum(normal_force, 1.0e-6),
    }


def contact_slip_summary(telemetry, selected, body_weight, force_threshold_ratio=0.05):
    """Report only the requested pre-fall samples; no-contact speed is undefined."""
    speed = telemetry["foot_contact_slip_speed"][selected]
    threshold = body_weight * force_threshold_ratio
    loaded = telemetry["foot_normal_force"][selected] > threshold
    return {
        "contact_point_convention": "foot_surface",
        "sample_count": int(selected.sum()),
        "loaded_foot_sample_count": int(loaded.sum()),
        "normal_force_threshold_n": float(threshold),
        "mean_loaded_foot_speed_mps": float(speed[loaded].mean()) if loaded.any() else None,
        "p95_loaded_foot_speed_mps": float(np.quantile(speed[loaded], .95)) if loaded.any() else None,
        "max_loaded_foot_speed_mps": float(speed[loaded].max()) if loaded.any() else None,
        "mean_sum_loaded_foot_speed_mps": float((speed * loaded).sum(axis=-1).mean()) if len(speed) else None,
    }


class Simulation:
    def __init__(self, mjcf: Path, env_config: dict, camera_distance: float, initial_height: float | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        self.jids = np.asarray([self.model.joint(name).id for name in MJ_JOINT_NAMES])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        self.aids = np.asarray(
            [np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)[0] for joint in self.jids]
        )
        self.foot_ids = np.asarray([
            self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")
        ])

        robot = env_config["scene"]["robot"]
        actuator = robot["actuators"]["all_joints"]
        self.default = values_by_regex(robot["init_state"]["joint_pos"], MJ_JOINT_NAMES, 0.0)
        self.kp = values_by_regex(actuator["stiffness"], MJ_JOINT_NAMES)
        self.kd = values_by_regex(actuator["damping"], MJ_JOINT_NAMES)
        self.effort_limit = values_by_regex(actuator["effort_limit_sim"], MJ_JOINT_NAMES)
        self.speed_limit = values_by_regex(actuator["velocity_limit_sim"], MJ_JOINT_NAMES)
        self.keep_velocity = values_by_regex(actuator["keep_velocity"], MJ_JOINT_NAMES)
        self.model.dof_armature[self.vids] = values_by_regex(actuator["armature"], MJ_JOINT_NAMES)
        self.model.dof_damping[self.vids] = values_by_regex(
            actuator["viscous_friction"], MJ_JOINT_NAMES
        )
        self.model.dof_frictionloss[self.vids] = values_by_regex(
            actuator["friction"], MJ_JOINT_NAMES
        )
        self.model.actuator_ctrllimited[self.aids] = 0
        self.model.actuator_forcelimited[self.aids] = 0
        self.action_scale = float(env_config["actions"]["joint_pos"]["scale"])

        self.initial_root_height = robot["init_state"]["pos"][2] if initial_height is None else initial_height
        self.data.qpos[:3] = (0.0, 0.0, self.initial_root_height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qids] = self.default
        mujoco.mj_forward(self.model, self.data)

        self.reset_command_origin()

        self.last_action = np.zeros(24, dtype=np.float32)
        self.target = self.default.copy()
        self.step_braking_power_peak = 0.0
        self.step_braking_energy = 0.0
        self.step_torque_peak = np.zeros(24)
        self.step_torque_delta_ratio_peak = 0.0
        self._previous_applied_torque = None
        self.histories = policy_histories_from_config(env_config)
        self.renderer = None
        self.camera_distance = camera_distance

    def reset_command_origin(self) -> None:
        """Start a path segment; the recorder holds one segment for the full clip."""
        self.command_origin_xy = self.data.qpos[:2].copy()
        rotation = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
        self.command_origin_heading = np.arctan2(rotation[1, 0], rotation[0, 0])

    def observe(self, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = self.data.xmat[self.model.body("base_link").id].reshape(3, 3)
        velocity = rotation.T @ self.data.qvel[:3]
        angular_velocity = self.data.sensor("angular-velocity").data.copy()
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        relative_position_mj = self.data.qpos[self.qids] - self.default
        joint_velocity_mj = self.data.qvel[self.vids]
        values = {
            "base_ang_vel": angular_velocity,
            "projected_gravity": gravity,
            "velocity_commands": command,
            "joint_pos": relative_position_mj[POLICY_TO_MJ],
            "joint_vel": joint_velocity_mj[POLICY_TO_MJ],
            "actions": self.last_action,
        }
        if "command_path_state" in self.histories:
            delta_xy = self.data.qpos[:2] - self.command_origin_xy
            sine, cosine = np.sin(self.command_origin_heading), np.cos(self.command_origin_heading)
            displacement = np.array([
                cosine * delta_xy[0] + sine * delta_xy[1],
                -sine * delta_xy[0] + cosine * delta_xy[1],
            ])
            heading = np.arctan2(rotation[1, 0], rotation[0, 0])
            values["command_path_state"] = command_path_features(
                torch.as_tensor(displacement),
                torch.as_tensor(heading - self.command_origin_heading),
                torch.as_tensor(command),
            ).numpy()
        for name, history in self.histories.items():
            history.append(values[name])
        observation = np.concatenate([self.histories[name].flat() for name in self.histories])
        return observation.astype(np.float32), velocity

    def set_action(self, action: np.ndarray) -> None:
        if not np.isfinite(action).all():
            raise FloatingPointError("AMP policy returned non-finite actions")
        self.last_action = action.astype(np.float32, copy=True)
        self.target[:] = self.default
        self.target[POLICY_TO_MJ] += self.action_scale * action
        self.step_braking_power_peak = 0.0
        self.step_braking_energy = 0.0
        self.step_torque_peak.fill(0.0)
        self.step_torque_delta_ratio_peak = 0.0

    def advance(self) -> None:
        velocity = self.data.qvel[self.vids]
        torque = self.kp * (self.target - self.data.qpos[self.qids]) - self.kd * velocity
        drive = self.effort_limit * np.clip(
            (self.speed_limit - np.abs(velocity)) / (self.speed_limit - self.keep_velocity), 0.0, 1.0
        )
        lower = np.where(velocity <= 0.0, -drive, -self.effort_limit)
        upper = np.where(velocity >= 0.0, drive, self.effort_limit)
        applied_torque = np.clip(torque, lower, upper)
        if self._previous_applied_torque is not None:
            delta = float(np.max(np.abs(applied_torque - self._previous_applied_torque) / self.effort_limit))
            self.step_torque_delta_ratio_peak = max(self.step_torque_delta_ratio_peak, delta)
        self._previous_applied_torque = applied_torque.copy()
        self.data.ctrl[self.aids] = applied_torque
        # Mechanical braking power is a proxy, not measured electrical bus power.
        braking_power = float(np.maximum(-applied_torque * velocity, 0.0).sum())
        self.step_braking_power_peak = max(self.step_braking_power_peak, braking_power)
        self.step_braking_energy += braking_power * self.model.opt.timestep
        self.step_torque_peak = np.maximum(self.step_torque_peak, np.abs(applied_torque))
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced non-finite state")

    def render(self, width: int, height: int) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.lookat[0] += 0.15
        camera.lookat[2] = max(0.58, camera.lookat[2])
        camera.distance = self.camera_distance
        camera.azimuth = 62.0
        camera.elevation = -12.0
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def measure(self) -> dict[str, np.ndarray]:
        """Read root state after integration and contacts from the last physics substep."""
        # MuJoCo stores wxyz; SciPy expects xyzw. This rotation maps body to world.
        root_quaternion_xyzw = self.data.qpos[[4, 5, 6, 3]]
        body_to_world = Rotation.from_quat(root_quaternion_xyzw).as_matrix()
        foot_forward_b = np.asarray([
            body_to_world.T @ self.data.xmat[foot].reshape(3, 3)[:, 0]
            for foot in self.foot_ids
        ])
        return {
            "root_position_w": self.data.qpos[:3].copy(),
            "root_quaternion_wxyz": self.data.qpos[3:7].copy(),
            "heading_rad": np.asarray(np.arctan2(body_to_world[1, 0], body_to_world[0, 0])),
            "body_velocity": body_to_world.T @ self.data.qvel[:3],
            "body_angular_velocity": self.data.sensor("angular-velocity").data.copy(),
            "joint_position": self.data.qpos[self.qids].copy(),
            "joint_velocity": self.data.qvel[self.vids].copy(),
            "joint_target": self.target.copy(),
            "joint_torque_peak_nm": self.step_torque_peak.copy(),
            "joint_torque_delta_ratio_peak": np.asarray(self.step_torque_delta_ratio_peak),
            "braking_power_peak_w": np.asarray(self.step_braking_power_peak),
            "braking_energy_j": np.asarray(self.step_braking_energy),
            "foot_position_w": self.data.xpos[self.foot_ids].copy(),
            "foot_heading_relative_base_rad": np.arctan2(foot_forward_b[:, 1], foot_forward_b[:, 0]),
            **foot_contact_metrics(self.model, self.data, self.foot_ids),
        }


def ramp_command(
    seconds: float, duration: float, maximum: float, ramp_seconds: float | None = None,
    deceleration_seconds: float | None = None, standing_seconds: float = 0.0,
) -> float:
    settle = min(0.5, duration * 0.05)
    hold = min(1.0, duration * 0.1)
    ramp_duration = ramp_seconds if ramp_seconds is not None else max(duration - settle - hold, 1.0e-6)
    fraction = (seconds - settle) / ramp_duration
    if deceleration_seconds is not None:
        braking_fraction = (duration - standing_seconds - seconds) / deceleration_seconds
        fraction = min(fraction, braking_fraction)
    return maximum * np.clip(fraction, 0.0, 1.0)


def abrupt_speed_command(seconds: float, maximum: float, start: float, stop: float) -> float:
    """Hold zero, step to the requested speed, then step back to zero."""
    return maximum if start <= seconds < stop else 0.0


def speed_phase_metrics(values, telemetry, start, end, fallen_at):
    """Report a requested phase separately, retaining failures rather than filtering them out."""
    selected = (values[:, 0] >= start) & (values[:, 0] < end)
    if not selected.any():
        return {"start_s": start, "end_s": end, "sample_count": 0}
    positions = telemetry["root_position_w"][selected]
    velocity = telemetry["body_velocity"][selected]
    headings = np.unwrap(telemetry["heading_rad"])
    return {
        "start_s": start, "end_s": end, "sample_count": int(selected.sum()),
        "completed_without_fall": fallen_at is None or fallen_at >= end,
        "mean_actual_vx_mps": float(values[selected, 2].mean()),
        "vx_tracking_mae_mps": float(np.abs(values[selected, 2] - values[selected, 1]).mean()),
        "mean_planar_speed_mps": float(np.linalg.norm(velocity[:, :2], axis=-1).mean()),
        "max_abs_world_y_m": float(np.abs(positions[:, 1]).max()),
        "final_world_y_m": float(positions[-1, 1]),
        "xy_displacement_during_phase_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
        "max_abs_heading_deg": float(np.rad2deg(np.abs(headings[selected] - headings[0]).max())),
    }


def overlay(
    frame: np.ndarray,
    seconds: float,
    duration: float,
    command_text: str,
    measured_text: str,
    auxiliary_text: str,
    iteration: int,
    fallen: bool,
) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (24, 24), (440, 190), (18, 23, 29), -1)
    cv2.putText(frame, "RP1 AMP | MuJoCo sim2sim", (42, 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, command_text, (42, 84), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (70, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, measured_text, (42, 114), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (110, 245, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, auxiliary_text, (42, 144), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (220, 175, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{seconds:04.1f}/{duration:.1f}s   model_{iteration}", (42, 176),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 210, 215), 1, cv2.LINE_AA)
    if fallen:
        cv2.putText(frame, "FALL DETECTED", (frame.shape[1] - 360, 72), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (30, 40, 240), 3, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mujoco-xml", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--ramp-seconds", type=float,
                        help="Time to accelerate after settling; hold maximum speed for the rest of the clip.")
    parser.add_argument("--deceleration-seconds", type=float,
                        help="Brake to zero before the final standing phase; requires --ramp-seconds.")
    parser.add_argument("--abrupt-stop-at", type=float,
                        help="Step the forward command directly to zero at this time; keeps the raw stop input.")
    parser.add_argument("--abrupt-start-at", type=float,
                        help="Step directly to --max-speed at this time; requires --abrupt-stop-at.")
    parser.add_argument("--standing-seconds", type=float, default=0.0,
                        help="Final zero-speed hold after braking.")
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("--max-speed", type=float)
    command_group.add_argument("--yaw-rate", type=float)
    command_group.add_argument("--forward-speed", type=float)
    parser.add_argument("--fps", type=int, choices=(25, 50), default=50)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-distance", type=float, default=2.15)
    parser.add_argument("--initial-height", type=float,
                        help="Override saved initial root height; use 0.78 to reproduce older recordings.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    max_speed = args.max_speed
    if args.yaw_rate is None and args.forward_speed is None and max_speed is None:
        max_speed = 3.6
    if args.duration <= 0 or (max_speed is not None and max_speed < 0) or args.camera_distance <= 0:
        parser.error("duration and camera distance must be positive; max speed must be nonnegative")
    if args.ramp_seconds is not None:
        if max_speed is None:
            parser.error("--ramp-seconds applies only to a speed ramp")
        available_ramp_time = args.duration - min(0.5, args.duration * 0.05)
        if not np.isfinite(args.ramp_seconds) or not 0 < args.ramp_seconds < available_ramp_time:
            parser.error("ramp seconds must be positive and leave time to hold the maximum speed")
    if args.deceleration_seconds is not None:
        if args.ramp_seconds is None or max_speed is None:
            parser.error("braking requires a speed ramp with --ramp-seconds")
        if not np.isfinite(args.deceleration_seconds) or args.deceleration_seconds <= 0:
            parser.error("deceleration seconds must be finite and positive")
        if not np.isfinite(args.standing_seconds) or args.standing_seconds < 0:
            parser.error("standing seconds must be finite and nonnegative")
        cruise_start = min(0.5, args.duration * 0.05) + args.ramp_seconds
        cruise_end = args.duration - args.standing_seconds - args.deceleration_seconds
        if cruise_end <= cruise_start:
            parser.error("duration must leave time to cruise between acceleration and braking")
    elif args.standing_seconds != 0:
        parser.error("--standing-seconds requires --deceleration-seconds")
    if args.abrupt_stop_at is not None:
        if max_speed is None or not np.isfinite(args.abrupt_stop_at) or not 0 < args.abrupt_stop_at < args.duration:
            parser.error("abrupt stop time must be finite, positive, and inside the clip")
        if args.deceleration_seconds is not None:
            parser.error("--abrupt-stop-at cannot be combined with --deceleration-seconds")
    if args.abrupt_start_at is not None:
        if (max_speed is None or args.abrupt_stop_at is None
                or not np.isfinite(args.abrupt_start_at)
                or not 0 <= args.abrupt_start_at < args.abrupt_stop_at):
            parser.error("abrupt start requires --max-speed and a later --abrupt-stop-at")
        if args.ramp_seconds is not None or args.deceleration_seconds is not None:
            parser.error("abrupt start cannot be combined with acceleration or braking ramps")
    if args.yaw_rate is not None and not np.isfinite(args.yaw_rate):
        parser.error("yaw rate must be finite")
    if args.forward_speed is not None and not np.isfinite(args.forward_speed):
        parser.error("forward speed must be finite")
    if args.width % 2 or args.height % 2:
        parser.error("video width and height must be even")

    checkpoint = args.checkpoint.resolve()
    if args.yaw_rate is not None:
        recording_suffix = f"yaw_{args.yaw_rate:g}"
        command_description = f"fixed yaw rate {args.yaw_rate:g} rad/s"
    elif args.forward_speed is not None:
        recording_suffix = f"vx_{args.forward_speed:g}"
        command_description = f"fixed forward speed {args.forward_speed:g} m/s"
    else:
        recording_suffix = f"ramp_0_to_{max_speed:g}"
        command_description = f"linear ramp 0 to {max_speed:g} m/s"
        if args.abrupt_start_at is not None:
            recording_suffix = f"step_0_to_{max_speed:g}_at_{args.abrupt_start_at:g}"
            command_description = f"raw step to {max_speed:g} m/s at {args.abrupt_start_at:g}s"
        if args.deceleration_seconds is not None:
            recording_suffix += "_hold_stop"
            command_description += ", cruise, brake to zero, stand"
        if args.abrupt_stop_at is not None:
            recording_suffix += f"_abrupt_stop_at_{args.abrupt_stop_at:g}"
            command_description += f", raw step to zero at {args.abrupt_stop_at:g}s"
    default_filename = f"{checkpoint.stem}_amp_sim2sim_{recording_suffix}.mp4"
    output = (args.output or checkpoint.parent / "videos" / default_filename).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing recording: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)

    policy = AMPPolicy(checkpoint)
    environment = Simulation(args.mujoco_xml.resolve(), read_config(checkpoint.parent / "params/env.yaml"),
                             args.camera_distance, initial_height=args.initial_height)
    policy_steps = round(args.duration * 50)
    render_every = 50 // args.fps
    temporary_video = None
    encoder = None
    records = []
    states = []
    fallen_at = None
    started = time.monotonic()
    try:
        with tempfile.NamedTemporaryFile(prefix=".amp-sim2sim-", suffix=".mp4", dir=output.parent,
                                         delete=False) as stream:
            temporary_video = Path(stream.name)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
            "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}",
            "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", str(args.threads),
            "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temporary_video),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for step in range(policy_steps):
            seconds = step / 50.0
            if args.yaw_rate is not None:
                command_value = args.yaw_rate
                policy_command = np.array([0.0, 0.0, command_value], dtype=np.float32)
            else:
                if args.forward_speed is not None:
                    command_value = args.forward_speed
                elif args.abrupt_start_at is not None:
                    command_value = abrupt_speed_command(
                        seconds, max_speed, args.abrupt_start_at, args.abrupt_stop_at,
                    )
                else:
                    command_value = float(ramp_command(
                        seconds, args.duration, max_speed, args.ramp_seconds,
                        args.deceleration_seconds, args.standing_seconds,
                    ))
                    if args.abrupt_stop_at is not None and seconds >= args.abrupt_stop_at:
                        command_value = 0.0
                policy_command = np.array([command_value, 0.0, 0.0], dtype=np.float32)
            observation, _ = environment.observe(policy_command)
            action, estimated_velocity = policy(observation)
            environment.set_action(action)
            for _ in range(4):
                environment.advance()
            state = environment.measure()
            states.append(state)
            actual_vx = float(state["body_velocity"][0])
            actual_yaw = float(state["body_angular_velocity"][2])
            linear_drift = float(np.linalg.norm(state["body_velocity"][:2]))
            root_height = float(environment.data.qpos[2])
            if fallen_at is None and root_height < 0.45:
                fallen_at = seconds
            if args.yaw_rate is not None:
                measured_value = actual_yaw
                auxiliary_value = linear_drift
                command_text = f"command yaw {command_value:+4.2f} rad/s"
                measured_text = f"actual yaw  {actual_yaw:+4.2f} rad/s"
                auxiliary_text = f"XY offset {np.linalg.norm(state['root_position_w'][:2]):5.3f} m"
            else:
                measured_value = actual_vx
                auxiliary_value = float(estimated_velocity[0])
                command_text = f"command  {command_value:4.2f} m/s"
                measured_text = f"actual    {actual_vx:+4.2f} m/s"
                auxiliary_text = f"lateral offset {state['root_position_w'][1]:+5.3f} m"
            records.append((seconds, command_value, measured_value, auxiliary_value, root_height))
            if step % render_every == 0:
                frame = overlay(
                    environment.render(args.width, args.height), seconds, args.duration, command_text,
                    measured_text, auxiliary_text, policy.iteration, fallen_at is not None,
                )
                encoder.stdin.write(frame.tobytes())
            if step % 100 == 0:
                if args.yaw_rate is not None:
                    print(
                        f"t={seconds:4.1f}s cmd_yaw={command_value:+4.2f} "
                        f"actual_yaw={actual_yaw:+5.2f} drift={linear_drift:4.2f} "
                        f"height={root_height:.3f}", flush=True,
                    )
                else:
                    print(
                        f"t={seconds:4.1f}s cmd={command_value:4.2f} actual={actual_vx:+5.2f} "
                        f"estimated={estimated_velocity[0]:+5.2f} height={root_height:.3f}",
                        flush=True,
                    )
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoder failed")
        encoder = None
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i", str(temporary_video),
             "-f", "null", "-"],
            check=True,
        )
        values = np.asarray(records, dtype=np.float32)
        telemetry = {name: np.asarray([state[name] for state in states]) for name in states[0]}
        settled = values[:, 0] >= min(1.0, args.duration * 0.1)
        heading = np.unwrap(telemetry["heading_rad"])
        contacts = telemetry["foot_normal_force"][settled] > FOOT_CONTACT_FORCE_THRESHOLD_N
        lateral_velocity = telemetry["body_velocity"][settled, 1]
        yaw_velocity = telemetry["body_angular_velocity"][settled, 2]
        tracking_error = values[settled, 2] - values[settled, 1]
        if args.yaw_rate is not None:
            np.savez_compressed(
                output.with_suffix(".npz"), time=values[:, 0], command_yaw=values[:, 1],
                actual_yaw=values[:, 2], linear_drift=values[:, 3], root_height=values[:, 4],
                joint_names=np.asarray(MJ_JOINT_NAMES), **telemetry,
            )
        else:
            np.savez_compressed(
                output.with_suffix(".npz"), time=values[:, 0], command_vx=values[:, 1],
                actual_vx=values[:, 2], estimated_vx=values[:, 3], root_height=values[:, 4],
                joint_names=np.asarray(MJ_JOINT_NAMES), **telemetry,
            )
        report = {
            "initial_root_height_m": environment.initial_root_height,
            "checkpoint": str(checkpoint),
            "command": command_description,
            "iteration": policy.iteration,
            "mujoco_xml": str(args.mujoco_xml.resolve()),
            "duration_s": args.duration,
            "ramp_seconds_override": args.ramp_seconds,
            "deceleration_seconds": args.deceleration_seconds,
            "abrupt_stop_at_s": args.abrupt_stop_at,
            "abrupt_start_at_s": args.abrupt_start_at,
            "standing_seconds": args.standing_seconds,
            "video_fps": args.fps,
            "resolution": [args.width, args.height],
            "control_hz": 50,
            "physics_hz": 200,
            "minimum_root_height_m": float(values[:, 4].min()),
            "fallen_at_s": fallen_at,
            "automatic_resets": False,
            "elapsed_seconds": time.monotonic() - started,
            "mean_abs_lateral_velocity_mps": float(np.abs(lateral_velocity).mean()),
            "final_position_xy_m": telemetry["root_position_w"][-1, :2].tolist(),
            "heading_change_deg": float(np.rad2deg(heading[-1] - heading[0])),
            "flight_fraction_after_settle": float(np.mean(~contacts.any(axis=1))),
            "double_support_fraction_after_settle": float(np.mean(contacts.all(axis=1))),
            "foot_contact_force_threshold_n": FOOT_CONTACT_FORCE_THRESHOLD_N,
            "contact_slip_before_fall": contact_slip_summary(
                telemetry, settled & (True if fallen_at is None else values[:, 0] < fallen_at),
                environment.model.body_mass.sum() * np.linalg.norm(environment.model.opt.gravity),
            ),
        }
        if args.yaw_rate is not None:
            planar_displacement = telemetry["root_position_w"][:, :2] - telemetry["root_position_w"][0, :2]
            planar_distance = np.linalg.norm(planar_displacement, axis=1)
            report.update({
                "mean_actual_yaw_rps": float(values[settled, 2].mean()),
                "yaw_tracking_mae_rps": float(np.abs(tracking_error).mean()),
                "max_linear_drift_mps": float(values[settled, 3].max()),
                "max_planar_displacement_m": float(planar_distance.max()),
                "final_planar_displacement_m": float(planar_distance[-1]),
            })
        else:
            report.update({
                "final_actual_vx_mps": float(values[-1, 2]),
                "max_actual_vx_mps": float(values[:, 2].max()),
                "mean_actual_vx_mps": float(values[settled, 2].mean()),
                "vx_tracking_mae_mps": float(np.abs(tracking_error).mean()),
                "yaw_tracking_mae_rps": float(np.abs(yaw_velocity).mean()),
                "max_abs_heading_deg": float(np.rad2deg(np.abs(heading - heading[0]).max())),
                "max_abs_world_y_m": float(np.abs(telemetry["root_position_w"][:, 1]).max()),
                "velocity_estimator_rmse_x_mps": float(
                    np.sqrt(np.mean((values[:, 3] - values[:, 2]) ** 2))
                ),
            })
            if args.deceleration_seconds is not None:
                report["cruise"] = speed_phase_metrics(values, telemetry, cruise_start, cruise_end, fallen_at)
                report["standing"] = speed_phase_metrics(
                    values, telemetry, args.duration - args.standing_seconds, args.duration, fallen_at
                )
                for name, start, end in (
                    ("cruise", cruise_start, cruise_end),
                    ("braking", cruise_end, args.duration),
                    ("standing", args.duration - args.standing_seconds, args.duration),
                ):
                    if name not in report:
                        report[name] = speed_phase_metrics(values, telemetry, start, end, fallen_at)
                    selected = (values[:, 0] >= start) & (values[:, 0] < end)
                    if fallen_at is not None:
                        selected &= values[:, 0] < fallen_at
                    report[name]["contact_slip"] = contact_slip_summary(
                        telemetry, selected,
                        environment.model.body_mass.sum() * np.linalg.norm(environment.model.opt.gravity),
                    )
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        temporary_video.replace(output)
        print(json.dumps(report, indent=2), flush=True)
        print(f"VIDEO={output}", flush=True)
    finally:
        environment.close()
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.wait(timeout=30)
        if temporary_video is not None and temporary_video.exists():
            temporary_video.unlink()


if __name__ == "__main__":
    main()
