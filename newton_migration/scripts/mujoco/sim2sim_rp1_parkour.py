# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim2sim for RP1 parkour policies exported as separate ONNX graphs
# (depth encoder + actor), aligned with parkour_env_cfg observations and
# delayed_visualizable_image subsampling.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from typing import Any, ClassVar

import cv2
import glfw
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError as e:
    ort = None
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None

from robolab.assets import ISAAC_DATA_DIR

# Depth pipeline (encoder input aligned with ``noise_model.crop_and_resize``):
#   ⑴ MuJoCo offscreen depth at native **64×36** (same 16:9 as Isaac ray grid).
#   ⑵ **Per-pixel linearize** depth buffer → metric Z.
#   ⑶ **Crop on the 64×36 tensor**: ``(up, down, left, right)`` = ``(18, 0, 16, 16)`` —
#      identical indexing to ``CropAndResizeCfg`` on ``data[:, H, W]`` in ``noise_model.py``.
#   ⑷ Blur / clip / normalize on the cropped 18×32 patch.
# No 640×360 render in sim2sim (reduces load; training in Isaac may use higher-res sensors).
_RAW_DEPTH_H, _RAW_DEPTH_W = 36, 64
_FOV_X_DEG = 89.51  # nominal horizontal FOV in Isaac cfg (must match aspect × fovy)
_FOV_Y_DEG = 58.29  # vertical FOV (MuJoCo free camera + frustum)
_ENCODER_H, _ENCODER_W = 18, 32
_CROP_REGION = (18, 0, 16, 16)  # on 64×36 grid; Isaac CropAndResizeCfg.crop_region
_DEPTH_CLIP = (0.0, 2.5)
# Must match delayed_visualizable_image: sensor_history_length=37, history_skip=5, delay=0.
# Encoder frame count is cfg.robot_config.depth_encoder_frames (indices still subsampled from this ring).
_DEPTH_HISTORY_LEN = 37
_DEPTH_HISTORY_SKIP = 5
_DEPTH_DELAY = 0
_OBS_HISTORY_KEYS: tuple[str, ...] = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
# Extrinsic: RP1ParkourEnvCfg scene.camera.offset on waist_yaw_link (world: x + R_waist @ pos).
_CAMERA_OFFSET_POS_BODY = np.array([0.09175, 0.01, 0.3982], dtype=np.float64)
# Local MuJoCo helper uses WXYZ; Isaac Lab 3.0 camera config uses XYZW.
_CAMERA_OFFSET_QUAT_WXYZ = np.array([0.866, 0.0, 0.5, 0.0], dtype=np.float64)
# Depth clip: MJCF <map znear> × extent; keep small to avoid near-field stair tread culling.
_DEPTH_ZNEAR_SCALE = 0.001
_DEPTH_ZFAR_SCALE = 50.0
# "ray" = mj_ray (Isaac GroupedRayCaster parity); "renderer" = MuJoCo OpenGL depth.
_DEPTH_BACKEND = "ray"
_DEPTH_RAY_MAX = 2.5

# Performance-oriented defaults (formerly CLI flags): less OpenCV/wall-clock sleep/terminal churn; edit here.
_SIM2SIM_PERF_DEPTH_VIS_SCALE = 6
_SIM2SIM_PERF_DEBUG_OBS = False
_SIM2SIM_PERF_ONNX_PROVIDERS: list[str] | None = None
_SIM2SIM_PERF_REALTIME_SYNC = True
_SIM2SIM_PERF_QUIET = True
_SIM2SIM_PERF_DEPTH_VIS_EVERY_STEP = False
_SIM2SIM_PERF_DEPTH_VIS_POLICY_STRIDE = 2
_SIM2SIM_PERF_VIEWER_SYNC_EVERY: int | None = None
_SIM2SIM_PERF_VIEWER_FALLBACK_W = 1920
_SIM2SIM_PERF_VIEWER_FALLBACK_H = 1080
# Passive MuJoCo viewer: overlay cmd vs base-frame velocities (``get_obs`` / policy frame).
_SIM2SIM_PERF_SHOW_VELOCITY_OVERLAY = True
_SIM2SIM_PERF_ONNX_PROVIDERS = ["CPUExecutionProvider"]
# Chase camera (key ``r``): behind-above base_link, rotates with robot heading.
_CHASE_BACK_M = 2.0
_CHASE_UP_M = 0.6
_CHASE_LOOK_AHEAD_M = 0.8
_CHASE_BODY_NAME = "base_link"
# Default orbit camera: front-oblique 45° (azimuth 135). Rear-oblique was 45.
_CAM_AZIMUTH_DEG = 135.0
_CAM_ELEVATION_DEG = -20.0
_CAM_DISTANCE_M = 4.0
# Deployment velocity limits (min, max) per axis: vx, vy, dyaw.
_CLIP_CMD_X = (-0.5, 0.6)
_CLIP_CMD_Y = (-0.6, 0.6)
_CLIP_CMD_Z = (-1.5, 1.5)


class CompactOverlayMujocoViewer(mujoco_viewer.MujocoViewer):
    """MujocoViewer with compact overlay and MuJoCo 3.3+ double-click select fix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_200.value)
        self._velocity_table = None

    def set_velocity_table(self, cmd_values, vel_values):
        self._velocity_table = (tuple(float(v) for v in cmd_values), tuple(float(v) for v in vel_values))

    def _create_overlay(self):
        super()._create_overlay()
        self._overlay.pop(mujoco.mjtGridPos.mjGRID_TOPLEFT, None)
        self._overlay.pop(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, None)

    def _draw_velocity_table(self):
        if self._velocity_table is None:
            return

        cmd_values, vel_values = self._velocity_table
        label_x = 0.365
        col_sign_x = (0.455, 0.535, 0.615)
        col_value_x = (0.477, 0.557, 0.637)
        header_x = (0.475, 0.555, 0.635)
        header_y = 0.950
        cmd_y = 0.910
        vel_y = 0.870

        def draw(text, x, y):
            mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, text, self.ctx, x, y, 0.0, 0.0, 0.0)
            mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, text, self.ctx, x + 0.001, y, 0.0, 0.0, 0.0)

        for axis, x in zip(("x", "y", "z"), header_x):
            draw(axis, x, header_y)
        draw("cmd", label_x, cmd_y)
        draw("vel", label_x, vel_y)

        for row_y, values in ((cmd_y, cmd_values), (vel_y, vel_values)):
            for sign_x, value_x, value in zip(col_sign_x, col_value_x, values):
                draw("+" if value >= 0.0 else "-", sign_x, row_y)
                draw(f"{abs(value):.2f}", value_x, row_y)

    def _mouse_button_callback(self, window, button, act, mods):
        """Same as mujoco_viewer, but skinselect must be int (MuJoCo 3.3+)."""
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()
            time_diff = time_now - self._last_left_click_time
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()
            time_diff = time_now - self._last_right_click_time
            if time_diff > 0.01 and time_diff < 0.2:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

        if self._left_double_click_pressed or self._right_double_click_pressed:
            selmode = 0
            if self._left_double_click_pressed:
                selmode = 1
            if self._right_double_click_pressed:
                selmode = 2
            if self._right_double_click_pressed and key:
                selmode = 3

            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selflex = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)

            selbody = mujoco.mjv_select(
                self.model,
                self.data,
                self.vopt,
                aspectratio,
                relx,
                rely,
                self.scn,
                selpnt,
                selgeom,
                selflex,
                selskin,
            )

            if selmode == 2 or selmode == 3:
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                if selmode == 3 and selbody > 0:
                    self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            else:
                if selbody >= 0:
                    self.pert.select = int(selbody)
                    # mjv_select writes into ndarray out-args; MjvPerturb fields need scalars.
                    self.pert.skinselect = int(np.asarray(selskin).flat[0])
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    self.pert.localpos = self.data.xmat[selbody].reshape(3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            self.pert.active = 0

        if act == glfw.RELEASE:
            self.pert.active = 0

    def render(self):
        if self.render_mode == "offscreen":
            raise NotImplementedError("Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        def update():
            self._create_overlay()
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                for marker in self._markers:
                    self._add_marker_to_scene(marker)

                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                for gridpos, (t1, t2) in self._overlay.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_200,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx,
                    )
                self._draw_velocity_table()

                if not self._hide_graph:
                    for idx, fig in enumerate(self.figs):
                        width_adjustment = width % 4
                        x = int(3 * width / 4) + width_adjustment
                        y = idx * int(height / 4)
                        viewport = mujoco.MjrRect(x, y, int(width / 4), int(height / 4))
                        has_lines = len([i for i in fig.linename if i != b""])
                        if has_lines:
                            mujoco.mjr_figure(viewport, fig, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        self._markers[:] = []
        self.apply_perturbations()


def mj_macro_step(model: mujoco.MjModel, data: mujoco.MjData, *, n_sub: int) -> None:
    """Advance simulation by one *policy* sub-step duration (``cfg.dt``) using ``n_sub`` MuJoCo steps."""
    for _ in range(max(1, n_sub)):
        mujoco.mj_step(model, data)


def open_interactive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fallback_width: int = 1920,
    fallback_height: int = 1080,
    show_left_ui: bool | None = False,
    show_right_ui: bool | None = False,
) -> tuple[Any, bool, Any]:
    """Open the same interactive MuJoCo viewer path used by the AMP sim2sim script."""
    v = CompactOverlayMujocoViewer(
        model, data, mode="window", width=int(fallback_width), height=int(fallback_height)
    )
    v.cam.distance = _CAM_DISTANCE_M
    v.cam.azimuth = _CAM_AZIMUTH_DEG
    v.cam.elevation = _CAM_ELEVATION_DEG
    v.cam.lookat = [0, 0, 1]
    return v, False, None


def passive_viewer_velocity_overlay(
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cmd_vx: float,
    cmd_vy: float,
    cmd_wz: float,
) -> None:
    """Show keyboard cmd vs measured base velocity in the active viewer (base link frame, same as obs).

    Do **not** call ``viewer.lock()`` here: the sim loop can outrun the GUI; holding the lock every
    sub-step freezes the viewer. Call only on the same cadence as viewer render/sync on the main thread.
    """
    try:
        _, _, _, v, omega, _ = get_obs(data, model)
        if hasattr(viewer, "set_velocity_table"):
            viewer.set_velocity_table((cmd_vx, cmd_vy, cmd_wz), (v[0], v[1], omega[2]))
        elif hasattr(viewer, "set_texts"):
            left_col = "\ncmd\nvel"
            right_col = (
                "   x       y       z\n"
                f"{cmd_vx:+.2f}   {cmd_vy:+.2f}   {cmd_wz:+.2f}\n"
                f"{v[0]:+.2f}   {v[1]:+.2f}   {omega[2]:+.2f}"
            )
            viewer.set_texts((None, None, left_col, right_col))
    except Exception:
        pass


def close_interactive_viewer(viewer: Any, use_passive: bool, passive_ctx: Any) -> None:
    if use_passive and passive_ctx is not None:
        passive_ctx.__exit__(None, None, None)
    elif viewer is not None:
        viewer.close()



class CameraMode:
    ORBIT = "orbit"
    CHASE = "chase"


class cmd:
    """Keyboard sets **target** velocity; published commands **ramp** toward that target each policy step."""

    _MOVE_KEYS: ClassVar[frozenset[str]] = frozenset({"8", "2", "4", "6", "7", "9"})
    _pressed: ClassVar[set[str]] = set()

    # Magnitude while the corresponding key is held (opposing keys cancel); matches deployment clip limits.
    hold_vx_forward = float(_CLIP_CMD_X[1])
    hold_vx_back = float(-_CLIP_CMD_X[0])
    hold_vy = float(_CLIP_CMD_Y[1])
    hold_dyaw = float(_CLIP_CMD_Z[1])

    # Max |d(command)/dt| when moving toward keyboard target (linear ramp; units/s).
    ramp_vx_per_s = 2.0
    ramp_vy_per_s = 2.0
    ramp_dyaw_per_s = 3.0

    _smooth_vx: ClassVar[float] = 0.0
    _smooth_vy: ClassVar[float] = 0.0
    _smooth_dyaw: ClassVar[float] = 0.0
    # If set, keyboard is ignored and this (vx, vy, dyaw) is the ramp target.
    scripted_target: ClassVar[tuple[float, float, float] | None] = None
    # Piecewise (t_end, vx, vy, dyaw); overrides scripted_target when set.
    scripted_schedule: ClassVar[list[tuple[float, float, float, float]] | None] = None
    sim_time: ClassVar[float] = 0.0

    camera_follow = True
    camera_mode: ClassVar[str] = CameraMode.ORBIT
    reset_requested = False

    @staticmethod
    def _step_axis(current: float, target: float, rate_per_s: float, dt: float) -> float:
        if rate_per_s <= 0.0:
            return target
        max_step = float(rate_per_s) * float(dt)
        err = target - current
        if abs(err) <= max_step:
            return target
        return current + np.sign(err) * max_step

    @classmethod
    def _scripted_cmd(cls) -> tuple[float, float, float] | None:
        if cls.scripted_schedule:
            for t_end, vx, vy, dyaw in cls.scripted_schedule:
                if cls.sim_time < t_end:
                    return float(vx), float(vy), float(dyaw)
            _, vx, vy, dyaw = cls.scripted_schedule[-1]
            return float(vx), float(vy), float(dyaw)
        if cls.scripted_target is not None:
            return cls.scripted_target
        return None

    @classmethod
    def _keyboard_target(cls) -> tuple[float, float, float]:
        scripted = cls._scripted_cmd()
        if scripted is not None:
            return scripted
        vx = vy = dyaw = 0.0
        p = cls._pressed
        if "8" in p:
            vx += cls.hold_vx_forward
        if "2" in p:
            vx -= cls.hold_vx_back
        if "4" in p:
            vy += cls.hold_vy
        if "6" in p:
            vy -= cls.hold_vy
        if "7" in p:
            dyaw += cls.hold_dyaw
        if "9" in p:
            dyaw -= cls.hold_dyaw
        return vx, vy, dyaw

    @classmethod
    def step_command_velocity(cls, dt_policy: float) -> tuple[float, float, float]:
        """Advance smoothed velocity command one policy interval toward the keyboard target."""
        tgt_vx, tgt_vy, tgt_dyaw = cls._keyboard_target()
        cls._smooth_vx = cls._step_axis(cls._smooth_vx, tgt_vx, cls.ramp_vx_per_s, dt_policy)
        cls._smooth_vy = cls._step_axis(cls._smooth_vy, tgt_vy, cls.ramp_vy_per_s, dt_policy)
        cls._smooth_dyaw = cls._step_axis(cls._smooth_dyaw, tgt_dyaw, cls.ramp_dyaw_per_s, dt_policy)
        if cls._scripted_cmd() is None:
            vx = float(np.clip(cls._smooth_vx, *_CLIP_CMD_X))
            vy = float(np.clip(cls._smooth_vy, *_CLIP_CMD_Y))
            dyaw = float(np.clip(cls._smooth_dyaw, *_CLIP_CMD_Z))
        else:
            vx, vy, dyaw = float(cls._smooth_vx), float(cls._smooth_vy), float(cls._smooth_dyaw)
        cls._smooth_vx, cls._smooth_vy, cls._smooth_dyaw = vx, vy, dyaw
        return vx, vy, dyaw

    @classmethod
    def toggle_camera_follow(cls) -> None:
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def toggle_chase_camera(cls) -> None:
        if cls.camera_mode == CameraMode.CHASE:
            cls.camera_mode = CameraMode.ORBIT
            print("Camera mode: orbit follow")
        else:
            cls.camera_mode = CameraMode.CHASE
            cls.camera_follow = True
            print("Camera mode: chase (behind-above)")

    @classmethod
    def reset(cls) -> None:
        cls._pressed.clear()
        cls._smooth_vx = 0.0
        cls._smooth_vy = 0.0
        cls._smooth_dyaw = 0.0


def print_controls_guide() -> None:
    """Print keyboard controls once at startup (always shown, independent of ``quiet``)."""
    rows: list[tuple[str, str, str]] = [
        ("8 / 2", "Forward / Back", f"Target vx +{cmd.hold_vx_forward}/-{cmd.hold_vx_back} m/s; ramps down on release"),
        ("7 / 9", "Turn left / right", f"Target dyaw ±{cmd.hold_dyaw} rad/s; ramps down on release"),
        (
            "R",
            "Toggle chase camera",
            "Behind-above third-person follow; press again for orbit view",
        ),
        (
            "F",
            "Toggle camera follow",
            f"Orbit: fixed angle, lookat only; chase: full pose (default: {'on' if cmd.camera_follow else 'off'})",
        ),
        ("0", "Reset environment", "Restore pose, obs history, depth buffer, and actions"),
        ("Close MuJoCo window", "Quit simulation", ""),
    ]
    if cmd.hold_vy > 0:
        rows.insert(
            1,
            ("4 / 6", "Strafe left / right", f"Target vy ±{cmd.hold_vy} m/s; ramps down on release"),
        )
    headers = ("Keys", "Action", "Notes")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(3)
    ]

    def _fmt_row(cells: tuple[str, str, str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep_inner = "-+-".join("-" * w for w in widths)
    border = "=" * (len(sep_inner) + 4)
    print(border)
    print("Keyboard controls (focus terminal or sim window)")
    print(f"+-{'-+-'.join('-' * w for w in widths)}-+")
    print(f"| {_fmt_row(headers)} |")
    print(f"+-{sep_inner}-+")
    for row in rows:
        print(f"| {_fmt_row(row)} |")
    print(f"+-{sep_inner}-+")
    print(border)


def on_press(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()
            if c in cmd._MOVE_KEYS:
                cmd._pressed.add(c)
            elif c == "f":
                cmd.toggle_camera_follow()
            elif c == "r":
                cmd.toggle_chase_camera()
            elif c == "0":
                cmd.reset_requested = True
    except AttributeError:
        pass


def on_release(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()
            if c in cmd._MOVE_KEYS:
                cmd._pressed.discard(c)
    except AttributeError:
        pass


class _NoopKeyboardListener:
    def stop(self) -> None:
        return None


def start_keyboard_listener(enabled: bool = True):
    if not enabled:
        return _NoopKeyboardListener()
    try:
        from pynput import keyboard
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        return listener
    except Exception as exc:
        print(f"[WARN] Keyboard listener unavailable ({exc}); continuing without it.")
        return _NoopKeyboardListener()


def get_obs(data, model):
    """Articulation observation from MuJoCo (matches sim2sim_rpo_amp pattern)."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, target_dq, dq, kp, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _rot_mat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Isaac Lab stores quaternions as (w, x, y, z) on cfg; scipy ``from_quat`` expects (x, y, z, w)."""
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def effective_depth_clip_planes(model: mujoco.MjModel) -> tuple[float, float]:
    """Near/far clip distances used by MuJoCo depth (znear/zfar × stat.extent)."""
    extent = max(float(model.stat.extent), 1e-6)
    znear_scale = float(getattr(model.vis.map, "znear", _DEPTH_ZNEAR_SCALE))
    zfar_attr = float(getattr(model.vis.map, "zfar", 0.0))
    zfar_scale = zfar_attr if zfar_attr > 0.0 else _DEPTH_ZFAR_SCALE
    near = znear_scale * extent
    far = max(zfar_scale * extent, near + 1e-3)
    return near, far


def configure_depth_rendering(model: mujoco.MjModel) -> None:
    """Tighten near clip for offscreen depth (stairs within ~0.1–2.5 m of camera)."""
    model.vis.map.znear = float(_DEPTH_ZNEAR_SCALE)
    if hasattr(model.vis.map, "zfar"):
        model.vis.map.zfar = float(_DEPTH_ZFAR_SCALE)


def get_depth_camera_pose(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
) -> dict[str, np.ndarray]:
    """Torso-mounted camera pose (Isaac offset); +X forward, +Z up in camera frame."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {torso_body_name}")

    xmat = data.xmat[bid].reshape(3, 3)
    xpos = data.xpos[bid].copy()
    r_off = _rot_mat_wxyz(_CAMERA_OFFSET_QUAT_WXYZ)
    r_world = (xmat @ r_off).astype(np.float64)
    eye = (xpos + xmat @ _CAMERA_OFFSET_POS_BODY).astype(np.float64)
    forward = r_world[:, 0].copy()
    fn = np.linalg.norm(forward)
    if fn < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward /= fn
    return {"eye": eye, "forward": forward, "r_world": r_world}


def pinhole_ray_dirs_cam(
    height: int,
    width: int,
    fovy_deg: float,
    fovx_deg: float,
) -> np.ndarray:
    """Unit ray directions in camera frame (+X forward, +Z up, +Y left). Shape (H, W, 3)."""
    fovy = np.radians(fovy_deg)
    fovx = np.radians(fovx_deg)
    u = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    v = (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0
    uu, vv = np.meshgrid(u, v)
    x = np.ones_like(uu)
    y = -uu * np.tan(fovx / 2.0)
    z = -vv * np.tan(fovy / 2.0)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def to_distance_to_image_plane(radial_dist: np.ndarray, dirs_cam: np.ndarray) -> np.ndarray:
    """Project metric distance along each ray onto camera +X (Isaac distance_to_image_plane)."""
    return radial_dist * dirs_cam[..., 0]


def _set_free_camera_eye_lookat(
    cam: mujoco.MjvCamera,
    eye: np.ndarray,
    lookat: np.ndarray,
) -> None:
    """Map world-space eye/lookat to MuJoCo FREE camera spherical parameters."""
    view_dir = lookat - eye
    dist = float(np.linalg.norm(view_dir))
    if dist < 1e-6:
        dist = 1e-6
        view_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        view_dir = (view_dir / dist).astype(np.float64)

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = float(np.degrees(np.arctan2(view_dir[1], view_dir[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(view_dir[2], -1.0, 1.0))))


def _horizontal_body_forward(xmat: np.ndarray) -> np.ndarray:
    """Body +X projected onto the world XY plane (yaw only, ignores roll/pitch)."""
    forward = xmat[:, 0].copy()
    forward[2] = 0.0
    fn = float(np.linalg.norm(forward))
    if fn < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (forward / fn).astype(np.float64)


def apply_chase_camera(
    cam: mujoco.MjvCamera,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    body_name: str = _CHASE_BODY_NAME,
    back_m: float = _CHASE_BACK_M,
    up_m: float = _CHASE_UP_M,
    look_ahead_m: float = _CHASE_LOOK_AHEAD_M,
) -> None:
    """Third-person chase cam behind ``body_name``; yaw follows robot, horizon stays level."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {body_name}")

    xpos = data.xpos[bid].copy()
    forward_h = _horizontal_body_forward(data.xmat[bid].reshape(3, 3))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = xpos - back_m * forward_h + up_m * world_up
    lookat = xpos + look_ahead_m * forward_h
    _set_free_camera_eye_lookat(cam, eye, lookat)


def update_follow_camera(cam: mujoco.MjvCamera, data: mujoco.MjData, model: mujoco.MjModel) -> None:
    """Apply orbit (lookat-only) or chase (full pose) follow to an ``MjvCamera``."""
    if cmd.camera_mode == CameraMode.CHASE:
        apply_chase_camera(cam, data, model)
    else:
        cam.lookat[:] = data.qpos[0:3].astype(np.float64)


def set_depth_camera_free(
    cam: mujoco.MjvCamera,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
) -> None:
    """Place FREE camera at torso_link with Isaac offset; optical axis matches IsaacLab ray caster."""
    pose = get_depth_camera_pose(data, model, torso_body_name)
    eye = pose["eye"]
    lookat = eye + pose["forward"]
    _set_free_camera_eye_lookat(cam, eye, lookat)
    if hasattr(cam, "fovy"):
        cam.fovy = vertical_fov_deg


def capture_depth_mj_ray(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
    horizontal_fov_deg: float,
) -> np.ndarray:
    """64×36 depth via mj_ray (first hit), Isaac ``distance_to_image_plane`` semantics."""
    pose = get_depth_camera_pose(data, model, torso_body_name)
    eye = pose["eye"]
    r_world = pose["r_world"]
    dirs_cam = pinhole_ray_dirs_cam(_RAW_DEPTH_H, _RAW_DEPTH_W, vertical_fov_deg, horizontal_fov_deg)
    dirs_world = dirs_cam @ r_world.T

    depth = np.full((_RAW_DEPTH_H, _RAW_DEPTH_W), _DEPTH_RAY_MAX, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    for j in range(_RAW_DEPTH_H):
        for i in range(_RAW_DEPTH_W):
            vec = dirs_world[j, i]
            dist = mujoco.mj_ray(model, data, eye, vec, None, 1, -1, geomid)
            if dist is not None and dist >= 0.0:
                depth[j, i] = min(float(dist) * float(dirs_cam[j, i, 0]), _DEPTH_RAY_MAX)
    return depth


def capture_depth_renderer(
    depth_renderer: Any,
    depth_cam: mujoco.MjvCamera,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
    horizontal_fov_deg: float,
) -> np.ndarray:
    """OpenGL depth → linearize → distance_to_image_plane."""
    set_depth_camera_free(depth_cam, data, model, torso_body_name, vertical_fov_deg)
    depth_renderer.update_scene(data, camera=depth_cam)
    if hasattr(depth_renderer, "enable_depth_rendering"):
        raw = np.asarray(depth_renderer.render(), dtype=np.float64)
    else:
        rgb_or_depth = depth_renderer.render()
        if hasattr(depth_renderer, "depth"):
            raw = np.array(depth_renderer.depth, copy=True, dtype=np.float64)
        elif isinstance(rgb_or_depth, np.ndarray) and rgb_or_depth.ndim == 2:
            raw = rgb_or_depth.astype(np.float64)
        else:
            raise RuntimeError(
                "Cannot read depth: use MuJoCo enable_depth_rendering + render() or a Renderer with .depth."
            )
    radial = linearize_depth_buffer(raw, model)
    dirs_cam = pinhole_ray_dirs_cam(_RAW_DEPTH_H, _RAW_DEPTH_W, vertical_fov_deg, horizontal_fov_deg)
    return to_distance_to_image_plane(radial, dirs_cam)


def capture_raycaster_depth(
    depth_renderer: Any | None,
    depth_cam: mujoco.MjvCamera | None,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
    horizontal_fov_deg: float,
) -> np.ndarray:
    """Policy depth frame: metric distance_to_image_plane, shape (H_raw, W_raw)."""
    if _DEPTH_BACKEND == "ray":
        return capture_depth_mj_ray(data, model, torso_body_name, vertical_fov_deg, horizontal_fov_deg)
    if depth_renderer is None or depth_cam is None:
        raise RuntimeError("renderer backend requires depth_renderer and depth_cam")
    return capture_depth_renderer(
        depth_renderer, depth_cam, data, model, torso_body_name, vertical_fov_deg, horizontal_fov_deg
    )


def linearize_depth_buffer(depth: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Map OpenGL depth buffer values to eye-space Z (meters) when non-linear."""
    d = depth.astype(np.float64)
    if d.size == 0:
        return d
    # Heuristic: MuJoCo >=3 often returns distances; if mostly in (0,1), treat as NDC depth.
    if np.nanmax(d) <= 1.0 + 1e-3 and np.nanmin(d) >= 0.0:
        near, far = effective_depth_clip_planes(model)
        z_ndc = d * 2.0 - 1.0
        z_eye = (2.0 * near * far) / (far + near - z_ndc * (far - near))
        return np.maximum(z_eye, 0.0)
    return np.abs(d)


def raw_depth_to_metric_grid(depth_hw: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Linearize renderer depth to metric Z (m). If shape ≠ policy grid, resize with INTER_AREA to 64×36."""
    z = linearize_depth_buffer(depth_hw.astype(np.float64), model)
    if z.shape[0] != _RAW_DEPTH_H or z.shape[1] != _RAW_DEPTH_W:
        z = cv2.resize(z, (_RAW_DEPTH_W, _RAW_DEPTH_H), interpolation=cv2.INTER_AREA)
    return z


def crop_depth(depth_hw: np.ndarray, crop_region: tuple[int, int, int, int]) -> np.ndarray:
    """Crop (H, W) depth; indices match ``crop_and_resize`` in ``noise_model.py`` (H=axis0, W=axis1)."""
    up, down, left, right = crop_region
    h, w = depth_hw.shape[:2]
    return depth_hw[up : h - down, left : w - right].copy()


def _metric_depth_nan_safe(z_hw: np.ndarray) -> np.ndarray:
    """Replace non-finite values for OpenCV preview (clip range unchanged)."""
    return np.nan_to_num(z_hw, nan=0.0, posinf=0.0, neginf=0.0)


def _draw_encoder_crop_roi_on_bgr(bgr: np.ndarray, scale: int) -> None:
    """Red rectangle = encoder crop on the 64×36 grid, after `bgr` has been upscaled by `scale`."""
    up, down, left, right = _CROP_REGION
    h, w = _RAW_DEPTH_H, _RAW_DEPTH_W
    fac = float(scale)
    x1 = int(round(left * fac))
    y1 = int(round(up * fac))
    x2 = int(round((w - right) * fac)) - 1
    y2 = int(round((h - down) * fac)) - 1
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), max(1, scale // 2))


def preprocess_depth_frame_from_lowres(z_low: np.ndarray) -> np.ndarray:
    """64×36 metric (native render grid) → crop → blur → clip → [0, 1] (encoder one frame)."""
    z = np.nan_to_num(z_low, nan=0.0, posinf=0.0, neginf=0.0)
    z = crop_depth(z, _CROP_REGION)
    if z.shape != (_ENCODER_H, _ENCODER_W):
        z = cv2.resize(z, (_ENCODER_W, _ENCODER_H), interpolation=cv2.INTER_AREA)
    z = cv2.GaussianBlur(z.astype(np.float64), (3, 3), 1.0)
    z = np.clip(z, _DEPTH_CLIP[0], _DEPTH_CLIP[1])
    z = (z - _DEPTH_CLIP[0]) / (_DEPTH_CLIP[1] - _DEPTH_CLIP[0])
    return z.astype(np.float32)


def show_depth_camera_side_by_side(
    metric_lowres_hw: np.ndarray,
    policy_normalized_hw: np.ndarray,
    window_name: str,
    scale: int,
) -> None:
    """One window: left = metric 64×36 + crop ROI; right = policy patch [0,1], same pixel height."""
    if metric_lowres_hw.shape != (_RAW_DEPTH_H, _RAW_DEPTH_W):
        raise ValueError(
            f"metric depth expects {_RAW_DEPTH_H}x{_RAW_DEPTH_W}, got {metric_lowres_hw.shape}"
        )
    tw, th = _RAW_DEPTH_W * scale, _RAW_DEPTH_H * scale

    p = np.clip(policy_normalized_hw, 0.0, 1.0)
    enc_bgr = cv2.cvtColor((p * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    enc_bgr = cv2.resize(enc_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)

    d = np.clip(metric_lowres_hw, _DEPTH_CLIP[0], _DEPTH_CLIP[1])
    metric_bgr = cv2.cvtColor((d / _DEPTH_CLIP[1] * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    metric_bgr = cv2.resize(metric_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)
    _draw_encoder_crop_roi_on_bgr(metric_bgr, scale)

    cv2.imshow(window_name, np.hstack([metric_bgr, enc_bgr]))
    cv2.waitKey(1)


def preprocess_depth_frame(depth_hw: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Native renderer buffer → 64×36 metric (via ``raw_depth_to_metric_grid``) → crop / blur / norm."""
    z_low = raw_depth_to_metric_grid(depth_hw, model)
    return preprocess_depth_frame_from_lowres(z_low)


def depth_frame_indices(num_output_frames: int) -> np.ndarray:
    """Subsample indices into the 37-frame ring (oldest → newest), same as delayed_visualizable_image."""
    n = max(1, int(num_output_frames))
    frames_needed = (n - 1) * _DEPTH_HISTORY_SKIP + 1 + _DEPTH_DELAY
    if frames_needed > _DEPTH_HISTORY_LEN:
        raise ValueError(
            f"depth_encoder_frames={n} needs {frames_needed} slots, but buffer is {_DEPTH_HISTORY_LEN}"
        )
    # frame_offset largest→smallest (oldest first); index = history_len - offset - delay - 1
    offsets = np.arange(0, n * _DEPTH_HISTORY_SKIP, _DEPTH_HISTORY_SKIP)[::-1]
    return (_DEPTH_HISTORY_LEN - offsets - _DEPTH_DELAY - 1).astype(np.int64)


def sample_depth_history(depth_buf37: list[np.ndarray], num_frames: int) -> np.ndarray:
    """Stack ``num_frames`` subsampled from the 37-frame buffer (oldest → newest), shape (N, H, W)."""
    indices = depth_frame_indices(num_frames)
    n = len(indices)
    if not depth_buf37:
        return np.zeros((n, _ENCODER_H, _ENCODER_W), dtype=np.float32)
    buf = list(depth_buf37)
    if len(buf) < _DEPTH_HISTORY_LEN:
        buf = [buf[0]] * (_DEPTH_HISTORY_LEN - len(buf)) + buf
    stacked = np.stack(buf[-_DEPTH_HISTORY_LEN:], axis=0)
    return stacked[indices].astype(np.float32)


class TermHistory:
    """Isaac CircularBuffer semantics: maxlen ring, iteration oldest->newest."""

    def __init__(self, max_len: int, term_dim: int):
        self.max_len = max_len
        self.term_dim = term_dim
        self._dq: deque[np.ndarray] = deque(maxlen=max_len)

    def reset(self) -> None:
        self._dq.clear()

    def append(self, x: np.ndarray) -> None:
        self._dq.append(np.asarray(x, dtype=np.float32).reshape(-1))

    def fill_tile(self, x: np.ndarray) -> None:
        self.reset()
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        for _ in range(self.max_len):
            self._dq.append(v.copy())

    def flat(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.zeros(self.max_len * self.term_dim, dtype=np.float32)
        return np.concatenate(list(self._dq), axis=0)


def build_onnx_sessions(
    depth_encoder_path: str, actor_path: str, providers: list[str] | None = None
) -> tuple[Any, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is required for this script") from _ORT_IMPORT_ERROR
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = providers or ort.get_available_providers()
    enc = ort.InferenceSession(depth_encoder_path, so, providers=prov)
    act = ort.InferenceSession(actor_path, so, providers=prov)
    return enc, act


def run_mujoco_onnx(
    depth_encoder: Any,
    actor: Any,
    cfg,
    headless: bool = False,
    debug_obs: bool = False,
    show_depth_vis: bool = True,
    depth_vis_scale: int = 8,
    *,
    realtime_sync: bool = True,
    quiet: bool = False,
    depth_vis_every_step: bool = False,
    viewer_sync_every: int | None = None,
    depth_vis_policy_stride: int = 1,
    viewer_fallback_width: int = 1280,
    viewer_fallback_height: int = 720,
    mujoco_full_ui: bool = False,
    record_video: bool = True,
    video_path: str = "simulation_parkour.mp4",
    save_plots: bool = True,
) -> None:
    enc_in_name = depth_encoder.get_inputs()[0].name
    act_in_name = actor.get_inputs()[0].name

    viewer_stride = max(
        1,
        viewer_sync_every if viewer_sync_every is not None else int(cfg.sim_config.decimation),
    )
    if headless:
        show_depth_vis = False
    keyboard_listener = start_keyboard_listener(enabled=not headless)
    print_controls_guide()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    configure_depth_rendering(model)
    xml_dt = float(model.opt.timestep)
    macro_dt = float(cfg.sim_config.dt)
    phys_substeps = max(1, int(round(macro_dt / xml_dt)))
    model.opt.timestep = macro_dt / float(phys_substeps)
    # Mesh–box contacts (feet vs stairs) penetrate much more than mesh–plane at a single large mj_step.
    model.opt.iterations = max(int(model.opt.iterations), 150)
    try:
        model.vis.global_.fovy = float(_FOV_Y_DEG)
    except Exception:
        pass
    ax = float(_RAW_DEPTH_W) / float(_RAW_DEPTH_H)
    fovx_implied = float(np.degrees(2.0 * np.arctan(np.tan(np.radians(_FOV_Y_DEG) / 2.0) * ax)))
    if abs(fovx_implied - _FOV_X_DEG) > 1.0:
        print(
            f"[WARN] Viewport {_RAW_DEPTH_W}×{_RAW_DEPTH_H} with fovy={_FOV_Y_DEG}° implies "
            f"fovx≈{fovx_implied:.2f}°, Isaac cfg uses {_FOV_X_DEG}° — check camera FOV consistency."
        )
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions :] = cfg.robot_config.default_pos
    mj_macro_step(model, data, n_sub=phys_substeps)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    depth_renderer: mujoco.Renderer | None = None
    depth_cam: mujoco.MjvCamera | None = None
    if _DEPTH_BACKEND == "renderer":
        depth_renderer = mujoco.Renderer(model, height=_RAW_DEPTH_H, width=_RAW_DEPTH_W)
        if hasattr(depth_renderer, "enable_depth_rendering"):
            depth_renderer.enable_depth_rendering()
        elif hasattr(depth_renderer, "enable_depth"):
            depth_renderer.enable_depth = True
        depth_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(depth_cam)
    vertical_fov_deg = _FOV_Y_DEG
    horizontal_fov_deg = fovx_implied

    target_pos = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)

    frame_stack = int(cfg.robot_config.frame_stack)
    hist = {
        "base_ang_vel": TermHistory(frame_stack, 3),
        "projected_gravity": TermHistory(frame_stack, 3),
        "velocity_commands": TermHistory(frame_stack, 3),
        "joint_pos": TermHistory(frame_stack, cfg.robot_config.num_actions),
        "joint_vel": TermHistory(frame_stack, cfg.robot_config.num_actions),
        "actions": TermHistory(frame_stack, cfg.robot_config.num_actions),
    }
    depth_ring: deque[np.ndarray] = deque(maxlen=_DEPTH_HISTORY_LEN)
    is_first_policy = True

    count_lowlevel = 0
    policy_step_idx = 0
    tau = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)

    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = []
    actual_ang_vel_data = []
    foot_z_data = []
    left_foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    start_time = time.time()
    depth_vis_pending: tuple[np.ndarray, np.ndarray] | None = None

    if headless:
        renderer = None
        out = None
        cam_vid = None
        if record_video:
            renderer = mujoco.Renderer(model, width=1920, height=1080)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cam_vid = mujoco.MjvCamera()
            cam_vid.distance = _CAM_DISTANCE_M
            cam_vid.azimuth = _CAM_AZIMUTH_DEG
            cam_vid.elevation = _CAM_ELEVATION_DEG
            cam_vid.lookat = [0, 0, 1]
            video_dir = os.path.dirname(os.path.abspath(video_path))
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            out = cv2.VideoWriter(
                video_path,
                fourcc,
                1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
                (1920, 1080),
            )
            print(f"[INFO] Recording video to {video_path}")
        viewer = None
        use_passive_viewer = False
        passive_viewer_ctx = None
    else:
        viewer, use_passive_viewer, passive_viewer_ctx = open_interactive_viewer(
            model,
            data,
            fallback_width=viewer_fallback_width,
            fallback_height=viewer_fallback_height,
            show_left_ui=mujoco_full_ui,
            show_right_ui=mujoco_full_ui,
        )

    win_depth = f"Raw Depth | Cropped Depth"
    if not headless and show_depth_vis:
        cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    dt_policy = float(cfg.sim_config.dt) * float(cfg.sim_config.decimation)
    n_step = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    last_cmd_vx = 0.0
    last_cmd_vy = 0.0
    last_cmd_dyaw = 0.0
    for step in tqdm(range(n_step), desc="Simulating..."):
        cmd.sim_time = step * cfg.sim_config.dt
        if not headless and use_passive_viewer and not viewer.is_running():
            print("[INFO] Viewer closed; exiting simulation loop.")
            break

        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            cmd.reset()
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            action[:] = 0.0
            tau[:] = 0.0
            target_pos[:] = cfg.robot_config.default_pos.copy()
            count_lowlevel = 0
            for h in hist.values():
                h.reset()
            depth_ring.clear()
            is_first_policy = True
            cmd.reset_requested = False
            depth_vis_pending = None
            policy_step_idx = 0
            last_cmd_vx = last_cmd_vy = last_cmd_dyaw = 0.0

        q, dq, quat, v, omega, gvec = get_obs(data, model)
        q = q[-cfg.robot_config.num_actions :]
        dq = dq[-cfg.robot_config.num_actions :]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            q_ = q - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            bav = (omega.astype(np.float32) * 0.25).astype(np.float32)
            pg = gvec.astype(np.float32)
            vx_cmd, vy_cmd, dyaw_cmd = cmd.step_command_velocity(dt_policy)
            last_cmd_vx, last_cmd_vy, last_cmd_dyaw = float(vx_cmd), float(vy_cmd), float(dyaw_cmd)
            vc = np.array([vx_cmd, vy_cmd, dyaw_cmd], dtype=np.float32)
            jp = q_obs.astype(np.float32)
            jv = (dq_obs.astype(np.float32) * 0.05).astype(np.float32)
            act_last = action.astype(np.float32)
            vecs_policy = (bav, pg, vc, jp, jv, act_last)

            if is_first_policy:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].fill_tile(vec)
                is_first_policy = False
            else:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].append(vec)

            raw_depth = capture_raycaster_depth(
                depth_renderer,
                depth_cam,
                data,
                model,
                cfg.sim_config.depth_camera_body,
                vertical_fov_deg,
                horizontal_fov_deg,
            )
            z_depth_low = raw_depth_to_metric_grid(raw_depth, model)
            dproc = preprocess_depth_frame_from_lowres(z_depth_low)
            if not headless and show_depth_vis and not depth_vis_every_step:
                ds_vis = max(1, int(depth_vis_policy_stride))
                if policy_step_idx % ds_vis == 0:
                    depth_vis_pending = (
                        np.asarray(z_depth_low, dtype=np.float64).copy(),
                        np.asarray(dproc, dtype=np.float32).copy(),
                    )
                else:
                    depth_vis_pending = None
            depth_ring.append(dproc)
            depth_n = sample_depth_history(
                list(depth_ring), int(cfg.robot_config.depth_encoder_frames)
            )
            depth_bchw = depth_n[None, ...]

            flat_proprio = np.concatenate([hist[k].flat() for k in _OBS_HISTORY_KEYS], axis=0)

            latent = depth_encoder.run(None, {enc_in_name: depth_bchw})[0]
            actor_in = np.concatenate([flat_proprio, latent.reshape(-1)], axis=0)[None, :]
            if debug_obs:
                print(
                    f"[debug] flat_proprio={flat_proprio.shape} latent={latent.shape} actor_in={actor_in.shape}"
                )

            action[:] = actor.run(None, {act_in_name: actor_in.astype(np.float32)})[0].reshape(-1)

            target_q = action * cfg.robot_config.action_scale
            target_pos[:] = cfg.robot_config.default_pos.copy()
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] += target_q[i]

            q_low_freq = q.copy()
            v_low_freq = v[:2].copy()
            omega_low_freq = omega[2]

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq)
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(vx_cmd)
            commanded_lin_vel_y_data.append(vy_cmd)
            commanded_ang_vel_z_data.append(dyaw_cmd)
            actual_lin_vel_data.append(v_low_freq)
            actual_ang_vel_data.append(omega_low_freq)
            if left_foot_bid >= 0 and right_foot_bid >= 0:
                foot_z_data.append(
                    (float(data.xpos[left_foot_bid, 2]), float(data.xpos[right_foot_bid, 2]))
                )

            if not quiet:
                print(
                    "cmd vx,vy,dyaw={:.2f},{:.2f},{:.2f} | vxy,wz={:.2f},{:.2f},{:.2f}".format(
                        vx_cmd, vy_cmd, dyaw_cmd, v[0], v[1], omega[2]
                    )
                )

            if headless and renderer is not None and out is not None and cam_vid is not None:
                if cmd.camera_follow:
                    update_follow_camera(cam_vid, data, model)
                renderer.update_scene(data, camera=cam_vid)
                img = renderer.render()
                if img.shape[-1] == 4:
                    img = img[..., :3]
                out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            policy_step_idx += 1

        target_vel = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
        tau = pd_control(target_pos, q, target_vel, dq, cfg.robot_config.kps, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mj_macro_step(model, data, n_sub=phys_substeps)

        if not headless and show_depth_vis:
            z_disp: np.ndarray | None = None
            d_enc_vis: np.ndarray | None = None
            if depth_vis_every_step:
                raw_vis = capture_raycaster_depth(
                    depth_renderer,
                    depth_cam,
                    data,
                    model,
                    cfg.sim_config.depth_camera_body,
                    vertical_fov_deg,
                    horizontal_fov_deg,
                )
                z_low = raw_depth_to_metric_grid(raw_vis, model)
                d_enc_vis = preprocess_depth_frame_from_lowres(z_low)
                z_disp = _metric_depth_nan_safe(z_low)
            elif depth_vis_pending is not None:
                z_low, d_enc_vis = depth_vis_pending
                depth_vis_pending = None
                z_disp = _metric_depth_nan_safe(z_low)
            if z_disp is not None and d_enc_vis is not None:
                show_depth_camera_side_by_side(z_disp, d_enc_vis, win_depth, depth_vis_scale)

        if not headless:
            if cmd.camera_follow:
                update_follow_camera(viewer.cam, data, model)
            stride = viewer_stride
            if count_lowlevel % stride == 0:
                if _SIM2SIM_PERF_SHOW_VELOCITY_OVERLAY:
                    passive_viewer_velocity_overlay(
                        viewer, model, data, last_cmd_vx, last_cmd_vy, last_cmd_dyaw
                    )
                if use_passive_viewer:
                    viewer.sync()
                else:
                    viewer.render()

        count_lowlevel += 1

        if realtime_sync:
            elapsed_real_time = time.time() - start_time
            target_sim_time = (step + 1) * cfg.sim_config.dt
            if elapsed_real_time < target_sim_time:
                time.sleep(target_sim_time - elapsed_real_time)

    if depth_renderer is not None and hasattr(depth_renderer, "close"):
        depth_renderer.close()
    if headless:
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()
        if out is not None:
            out.release()
    else:
        if show_depth_vis:
            cv2.destroyAllWindows()
        close_interactive_viewer(viewer, use_passive_viewer, passive_viewer_ctx)
    keyboard_listener.stop()

    q_act = np.asarray(actual_joint_pos_data)
    t_act = np.asarray(time_data)
    if q_act.size:
        # URDF order: left_knee=3, right_knee=9. Straight knee ≈ 0 rad.
        lk = np.abs(q_act[:, 3])
        rk = np.abs(q_act[:, 9])
        wz = np.asarray(actual_ang_vel_data)
        dyaw = np.asarray(commanded_ang_vel_z_data)
        # URDF: L hip_pitch=0, L knee=3, L ankle_pitch=4; R hip=6, R knee=9, R ankle=10.
        # Flexion from default: left hip more negative, left knee more negative, right hip/knee more positive.
        dpos = np.asarray(cfg.robot_config.default_pos)
        lhp = -(q_act[:, 0] - dpos[0])
        rhp = q_act[:, 6] - dpos[6]
        lk_flex = -(q_act[:, 3] - dpos[3])
        rk_flex = q_act[:, 9] - dpos[9]
        lap = q_act[:, 4]
        rap = q_act[:, 10]
        fz = np.asarray(foot_z_data) if foot_z_data else None
        print("[turn joints] flexion from default (rad/deg); foot z; |wz|:")
        for t0, t1, name in ((0.0, 1.5, "0-1.5s"), (2.0, 4.0, "2-4s"), (4.0, 8.0, "4-8s")):
            m = (t_act >= t0) & (t_act < t1)
            if not np.any(m):
                continue
            def _deg(x):
                return float(np.degrees(x))
            msg = (
                f"  {name}: hip {lhp[m].mean():.2f}/{rhp[m].mean():.2f} rad "
                f"({_deg(lhp[m].mean()):.0f}/{_deg(rhp[m].mean()):.0f}°) p95="
                f"{_deg(max(np.percentile(lhp[m], 95), np.percentile(rhp[m], 95))):.0f}° | "
                f"knee {lk_flex[m].mean():.2f}/{rk_flex[m].mean():.2f} "
                f"({_deg(lk_flex[m].mean()):.0f}/{_deg(rk_flex[m].mean()):.0f}°) p95="
                f"{_deg(max(np.percentile(lk_flex[m], 95), np.percentile(rk_flex[m], 95))):.0f}° "
                f"peak={_deg(max(lk_flex[m].max(), rk_flex[m].max())):.0f}° | "
                f"|q_knee|={lk[m].mean():.3f}/{rk[m].mean():.3f} | "
                f"ank {lap[m].mean():.2f}/{rap[m].mean():.2f} | "
                f"|wz|={np.abs(wz[m]).mean():.3f}"
            )
            if fz is not None and len(fz) == len(t_act):
                zmax = np.maximum(fz[m, 0], fz[m, 1])
                msg += (
                    f" | foot_z mean={fz[m].mean():.3f} "
                    f"swing_max={zmax.max():.3f} p95={np.percentile(zmax, 95):.3f}"
                )
            print(msg)

    if not save_plots:
        if headless and out is not None:
            print(f"[INFO] Video saved: {video_path}")
        return

    # Plots (low frequency)
    time_data = np.asarray(time_data)
    commanded_joint_pos_data = np.asarray(commanded_joint_pos_data)
    actual_joint_pos_data = np.asarray(actual_joint_pos_data)
    num_joints = cfg.robot_config.num_actions
    n_cols = 4
    n_rows = (num_joints + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()
    for i in range(num_joints):
        ax = axes1[i]
        ax.plot(time_data, commanded_joint_pos_data[:, i], label="cmd", linestyle="--")
        ax.plot(time_data, actual_joint_pos_data[:, i], label="act")
        ax.set_title(f"Joint {i}")
        ax.grid(True)
        ax.legend()
    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])
    fig1.suptitle("Joint positions")
    plt.tight_layout()
    fig1.savefig("joint_positions_parkour.png")

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes2[0].plot(time_data, commanded_lin_vel_x_data, label="cmd vx", linestyle="--")
    axes2[0].plot(time_data, np.array(actual_lin_vel_data)[:, 0], label="vx")
    axes2[0].grid(True)
    axes2[0].legend()
    axes2[1].plot(time_data, commanded_lin_vel_y_data, label="cmd vy", linestyle="--")
    axes2[1].plot(time_data, np.array(actual_lin_vel_data)[:, 1], label="vy")
    axes2[1].grid(True)
    axes2[1].legend()
    axes2[2].plot(time_data, commanded_ang_vel_z_data, label="cmd dyaw", linestyle="--")
    axes2[2].plot(time_data, actual_ang_vel_data, label="wz")
    axes2[2].grid(True)
    axes2[2].legend()
    fig2.suptitle("Base velocity")
    plt.tight_layout()
    fig2.savefig("base_velocities_parkour.png")
    print("Saved joint_positions_parkour.png, base_velocities_parkour.png")


_FFMPEG_BIN = (
    "/data/conda/envs/zym_isaaclab/lib/python3.11/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
_BEHAVIOR_GRID_DURATION_S = 15.0


def _behavior_grid_clips(duration: float):
    half = duration * 0.5
    return (
        (
            "back_then_run",
            "vx=-0.5 then 3.0",
            ("--scene", "plane", "--cmd_schedule", f"{half},-0.5,0,0;{duration},3.0,0,0"),
        ),
        (
            "strafe_0.5",
            "vy=+/-0.5 m/s",
            ("--scene", "plane", "--cmd_schedule", f"{half},0,0.5,0;{duration},0,-0.5,0"),
        ),
        (
            "turn_1.5",
            "yaw=+/-1.5 rad/s",
            ("--scene", "plane", "--cmd_schedule", f"{half},0,0,1.5;{duration},0,0,-1.5"),
        ),
        (
            "stairs_0.6",
            "stairs vx=0.6",
            ("--scene", "stairs", "--cmd_vx", "0.6"),
        ),
    )


def _model_video_stem(checkpoint: str) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    return stem[6:] if stem.startswith("model_") else stem


def _behavior_grid_out_path(args: argparse.Namespace) -> str:
    if args.video_path:
        return os.path.abspath(args.video_path)
    ckpt = getattr(args, "checkpoint", None)
    if ckpt:
        run_dir = os.path.dirname(os.path.abspath(ckpt))
        return os.path.join(run_dir, "videos", f"{_model_video_stem(ckpt)}.mp4")
    actor = os.path.abspath(args.actor)
    if os.path.basename(os.path.dirname(actor)) == "exported":
        return os.path.join(os.path.dirname(os.path.dirname(actor)), "videos", "grid.mp4")
    return os.path.abspath("grid.mp4")


def _record_behavior_grid(args: argparse.Namespace) -> None:
    """Record four 15s behaviors and stitch a 2x2 H.264 grid (top-left to bottom-right)."""
    out_path = _behavior_grid_out_path(args)
    clips_dir = os.path.join(os.path.dirname(out_path), "clips")
    os.makedirs(clips_dir, exist_ok=True)
    duration = _BEHAVIOR_GRID_DURATION_S if args.duration is None else float(args.duration)
    clips = _behavior_grid_clips(duration)
    common = [
        sys.executable,
        os.path.abspath(__file__),
        "--depth_encoder",
        args.depth_encoder,
        "--actor",
        args.actor,
        "--headless",
        "--no_plots",
        "--duration",
        str(duration),
    ]
    if args.mujoco_xml:
        common.extend(["--mujoco_xml", args.mujoco_xml])
    if args.chase_camera:
        common.append("--chase_camera")
    clip_paths = []
    log_paths = []
    jobs: list[tuple[str, subprocess.Popen, object]] = []
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MPLBACKEND", "Agg")
    env["TQDM_DISABLE"] = "1"
    print(f"[INFO] Recording {len(clips)} clips in parallel")
    for name, _label, extra in clips:
        clip = os.path.join(clips_dir, f"sim2sim_{name}.mp4")
        log_path = os.path.join(clips_dir, f"{name}.log")
        cmd = common + ["--video_path", clip, *extra]
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        jobs.append((name, proc, log_f))
        clip_paths.append(clip)
        log_paths.append(log_path)
        print(f"[INFO] Recording {name} pid={proc.pid}")
    failed: list[tuple[str, int, str]] = []
    for (name, proc, log_f), log_path in zip(jobs, log_paths):
        rc = proc.wait()
        log_f.close()
        if rc != 0:
            failed.append((name, rc, log_path))
        else:
            print(f"[INFO] Finished {name}")
    if failed:
        for name, rc, log_path in failed:
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-40:])
            except OSError:
                tail = ""
            print(f"[ERROR] {name} exited {rc}\n{tail}")
        raise RuntimeError(f"behavior_grid clip failed: {[n for n, _, _ in failed]}")
    ffmpeg = _FFMPEG_BIN if os.path.isfile(_FFMPEG_BIN) else "ffmpeg"
    tmp = out_path + ".tmp.mp4"
    filt = (
        "[0:v]scale=960:540,setsar=1[v0];"
        "[1:v]scale=960:540,setsar=1[v1];"
        "[2:v]scale=960:540,setsar=1[v2];"
        "[3:v]scale=960:540,setsar=1[v3];"
        "[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
    )
    subprocess.run(
        [
            ffmpeg, "-y",
            *[x for p in clip_paths for x in ("-i", p)],
            "-filter_complex", filt, "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            "-movflags", "+faststart", tmp,
        ],
        check=True,
    )
    cap = cv2.VideoCapture(tmp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labeled = out_path + ".labeled.mp4"
    writer = cv2.VideoWriter(labeled, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    origins = ((20, 40), (w // 2 + 20, 40), (20, h // 2 + 40), (w // 2 + 20, h // 2 + 40))
    labels = [item[1] for item in clips]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for (x, y), text in zip(origins, labels):
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    subprocess.run(
        [
            ffmpeg, "-y", "-i", labeled,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            "-movflags", "+faststart", out_path,
        ],
        check=True,
    )
    for extra in (tmp, labeled, *clip_paths, *log_paths):
        if os.path.isfile(extra):
            os.remove(extra)
    try:
        os.rmdir(clips_dir)
    except OSError:
        pass
    print(f"[INFO] Behavior grid saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP1 parkour sim2sim (depth_encoder.onnx + actor.onnx).")
    default_export = (
        "exported"
    )
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=f"{default_export}/depth_encoder.onnx",
        help="Path to depth encoder ONNX.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=f"{default_export}/policy_ssr.onnx",
        help="Path to actor ONNX (includes obs normalizer if exported with normalization).",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=None,
        help="MJCF path (absolute or relative); if set, overrides --scene (stairs/terrain/plane).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=("stairs", "terrain", "plane"),
        default="stairs",
        help="Scene: stairs=rp1_stairs.xml (pyramids + trapezoid + platforms); terrain=rp1_rough.xml; plane=rp1_flat.xml.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI; record simulation_parkour.mp4 (depth preview and MuJoCo viewer disabled).",
    )
    parser.add_argument(
        "--no_depth_vis",
        action="store_true",
        default=False,
        help="Do not open OpenCV depth preview (default: one window, metric + encoder side by side).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds (default: cfg.sim_config.sim_duration).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Headless mp4 path (default: simulation_parkour.mp4). With --behavior_grid, omit this and pass --checkpoint to name the file as videos/<iter>.mp4.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Training .pt path. With --behavior_grid and no --video_path, output is <run>/videos/<iter>.mp4 (e.g. model_41000.pt -> 41000.mp4).",
    )
    parser.add_argument(
        "--behavior_grid",
        action="store_true",
        default=False,
        help=(
            "Record four 15s clips: back 0.5 then run 3.0, strafe ±0.5, yaw ±1.5, "
            "and stairs walk 0.6. Bidirectional clips are half duration per direction."
        ),
    )
    parser.add_argument(
        "--cmd_vx",
        type=float,
        default=None,
        help="Scripted forward velocity command (m/s). Unspecified axes default to 0.",
    )
    parser.add_argument(
        "--cmd_vy",
        type=float,
        default=None,
        help="Scripted lateral velocity command (m/s).",
    )
    parser.add_argument(
        "--cmd_dyaw",
        type=float,
        default=None,
        help="Scripted yaw-rate command (rad/s). Right turn is negative.",
    )
    parser.add_argument(
        "--cmd_schedule",
        type=str,
        default=None,
        help="Piecewise velocity 't_end,vx,vy,dyaw;...' in seconds. Overrides --cmd_vx/vy/dyaw.",
    )
    parser.add_argument(
        "--chase_camera",
        action="store_true",
        default=False,
        help="Use third-person chase camera (default: orbit follow).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Do not write joint/velocity matplotlib plots.",
    )
    args = parser.parse_args()
    if args.behavior_grid:
        _record_behavior_grid(args)
        raise SystemExit(0)

    mjcf_dir = f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/mjcf"
    scene_xml = {
        "stairs": f"{mjcf_dir}/rp1_stairs.xml",
        "terrain": f"{mjcf_dir}/rp1_rough.xml",
        "plane": f"{mjcf_dir}/rp1_flat.xml",
    }
    if args.mujoco_xml:
        xml_path = args.mujoco_xml
    else:
        xml_path = scene_xml[args.scene]

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = xml_path
            sim_duration = 1_000_000.0
            dt = 0.005
            decimation = 4
            depth_camera_body = "waist_yaw_link"

        class robot_config:
            # PD gains / default pos in URDF order (gmr_dof_names).
            kps = np.array(
                [
                    120.0, 120.0, 90.0, 150.0, 40.0, 40.0,
                    120.0, 120.0, 90.0, 150.0, 40.0, 40.0,
                    300.0, 250.0,
                    40.0, 40.0, 20.0, 30.0, 20.0,
                    40.0, 40.0, 20.0, 30.0, 20.0,
                ],
                dtype=np.double,
            )
            kds = np.array(
                [
                    8.0, 8.0, 6.0, 10.0, 2.0, 2.0,
                    8.0, 8.0, 6.0, 10.0, 2.0, 2.0,
                    25.0, 20.0,
                    3.0, 3.0, 1.0, 1.5, 1.0,
                    3.0, 3.0, 1.0, 1.5, 1.0,
                ],
                dtype=np.double,
            )
            default_pos = np.array(
                [
                    -0.1, 0.0, 0.0, -0.3, -0.2, 0.0,
                    0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    0.0, 0.0,
                    0.2, 0.2, 0.0, -1.2, 0.0,
                    -0.2, -0.2, 0.0, 1.2, 0.0,
                ],
                dtype=np.double,
            )
            # effort_limit_sim (hip/knee/waist 145.53, ankle 56, arm 28).
            tau_limit = np.array(
                [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
                + [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
                + [145.53, 145.53]
                + [28.0] * 5
                + [28.0] * 5,
                dtype=np.double,
            )
            frame_stack = 8  # obs history length
            depth_encoder_frames = 8  # encoder depth frames (subsampled from 37-frame buffer)

            num_actions = 24
            action_scale = 0.25
            # lab_dof_names[i] -> gmr/URDF index (deployment usd2urdf).
            usd2urdf = [0, 6, 12, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 18, 23]

    if args.duration is not None:
        Sim2simCfg.sim_config.sim_duration = float(args.duration)
    if args.cmd_schedule:
        segs: list[tuple[float, float, float, float]] = []
        for part in args.cmd_schedule.split(";"):
            part = part.strip()
            if not part:
                continue
            t_end, vx, vy, dyaw = (float(x) for x in part.split(","))
            segs.append((t_end, vx, vy, dyaw))
        if not segs:
            raise ValueError("--cmd_schedule is empty")
        cmd.scripted_schedule = segs
    elif any(v is not None for v in (args.cmd_vx, args.cmd_vy, args.cmd_dyaw)):
        cmd.scripted_target = (
            0.0 if args.cmd_vx is None else float(args.cmd_vx),
            0.0 if args.cmd_vy is None else float(args.cmd_vy),
            0.0 if args.cmd_dyaw is None else float(args.cmd_dyaw),
        )
    if args.chase_camera:
        cmd.camera_mode = CameraMode.CHASE
        cmd.camera_follow = True

    enc_sess, act_sess = build_onnx_sessions(
        args.depth_encoder, args.actor, providers=_SIM2SIM_PERF_ONNX_PROVIDERS
    )
    run_mujoco_onnx(
        enc_sess,
        act_sess,
        Sim2simCfg(),
        headless=args.headless,
        debug_obs=_SIM2SIM_PERF_DEBUG_OBS,
        show_depth_vis=not args.no_depth_vis,
        depth_vis_scale=max(1, _SIM2SIM_PERF_DEPTH_VIS_SCALE),
        realtime_sync=_SIM2SIM_PERF_REALTIME_SYNC and not args.headless,
        quiet=_SIM2SIM_PERF_QUIET,
        depth_vis_every_step=_SIM2SIM_PERF_DEPTH_VIS_EVERY_STEP,
        depth_vis_policy_stride=max(1, _SIM2SIM_PERF_DEPTH_VIS_POLICY_STRIDE),
        viewer_sync_every=_SIM2SIM_PERF_VIEWER_SYNC_EVERY,
        viewer_fallback_width=max(320, _SIM2SIM_PERF_VIEWER_FALLBACK_W),
        viewer_fallback_height=max(240, _SIM2SIM_PERF_VIEWER_FALLBACK_H),
        video_path=args.video_path or "simulation_parkour.mp4",
        save_plots=not args.no_plots,
    )
