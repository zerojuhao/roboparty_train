# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import mujoco, mujoco_viewer
from collections import deque
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import matplotlib.pyplot as plt
import glfw
from pynput import keyboard
import time
from robolab.assets import ISAAC_DATA_DIR

_OBS_HISTORY_KEYS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)


class TermHistory:
    """Isaac CircularBuffer semantics: maxlen ring, flattened oldest-to-newest per observation term."""

    def __init__(self, max_len: int, term_dim: int):
        self.max_len = max_len
        self.term_dim = term_dim
        self._dq: deque[np.ndarray] = deque(maxlen=max_len)

    def reset(self):
        self._dq.clear()

    def append(self, x: np.ndarray):
        self._dq.append(np.asarray(x, dtype=np.float32).reshape(-1))

    def fill_tile(self, x: np.ndarray):
        self.reset()
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        for _ in range(self.max_len):
            self._dq.append(v.copy())

    def flat(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.zeros(self.max_len * self.term_dim, dtype=np.float32)
        return np.concatenate(list(self._dq), axis=0)


class CompactOverlayMujocoViewer(mujoco_viewer.MujocoViewer):
    """MujocoViewer with the default left-side overlay hidden."""

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


def open_interactive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fallback_width: int = 1920,
    fallback_height: int = 1080,
) -> mujoco_viewer.MujocoViewer:
    viewer = CompactOverlayMujocoViewer(
        model, data, mode="window", width=int(fallback_width), height=int(fallback_height)
    )
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]
    return viewer


def sleep_until(target_time: float, busy_wait_margin: float) -> None:
    remaining = target_time - time.perf_counter()
    if remaining > busy_wait_margin:
        time.sleep(remaining - busy_wait_margin)
    while time.perf_counter() < target_time:
        pass


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.1
    vy_increment = 0.1
    dyaw_increment = 0.1

    min_vx = -1
    max_vx = 2.5
    min_vy = -0.8
    max_vy = 0.8
    min_dyaw = -1.5
    max_dyaw = 1.5
    camera_follow = True
    reset_requested = False
    
    @classmethod
    def update_vx(cls, delta):
        """update forward velocity"""
        cls.vx = np.clip(cls.vx + delta, cls.min_vx, cls.max_vx)
    
    @classmethod
    def update_vy(cls, delta):
        """update lateral velocity"""
        cls.vy = np.clip(cls.vy + delta, cls.min_vy, cls.max_vy)
    
    @classmethod
    def update_dyaw(cls, delta):
        """update angular velocity"""
        cls.dyaw = np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw)

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")
    
    @classmethod
    def reset(cls):
        """reset all velocities to zero"""
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        print(f"Velocities reset: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")
def on_press(key):
    """Key press event handler"""
    try:
        # Number key controls: 8/5 control forward/backward (vx), 4/6 control left/right (vy), 7/9 control left/right turn (dyaw)
        if hasattr(key, 'char') and key.char is not None:
            c = key.char.lower()
            if c == '8':
                # 8 -> forward (increase vx)
                cmd.update_vx(cmd.vx_increment)
            elif c == '2':
                # 2 -> backward (decrease vx)
                cmd.update_vx(-cmd.vx_increment)
            elif c == '4':
                # 4 -> left (decrease vy)
                cmd.update_vy(cmd.vy_increment)
            elif c == '6':
                # 6 -> right (increase vy)
                cmd.update_vy(-cmd.vy_increment)
            elif c == '7':
                # 7 -> turn left (increase dyaw)
                cmd.update_dyaw(cmd.dyaw_increment)
            elif c == '9':
                # 9 -> turn right (decrease dyaw)
                cmd.update_dyaw(-cmd.dyaw_increment)
            elif c == 'f':
                # toggle camera follow
                cmd.toggle_camera_follow()
            elif c == '0':
                # request reset robot state in main loop (thread-safe flag)
                cmd.reset_requested = True
                print('Reset requested (0 key pressed)')
    except AttributeError:
        pass

def on_release(key):
    """Key release event handler"""
    # If movement should only occur while keys are held down, handle it here
    pass

def start_keyboard_listener():
    """Start keyboard listener"""
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def viewer_velocity_overlay(viewer, cmd_vx, cmd_vy, cmd_wz, base_v, base_omega):
    """Show command and measured base-frame velocity in the active MuJoCo viewer."""
    if hasattr(viewer, "set_velocity_table"):
        viewer.set_velocity_table((cmd_vx, cmd_vy, cmd_wz), (base_v[0], base_v[1], base_omega[2]))
    elif hasattr(viewer, "set_texts"):
        left_col = "\ncmd\nvel"
        right_col = (
            "   x       y       z\n"
            f"{cmd_vx:+.2f}   {cmd_vy:+.2f}   {cmd_wz:+.2f}\n"
            f"{base_v[0]:+.2f}   {base_v[1]:+.2f}   {base_omega[2]:+.2f}"
        )
        viewer.set_texts((None, None, left_col, right_col))

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg, headless=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    # Start keyboard listener
    print("=" * 60)
    print("Keyboard control instructions:")
    print("  ↑ Up arrow: Increase forward speed (vx)")
    print("  ↓ Down arrow: Decrease forward speed (vx)")
    print("  ← Left arrow: Increase left turn rate (dyaw)")
    print("  → Right arrow: Increase right turn rate (dyaw)")
    print("  0 key: Reset all speeds to 0")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)

   
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    

    
    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     
        cam = mujoco.MjvCamera()
        cam.distance = 4.0      
        cam.azimuth = 45.0   
        cam.elevation = -20.0   
        cam.lookat = [0, 0, 1]  
        out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        viewer = open_interactive_viewer(
            model,
            data,
            fallback_width=1920,
            fallback_height=1080,
        )


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    hist = {
        "base_ang_vel": TermHistory(cfg.robot_config.frame_stack, 3),
        "projected_gravity": TermHistory(cfg.robot_config.frame_stack, 3),
        "velocity_commands": TermHistory(cfg.robot_config.frame_stack, 3),
        "joint_pos": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
        "joint_vel": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
        "actions": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
    }

    count_lowlevel = 0
    next_render_time = 0.0
    render_interval = 1.0 / cfg.sim_config.render_fps

    # --- Data collection lists for plotting (LOW FREQUENCY ONLY) ---
    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)  # Initialize tau
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = [] # Store [vx, vy] at low freq
    actual_ang_vel_data = [] # Store [wz] at low freq
    # -------------------------------------------------------------
    is_first_frame = True
    
    start_time = time.perf_counter()
    
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating...", mininterval=1.0):
        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            # clear commands and history
            cmd.reset()
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            action[:] = 0.0
            target_pos[:] = cfg.robot_config.default_pos.copy()
            tau[:] = 0.0
            for h in hist.values():
                h.reset()
            is_first_frame = True
            cmd.reset_requested = False

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot_config.num_actions:]
        dq = dq[-cfg.robot_config.num_actions:]

        # 1000hz -> 100hz/50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            q_ = q - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            vecs_policy = (
                omega.astype(np.float32),
                gvec.astype(np.float32),
                np.array([cmd.vx, cmd.vy, cmd.dyaw], dtype=np.float32),
                q_obs.astype(np.float32),
                dq_obs.astype(np.float32),
                action.astype(np.float32),
            )

            if is_first_frame:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].fill_tile(vec)
                is_first_frame = False
            else:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].append(vec)

            policy_input = np.concatenate([hist[k].flat() for k in _OBS_HISTORY_KEYS], axis=0)[None, :].astype(np.float32)
            assert policy_input.shape[1] == cfg.robot_config.num_observations, (
                f"Expected policy input dim {cfg.robot_config.num_observations}, got {policy_input.shape[1]}."
            )
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]
            target_pos = target_pos + cfg.robot_config.default_pos

            # --- Capture actual state at this low-frequency step ---
            # Note: q, v, omega were just computed by get_obs() for the current simulation step
            q_low_freq = q.copy()
            v_low_freq = v[:2].copy() # Capture x and y linear velocity
            omega_low_freq = omega[2].copy() # Capture z angular velocity
            # -----------------------------------------------------

            # --- Collect low-frequency data for plotting ---
            # Use the exact simulation time at this low-freq step
            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq) # Use the captured actual joint pos
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v_low_freq) # Use the captured actual lin vel
            actual_ang_vel_data.append(omega_low_freq) # Use the captured actual ang vel
            # ----------------------------------------------

            if headless:
                renderer.update_scene(data, camera=cam)
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                img = renderer.render() 
                out.write(img)
            
        target_vel = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_pos, q, cfg.robot_config.kps,
                        target_vel, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau
        mujoco.mj_step(model, data)

        count_lowlevel += 1

        if not headless:
            sim_time = (step + 1) * cfg.sim_config.dt
            if sim_time >= next_render_time:
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer_velocity_overlay(viewer, cmd.vx, cmd.vy, cmd.dyaw, v, omega)
                viewer.render()
                while next_render_time <= sim_time:
                    next_render_time += render_interval
        
        target_wall_time = start_time + (step + 1) * cfg.sim_config.dt
        sleep_until(target_wall_time, cfg.sim_config.busy_wait_margin)

    if headless:
        out.release()
    else:
        viewer.close()
    
    # Stop keyboard listener
    keyboard_listener.stop()

     # --- Plotting Section (Using only low-frequency data) ---

    print("Simulation finished. Generating plots...")

    # Convert collected data to numpy arrays
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    tau_data = np.array(tau_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)


    # Plot 1: Commanded vs Actual Joint Positions
    num_joints = cfg.robot_config.num_actions
    n_cols = 4 # Or adjust based on num_joints
    n_rows = (num_joints + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    joint_names = [f'Joint {i+1}' for i in range(num_joints)] # Generic names (consider using specific robot joint names if available)

    for i in range(num_joints):
        ax = axes1[i]
        # Plotting low-frequency commanded and actual joint positions
        ax.plot(time_data, commanded_joint_pos_data[:, i], label='Commanded', linestyle='--')
        ax.plot(time_data, actual_joint_pos_data[:, i], label='Actual')
        ax.set_title(joint_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])

    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()


    # Plot 2: Commanded vs Actual Base Velocities
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Linear Velocity X
    # Plotting low-frequency commanded and actual velocities
    axes2[0].plot(time_data, commanded_lin_vel_x_data, label='Commanded Vx', linestyle='--')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label='Actual Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    # Linear Velocity Y
    axes2[1].plot(time_data, commanded_lin_vel_y_data, label='Commanded Vy', linestyle='--')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label='Actual Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    # Angular Velocity Z
    axes2[2].plot(time_data, commanded_ang_vel_z_data, label='Commanded Dyaw', linestyle='--')
    axes2[2].plot(time_data, actual_ang_vel_data, label='Actual Dyaw') # actual_ang_vel_data is already 1D
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()

    # plt.show()
    fig1.savefig("joint_positions.png")
    fig2.savefig("base_velocities.png")

    print("Plots finished.")
    # --- End Plotting Section ---

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', 
                        # type=str, 
                        default="policy.pt",
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default='plane', help='terrain or plane')
    parser.add_argument('--headless', action='store_true', help='Run without GUI and save video')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo.xml'
            sim_duration = 1000000.0
            dt = 0.005
            decimation = 4
            render_fps = 120.0
            busy_wait_margin = 0.0005

        class robot_config:
            kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 40, 40, 40, 30, 20, 40, 40, 40, 30, 20], dtype=np.double)
            kds = np.array([3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.5, 1.0], dtype=np.double)
            default_pos = np.array([0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0, 0, 0.18, 0.06, 0, 0.78, 0, 0.18, -0.06, 0, 0.78, 0], dtype=np.double)
            tau_limit = 200. * np.ones(23, dtype=np.double)
            frame_stack = 3 # obs history length
            num_single_obs = 78
            num_observations = num_single_obs * frame_stack
            num_actions = 23
            action_scale = 0.25
            # 'left_thigh_yaw_joint', 'right_thigh_yaw_joint', 'torso_joint', 'left_thigh_roll_joint', 'right_thigh_roll_joint', 'left_arm_pitch_joint', 'right_arm_pitch_joint', 'left_thigh_pitch_joint', 'right_thigh_pitch_joint', 'left_arm_roll_joint', 'right_arm_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_arm_yaw_joint', 'right_arm_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_yaw_joint', 'right_elbow_yaw_joint'
            usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
