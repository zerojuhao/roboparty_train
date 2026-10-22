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

from __future__ import annotations

import os
import numpy as np
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation

from isaaclab.managers import ManagerBase, ManagerTermBase
from .motion_data_term_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from robolab.utils.math import vel_forward_diff, ang_vel_from_quat_diff, quat_slerp, linear_interpolate, calc_frame_blend
from robolab.utils.motion_npz import load_motion, name_indices, resolve_name_patterns
from robolab.utils.tensors import as_torch_tensor


class MotionDataTerm(ManagerTermBase):
    
    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert os.path.exists(cfg.motion_data_dir), \
            f"Motion data directory {cfg.motion_data_dir} does not exist."
            
        self._load_motion_data()
        
    def _load_motion_data(self):
        # list the motion data files in the directory
        motion_files = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith('.npz')]
        if not motion_files:
            raise ValueError(f"No unified .npz motion files found in {self.cfg.motion_data_dir}.")
        self.joint_names = list(self._env.scene["robot"].joint_names)
        
        self.motion_weights_dict = self.cfg.motion_data_weights
        self.motion_names = []

        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []
        self.motion_features = []
        
        self.root_pos_w = []
        self.root_quat = []
        self.root_vel_w = []
        self.root_ang_vel_w = []
        self.dof_pos = []
        self.dof_vel = []
        self.key_body_pos_w = []

        # only load the motion data files that are in the motion weights dict
        for motion_name, motion_weight in self.motion_weights_dict.items():
            self.motion_names.append(motion_name)
            # check if the motion file name is valid
            motion_file = f"{motion_name}.npz"
            if motion_file not in motion_files:
                raise ValueError(f"Motion name {motion_name} defined in motion weights not found in motion data directory {self.cfg.motion_data_dir}. Available files: {motion_files}")

            # load the motion data file
            motion_path = os.path.join(self.cfg.motion_data_dir, motion_file)
            print(f"[Motion Data Manager] Loading motion data from {motion_path}...")
            # Do not request key bodies from ``load_motion`` here.  Older AMP
            # files contain only six cached key bodies, while RP1 now requests
            # every non-root link and computes missing positions from FK below.
            motion_raw_data = load_motion(motion_path, joint_names=self.joint_names)
            src_fps = float(motion_raw_data["fps"].reshape(-1)[0])
            if self.cfg.resample_fps is not None and abs(src_fps - float(self.cfg.resample_fps)) > 1e-6:
                motion_raw_data = self._resample_motion_raw(motion_raw_data, float(self.cfg.resample_fps))
                print(
                    f"[Motion Data Manager] Resampled {motion_file}: {src_fps:g} Hz -> "
                    f"{self.cfg.resample_fps:g} Hz ({len(motion_raw_data['root_pos_w'])} frames)"
                )
            requested_key_body_names = self.cfg.key_body_names
            if requested_key_body_names is None:
                requested_key_body_names = motion_raw_data["key_body_names"].tolist()
            else:
                requested_key_body_names = resolve_name_patterns(
                    self._env.scene["robot"].body_names, requested_key_body_names
                )
            requested_key_body_names = list(requested_key_body_names)
            if not self.key_body_pos_w:
                self.key_body_names = requested_key_body_names
            elif self.key_body_names != requested_key_body_names:
                raise ValueError(f"Inconsistent key-body order in {motion_file}.")
            
            # Some info about the motion
            fps = float(motion_raw_data["fps"].reshape(-1)[0])
            dt = 1.0 / fps
            num_frames = len(motion_raw_data["root_pos_w"])
            if num_frames < 2:
                raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
            duration = dt * (num_frames - 1)

            self.motion_durations.append(duration)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_weights.append(motion_weight)
            
            # Get the motion data
            
            # root position in world frame, shape (num_frames, 3)
            root_pos_w = torch.from_numpy(motion_raw_data["root_pos_w"]).to(self.device).float()
            root_pos_w.requires_grad_(False)
            # Unified files and Isaac Lab 3.0 both use XYZW.
            root_quat = torch.from_numpy(motion_raw_data["root_quat_w"]).to(self.device).float()
            root_quat.requires_grad_(False)
            
            # root velocity in world frame, shape (num_frames, 3)
            root_vel_w = vel_forward_diff(root_pos_w, dt)
            root_vel_w.requires_grad_(False)
            
            # root angular velocity in world frame, shape (num_frames, 3)
            root_ang_vel_w = ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
            root_ang_vel_w.requires_grad_(False)

            root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
            root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
            if self.cfg.sampling_mode == "command_match":
                self.motion_features.append(self._command_match_feature(root_vel_b, root_ang_vel_b))
            
            # dof position, shape (num_frames, num_dofs)
            dof_pos = torch.from_numpy(motion_raw_data["joint_pos"]).to(self.device).float()
            dof_pos.requires_grad_(False)
            
            # dof velocity, shape (num_frames, num_dofs)
            dof_vel = vel_forward_diff(dof_pos, dt)
            dof_vel.requires_grad_(False)
            
            # Key-body positions are read from the file when available.  RP1's
            # all-link configuration falls back to batched Newton FK, so no
            # second dataset-conversion pass is needed when the cached file only
            # contains the historical six key bodies.
            key_body_pos_w = self._load_or_compute_key_body_pos_w(
                motion_raw_data, requested_key_body_names, motion_file
            )
            
            self.root_pos_w.append(root_pos_w)
            self.root_quat.append(root_quat)
            self.root_vel_w.append(root_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.dof_pos.append(dof_pos)
            self.dof_vel.append(dof_vel)
            self.key_body_pos_w.append(key_body_pos_w)
        
        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float32, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float32, device=self.device)
        self.motion_durations = torch.tensor(self.motion_durations, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.int32, device=self.device)
        if self.cfg.sampling_mode == "command_match":
            self.motion_features = torch.stack(self.motion_features)
        # Get the normalized motion weights
        self.motion_weights = torch.tensor(self.motion_weights, dtype=torch.float32, device=self.device)
        self.motion_weights = self.motion_weights / torch.sum(self.motion_weights)
        static_names = set(self.cfg.command_match_static_motion_names)
        missing_static_names = static_names.difference(self.motion_names)
        if missing_static_names:
            raise ValueError(
                "Static command-match motions are not configured in motion_data_weights: "
                f"{sorted(missing_static_names)}"
            )
        if self.cfg.command_match_static_threshold < 0.0:
            raise ValueError("command_match_static_threshold must be non-negative.")
        self.static_motion_ids = torch.tensor(
            [index for index, name in enumerate(self.motion_names) if name in static_names],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.sampling_mode not in ("static", "command_match"):
            raise ValueError(f"Unsupported motion sampling_mode: {self.cfg.sampling_mode}")
        if self.cfg.sampling_mode == "command_match":
            if not self.cfg.command_name:
                raise ValueError("command_match sampling requires command_name")
            if self.cfg.command_match_temperature <= 0.0:
                raise ValueError("command_match_temperature must be positive")
            if not 0.0 <= self.cfg.command_match_min_probability <= 1.0:
                raise ValueError("command_match_min_probability must be in [0, 1]")
            if self.cfg.command_match_velocity_scale <= 0.0 or self.cfg.command_match_yaw_scale <= 0.0:
                raise ValueError("command matching scales must be positive")
            if self.cfg.command_match_direction_weight < 0.0:
                raise ValueError("command_match_direction_weight must be non-negative")
            if self.cfg.command_match_direction_threshold < 0.0:
                raise ValueError("command_match_direction_threshold must be non-negative")
            if self.cfg.command_match_current_weight < 0.0 or self.cfg.command_match_target_weight < 0.0:
                raise ValueError("command match weights must be non-negative")
            if self.cfg.command_match_current_weight + self.cfg.command_match_target_weight <= 0.0:
                raise ValueError("command match weights cannot both be zero")
            if self.cfg.command_match_cruise_speed_threshold < 0.0:
                raise ValueError("command_match_cruise_speed_threshold must be non-negative")
            if not 0.0 <= self.cfg.command_match_cruise_min_fraction <= 1.0:
                raise ValueError("command_match_cruise_min_fraction must be in [0, 1]")
        
        # Some other information
        self.num_dofs = self.dof_pos[0].shape[1]
        self.num_key_bodies = self.key_body_pos_w[0].shape[1]
        
        # Concatenate all motion data along the first dimension
        self.root_pos_w = torch.cat(self.root_pos_w, dim=0)
        self.root_quat = torch.cat(self.root_quat, dim=0)
        self.root_vel_w = torch.cat(self.root_vel_w, dim=0)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w, dim=0)
        self.dof_pos = torch.cat(self.dof_pos, dim=0)
        self.dof_vel = torch.cat(self.dof_vel, dim=0)
        self.key_body_pos_w = torch.cat(self.key_body_pos_w, dim=0)
        
        num_motions = self.get_num_motions()
        self.motion_ids = torch.arange(num_motions, dtype=torch.long, device=self.device)
        
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)
        
        return
         
    # Some helper functions
    
    def get_num_motions(self) -> int:
        """Get the number of motions loaded."""
        return self.motion_num_frames.shape[0]
    
    def get_total_duration(self) -> float:
        """Get the total duration of all motions."""
        return torch.sum(self.motion_durations).item()

    def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the duration of a specific motion.

        Args:
            motion_id (torch.Tensor): A tensor of motion IDs for which to get the duration.

        Returns:
            float: The duration of the motion in seconds.
        """
        return self.motion_durations[motion_ids]

    @staticmethod
    def _resample_motion_raw(data: dict, target_fps: float) -> dict:
        """Uniformly resample a loaded clip to ``target_fps`` (lerp + slerp)."""
        src_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        n_src = len(data["root_pos_w"])
        duration = (n_src - 1) / src_fps
        n_dst = int(np.floor(duration * target_fps + 1e-6)) + 1
        t_src = np.arange(n_src) / src_fps
        t_dst = np.minimum(np.arange(n_dst) / target_fps, t_src[-1])
        i0 = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, n_src - 2)
        i1 = i0 + 1
        alpha = ((t_dst - t_src[i0]) * src_fps).astype(np.float32)

        def lerp(x):
            x = np.asarray(x, dtype=np.float32)
            a = alpha.reshape((-1,) + (1,) * (x.ndim - 1))
            return (1.0 - a) * x[i0] + a * x[i1]

        def slerp(q):
            q = np.asarray(q, dtype=np.float32)
            q0, q1 = q[i0], q[i1]
            dot = np.sum(q0 * q1, axis=-1, keepdims=True)
            q1 = np.where(dot < 0.0, -q1, q1)
            dot = np.abs(dot).clip(-1.0, 1.0)
            a = alpha.reshape((-1,) + (1,) * (q.ndim - 1))
            theta = np.arccos(dot)
            sin_t = np.sin(theta)
            small = sin_t < 1e-6
            w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / np.where(small, 1.0, sin_t))
            w1 = np.where(small, a, np.sin(a * theta) / np.where(small, 1.0, sin_t))
            out = w0 * q0 + w1 * q1
            return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-8, None)

        out = dict(data)
        for key, value in data.items():
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[0] != n_src or arr.dtype.kind not in "fc":
                continue
            out[key] = slerp(arr) if key in ("root_quat_w", "body_quat_w") else lerp(arr)
        out["fps"] = np.asarray(float(target_fps))
        return out

    def _load_or_compute_key_body_pos_w(
        self,
        motion_raw_data: dict,
        requested_names: list[str],
        motion_file: str,
    ) -> torch.Tensor:
        """Return requested link positions, using startup Newton FK when needed.

        The original RP1 AMP files cache only six key bodies. Keeping that
        format is useful for compatibility, but it cannot satisfy the current
        all-link discriminator configuration. In that case the motion's root
        pose and joint positions are replayed into a batched Newton articulation
        and the resulting link poses are copied into the motion buffer.
        """
        raw_key_names = motion_raw_data.get("key_body_names")
        raw_key_names = raw_key_names.tolist() if raw_key_names is not None else []
        if "key_body_pos_w" in motion_raw_data and set(requested_names).issubset(raw_key_names):
            indices = name_indices(raw_key_names, requested_names)
            key_body_pos_w = torch.from_numpy(
                motion_raw_data["key_body_pos_w"][:, indices]
            ).to(self.device).float()
            key_body_pos_w.requires_grad_(False)
            return key_body_pos_w

        raw_body_names = motion_raw_data.get("body_names")
        raw_body_names = raw_body_names.tolist() if raw_body_names is not None else []
        if "body_pos_w" in motion_raw_data and set(requested_names).issubset(raw_body_names):
            indices = name_indices(raw_body_names, requested_names)
            key_body_pos_w = torch.from_numpy(
                motion_raw_data["body_pos_w"][:, indices]
            ).to(self.device).float()
            key_body_pos_w.requires_grad_(False)
            return key_body_pos_w

        print(
            f"[Motion Data Manager] {motion_file}: computing "
            f"{len(requested_names)} key-body positions with Newton FK"
        )
        return self._compute_key_body_pos_w_with_newton(
            motion_raw_data, requested_names, motion_file
        )

    def _compute_key_body_pos_w_with_newton(
        self,
        motion_raw_data: dict,
        requested_names: list[str],
        motion_file: str,
    ) -> torch.Tensor:
        """Compute link positions from a motion using the live Newton model.

        This runs at manager initialization, before rollout or learning starts.
        It uses the same articulation model and body-frame conventions as the
        policy observations, avoiding a second URDF FK implementation that could
        disagree with Newton at fixed-link offsets.
        """
        robot = self._env.scene["robot"]
        backend_name = getattr(robot.data, "__backend_name__", None)
        if backend_name != "newton":
            raise RuntimeError(
                f"{motion_file}: all-link AMP features require the Newton backend; "
                f"got {backend_name!r}."
            )

        body_names = list(robot.body_names)
        if not body_names or body_names[0] != "base_link":
            raise RuntimeError(
                f"{motion_file}: RP1 FK requires body index 0 to be base_link, got {body_names[:3]}"
            )
        try:
            body_indices = name_indices(body_names, requested_names)
        except ValueError as exc:
            raise ValueError(
                f"{motion_file}: requested RP1 key bodies are not present in the Newton articulation"
            ) from exc

        root_pos = torch.from_numpy(motion_raw_data["root_pos_w"]).to(self.device).float()
        root_quat = torch.from_numpy(motion_raw_data["root_quat_w"]).to(self.device).float()
        joint_pos = torch.from_numpy(motion_raw_data["joint_pos"]).to(self.device).float()
        num_frames = int(root_pos.shape[0])
        num_envs = int(self._env.num_envs)
        if num_envs < 1:
            raise RuntimeError("Cannot compute motion FK without at least one environment")

        env_origins = as_torch_tensor(self._env.scene.env_origins).to(
            device=self.device, dtype=torch.float32
        )[:, :3]
        if env_origins.shape[0] < min(num_envs, num_frames):
            raise RuntimeError("Scene environment origins do not cover the FK batch")

        # Save the live training scene because FK is evaluated by temporarily
        # writing the motion poses into the first batch of environments.
        saved_root_state = as_torch_tensor(robot.data.root_link_state_w).clone()
        saved_joint_pos = as_torch_tensor(robot.data.joint_pos).clone()
        saved_joint_vel = as_torch_tensor(robot.data.joint_vel).clone()
        key_body_pos_w = torch.empty(
            (num_frames, len(requested_names), 3), device=self.device, dtype=torch.float32
        )

        try:
            with torch.no_grad():
                batch_size = min(num_envs, num_frames)
                for start in range(0, num_frames, batch_size):
                    end = min(start + batch_size, num_frames)
                    env_ids = torch.arange(end - start, device=self.device, dtype=torch.long)
                    origins = env_origins[env_ids]
                    root_pose = torch.cat(
                        (root_pos[start:end] + origins, root_quat[start:end]), dim=-1
                    )
                    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
                    robot.write_joint_position_to_sim_index(
                        position=joint_pos[start:end], env_ids=env_ids
                    )

                    # Newton refreshes FK on this read after the manual pose
                    # writes; no physics step or dataset write is required.
                    body_pos = as_torch_tensor(robot.data.body_pos_w)
                    key_body_pos_w[start:end] = (
                        body_pos[env_ids][:, body_indices, :] - origins.unsqueeze(1)
                    )
        finally:
            with torch.no_grad():
                restore_env_ids = torch.arange(
                    saved_root_state.shape[0], device=self.device, dtype=torch.long
                )
                robot.write_root_link_pose_to_sim_index(
                    root_pose=saved_root_state[:, :7], env_ids=restore_env_ids
                )
                robot.write_root_link_velocity_to_sim_index(
                    root_velocity=saved_root_state[:, 7:], env_ids=restore_env_ids
                )
                robot.write_joint_state_to_sim_index(
                    position=saved_joint_pos,
                    velocity=saved_joint_vel,
                    env_ids=restore_env_ids,
                )
                # Ensure subsequent observations see the restored articulation.
                _ = as_torch_tensor(robot.data.body_pos_w)

        key_body_pos_w.requires_grad_(False)
        return key_body_pos_w

    def _command_match_feature(
        self, root_vel_b: torch.Tensor, root_ang_vel_b: torch.Tensor
    ) -> torch.Tensor:
        """Mean body-frame ``(vx, vy, wz)``, optionally gated to moving frames."""
        vx = root_vel_b[:, 0]
        vy = root_vel_b[:, 1]
        wz = root_ang_vel_b[:, 2]
        cruise = torch.linalg.norm(root_vel_b[:, :2], dim=-1) >= self.cfg.command_match_cruise_speed_threshold
        if (
            cruise.any()
            and cruise.to(dtype=vx.dtype).mean() >= self.cfg.command_match_cruise_min_fraction
        ):
            vx, vy, wz = vx[cruise], vy[cruise], wz[cruise]
        return torch.stack((vx.mean(), vy.mean(), wz.mean()))

    def _matching_target(
        self, commands: torch.Tensor, current_velocity: torch.Tensor | None = None
    ) -> torch.Tensor:
        target = commands[:, :3]
        if current_velocity is None:
            return target
        current = current_velocity[:, :3]
        current_weight = self.cfg.command_match_current_weight
        target_weight = self.cfg.command_match_target_weight
        return (current_weight * current + target_weight * target) / (current_weight + target_weight)

    def motion_probabilities(
        self,
        commands: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return command-matching probabilities based on each motion's cruise velocity."""
        commands = commands.to(device=self.device, dtype=torch.float32)
        if commands.ndim != 2 or commands.shape[1] < 3:
            raise ValueError(f"Expected commands with shape (N, >=3), got {tuple(commands.shape)}")
        if current_velocity is not None:
            current_velocity = current_velocity.to(device=self.device, dtype=torch.float32)
            if current_velocity.shape != commands.shape:
                raise ValueError("Current velocity must have the same shape as commands")
        target = self._matching_target(commands, current_velocity)
        scale = torch.tensor(
            [
                self.cfg.command_match_velocity_scale,
                self.cfg.command_match_velocity_scale,
                self.cfg.command_match_yaw_scale,
            ],
            device=self.device, dtype=torch.float32,
        )
        features = self.motion_features
        component_error = torch.abs(target[:, None, :] - features[None, :, :]) / scale
        distance = component_error.sum(dim=-1)

        # A single-axis command should not be matched by a motion whose main
        # activity is on another axis (for example, sidestep or backward motion
        # for a pure yaw command). Also penalize clips moving in the opposite
        # signed direction on an active axis.
        direction_weight = self.cfg.command_match_direction_weight
        if direction_weight > 0.0:
            command_axes = commands[:, :3]
            direction_threshold = self.cfg.command_match_direction_threshold
            command_active = command_axes.abs() >= direction_threshold
            motion_active = features.abs() >= direction_threshold

            inactive_axis_penalty = (
                features.abs()[None, :, :] / scale[None, None, :]
            ) * (~command_active[:, None, :]).to(features.dtype)
            distance = distance + direction_weight * inactive_axis_penalty.sum(dim=-1)

            opposite_direction = (
                command_active[:, None, :]
                & motion_active[None, :, :]
                & (command_axes[:, None, :] * features[None, :, :] < 0.0)
            )
            distance = distance + direction_weight * opposite_direction.to(features.dtype).sum(dim=-1)
        has_static_routing = (
            self.static_motion_ids.numel() > 0 and self.cfg.command_match_static_threshold > 0.0
        )
        if has_static_routing:
            command_magnitude = torch.maximum(
                torch.linalg.vector_norm(commands[:, :2], dim=1), commands[:, 2].abs()
            )
            static_command = command_magnitude < self.cfg.command_match_static_threshold

            # Remove static motions before the active-command softmax. Applying
            # the exploration floor first would redistribute their mass back
            # into active motions when the static candidates are removed.
            allowed = torch.ones(
                (commands.shape[0], self.motion_weights.shape[0]),
                dtype=torch.bool,
                device=self.device,
            )
            allowed[:, self.static_motion_ids] = static_command[:, None]
        else:
            static_command = torch.zeros(commands.shape[0], dtype=torch.bool, device=self.device)
            allowed = torch.ones(
                (commands.shape[0], self.motion_weights.shape[0]),
                dtype=torch.bool,
                device=self.device,
            )

        # Log-space normalization keeps the exploration floor at its configured
        # mass even when a target is far from every demonstration motion.
        logits = -distance / self.cfg.command_match_temperature + self.motion_weights.log()[None, :]
        probabilities = torch.softmax(logits.masked_fill(~allowed, -torch.inf), dim=1)
        floor = self.cfg.command_match_min_probability
        if floor > 0.0:
            prior = self.motion_weights[None, :] * allowed.to(self.motion_weights.dtype)
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(prior.dtype).eps
            )
            probabilities = (1.0 - floor) * probabilities + floor * prior

        if has_static_routing:
            static_prior = torch.zeros_like(probabilities)
            static_prior[:, self.static_motion_ids] = self.motion_weights[self.static_motion_ids]
            static_prior /= static_prior.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(probabilities.dtype).eps
            )

            probabilities = torch.where(static_command[:, None], static_prior, probabilities)
        return probabilities

    def sample_motions(
        self,
        n: int,
        commands: torch.Tensor | None = None,
        current_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample a batch of motion IDs.

        Args:
            n (int): The number of motion IDs to sample.
            commands (torch.Tensor | None): Command batch used for command matching.
            current_velocity (torch.Tensor | None): Current velocity batch blended into the match target.

        Returns:
            torch.Tensor: A tensor of sampled motion IDs, shape (n,).
        """
        if self.cfg.sampling_mode == "static" or commands is None:
            return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)
        if commands.shape[0] != n:
            raise ValueError("Command batch size must equal n")
        probabilities = self.motion_probabilities(commands, current_velocity)
        return torch.multinomial(probabilities, num_samples=1, replacement=True).squeeze(1)
        
    def sample_times(self, motion_ids: torch.Tensor, truncate_time_start: float = None, truncate_time_end: float = None):
        """Sample time within the duration of the given motions.
        
        Args:
            motion_ids (torch.Tensor): A tensor of motion IDs, shape (batch_size,).
            truncate_time_start (float | None): If provided, the sampled time will be truncated
                from the start, i.e., sampled in [truncate_time_start, duration]. Default is None.
            truncate_time_end (float | None): If provided, the sampled time will be truncated
                from the end, i.e., sampled in [0, duration - truncate_time_end]. Default is None.
                
        Returns:
            torch.Tensor: A tensor of sampled times, shape (batch_size,).
        """
        motion_durations = self.motion_durations[motion_ids]
        
        # Calculate valid time range
        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()
        
        if truncate_time_start is not None:
            assert truncate_time_start >= 0, f"[MotionLoader] truncate_time_start must be non-negative, but got {truncate_time_start}."
            time_start = torch.minimum(
                (time_start + truncate_time_start).clamp_min(0.0), motion_durations
            )
        
        if truncate_time_end is not None:
            assert truncate_time_end >= 0, f"[MotionLoader] truncate_time_end must be non-negative, but got {truncate_time_end}."
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)
        
        # Check if valid range exists
        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print("[Warning] Some motions have invalid time range after truncation (start >= end).")
            valid_range = torch.clamp(valid_range, min=1e-6)  # Prevent division by zero
        
        # Sample time within the valid range
        phase = torch.rand(motion_ids.shape, device=self.device)
        sample_times = time_start + phase * valid_range
        
        return sample_times

    def _calc_frame_blend(self, motion_ids: torch.Tensor, times: torch.Tensor):
        num_frames = self.motion_num_frames[motion_ids]
        motion_start_indices = self.motion_start_indices[motion_ids]
        phase = torch.clip(times / self.motion_durations[motion_ids], 0.0, 1.0)
        
        frame_idx0 = (phase * (num_frames - 1).float()).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1).float() - frame_idx0.float()
        
        frame_idx0 = frame_idx0 + motion_start_indices
        frame_idx1 = frame_idx1 + motion_start_indices
        
        return frame_idx0, frame_idx1, blend
        
    
    # def _allocate_temp_tensors(self, n):
    #     """Allocate temporary tensors for motion state computation."""
    #     root_pos_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_pos_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_quat_0 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_quat_1 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     dof_pos_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_pos_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_0 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_1 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)

    #     return (root_pos_w_0, root_pos_w_1,
    #             root_quat_0, root_quat_1,
    #             root_vel_w_0, root_vel_w_1,
    #             root_ang_vel_w_0, root_ang_vel_w_1,
    #             dof_pos_0, dof_pos_1,
    #             dof_vel_0, dof_vel_1,
    #             key_body_pos_w_0, key_body_pos_w_1)
    
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
    
        root_pos_w_0 = self.root_pos_w[frame_idx0]
        root_pos_w_1 = self.root_pos_w[frame_idx1]
        root_quat_0 = self.root_quat[frame_idx0]
        root_quat_1 = self.root_quat[frame_idx1]
        root_vel_w_0 = self.root_vel_w[frame_idx0]
        root_vel_w_1 = self.root_vel_w[frame_idx1]
        root_ang_vel_w_0 = self.root_ang_vel_w[frame_idx0]
        root_ang_vel_w_1 = self.root_ang_vel_w[frame_idx1]
        dof_pos_0 = self.dof_pos[frame_idx0]
        dof_pos_1 = self.dof_pos[frame_idx1]
        dof_vel_0 = self.dof_vel[frame_idx0]
        dof_vel_1 = self.dof_vel[frame_idx1]
        key_body_pos_w_0 = self.key_body_pos_w[frame_idx0]
        key_body_pos_w_1 = self.key_body_pos_w[frame_idx1]
        
        # interpolate the values

        root_quat = quat_slerp(q0=root_quat_0, q1=root_quat_1, blend=blend)

        blend = blend.unsqueeze(-1)  # make it (n, 1) for broadcasting
        root_pos_w = torch.lerp(root_pos_w_0, root_pos_w_1, blend)
        root_vel_w = torch.lerp(root_vel_w_0, root_vel_w_1, blend)
        root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
        root_ang_vel_w = torch.lerp(root_ang_vel_w_0, root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
        dof_pos = torch.lerp(dof_pos_0, dof_pos_1, blend)
        dof_vel = torch.lerp(dof_vel_0, dof_vel_1, blend)
        key_body_pos_w = torch.lerp(key_body_pos_w_0, key_body_pos_w_1, blend.unsqueeze(1))
        key_body_pos_b = math_utils.quat_apply_inverse(
            root_quat.unsqueeze(1).expand(-1, self.num_key_bodies, -1),
            key_body_pos_w - root_pos_w.unsqueeze(1)
        )

        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_body_pos_b": key_body_pos_b,
        }
        
        
class MotionDataManager(ManagerBase):
    """Manager for motion data.
    
    This manager is responsible for loading and managing motion data terms.
    Each motion data term is responsible for managing a group of data.
    """
    
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        
        # check that cfg is not None
        if cfg is None:
            raise ValueError("MotionDataManager requires a valid configuration object.")
        
        self._terms: dict[str, MotionDataTerm] = {}
        self._term_cfgs: dict[str, MotionDataTermCfg] = {}
        
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for motion data manager."""
        msg = f"<MotionDataManager> contains {len(self._terms)} active terms.\n"
        
        # create table for term information
        table = PrettyTable()
        table.title = "Motion Data Manager Terms"
        table.field_names = ["Index", "Motion Dataset", "Total Duration"]
        # set alignment of table columns
        table.align["Motion Dataset"] = "l"
        table.align["Total Duration"] = "r"
        # add info on each term
        for index, (term_name, term) in enumerate(self._terms.items()):
            table.add_row([index, term_name, term.get_total_duration()])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    
    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    def get_term(self, term_name: str) -> MotionDataTerm:
        """Get the motion data term by name."""
        if term_name not in self._terms:
            raise KeyError(f"Motion data term '{term_name}' not found.")
        return self._terms[term_name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, MotionDataTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MotionDataTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = MotionDataTerm(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg
