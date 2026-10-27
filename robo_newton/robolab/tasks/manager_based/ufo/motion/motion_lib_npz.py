"""Directory-backed NPZ motion library for the UFO task.

Data management follows the WBC ``MotionStore`` conventions: each NPZ stores
its own metadata, clip boundaries are explicit, all clips share
one fps, quaternions are XYZW, and every read stays inside one clip.

Sampling preserves each source's original non-overlapping window mass when
overlap is enabled. Evaluation can update these weights through the UFO API.

Continuous-time queries blend the two neighbouring frames exactly like the
vendored UFO ``_calc_frame_blend`` path: linear for positions and velocities,
SLERP for quaternions.
"""

from __future__ import annotations

import numpy as np
import torch

from .torch_utils import slerp
from robolab.utils.motion_npz import load_motion, motion_clip_paths, name_indices, resolve_motion_directory
from robolab.utils.newton_motion_fk import fill_full_body_tracks

MOTION_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _window_ranges(frame_count, clip_frames, min_frames, stride):
    start = 0
    while start < frame_count:
        end = min(start + clip_frames, frame_count)
        if start > 0 and end - start < min_frames:
            break
        yield start, end
        if end == frame_count:
            break
        start += stride


class MotionLibNPZ:
    """Serves the UFO motion API from unified XYZW / named NPZ clips."""

    def __init__(self, motion_lib_cfg, num_envs, device, robot=None, env=None):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self._robot = robot
        self._env = env

        self.motion_dir = tuple(
            resolve_motion_directory(entry) for entry in motion_lib_cfg["motion_file"]
        )
        paths = motion_clip_paths(self.motion_dir, motion_lib_cfg.get("motion_names"))
        excluded = set(motion_lib_cfg.get("excluded_motion_names", ()))
        paths = [path for path in paths if path.stem not in excluded]
        if not paths:
            raise ValueError("No motion clips remain after applying the motion selection")
        target_joints = target_bodies = None

        names: list[str] = []
        fps_values: list[float] = []
        arrays: dict[str, list[torch.Tensor]] = {field: [] for field in MOTION_FIELDS}
        for path in paths:
            name = path.stem
            loaded = load_motion(path)
            # Desktop joint_q lacks body_*; Newton FK when scene exists, else URDF.
            if any(field not in loaded for field in MOTION_FIELDS + ("body_names",)):
                loaded = fill_full_body_tracks(
                    loaded, robot=robot, env=env, motion_file=name, device=self.device
                )
            missing = [field for field in MOTION_FIELDS + ("fps", "body_names") if field not in loaded]
            if missing:
                raise ValueError(f"Clip {name!r} is missing NPZ keys: {missing}")
            joint_order = loaded["joint_names"].tolist()
            body_order = loaded["body_names"].tolist()
            if target_joints is None:
                target_joints = list(motion_lib_cfg.get("joint_names", joint_order))
                target_bodies = list(motion_lib_cfg.get("body_names", body_order))
                if not target_bodies or target_bodies[0] != "base_link":
                    raise ValueError("UFO requires base_link at body index zero.")
            joint_indices = name_indices(joint_order, target_joints)
            body_indices = name_indices(body_order, target_bodies)
            fps = float(np.asarray(loaded["fps"]).reshape(-1)[0])
            for field in MOTION_FIELDS:
                value = np.asarray(loaded[field], dtype=np.float32)
                indices = joint_indices if field.startswith("joint") else body_indices
                if indices != list(range(value.shape[1])):
                    value = value[:, indices]
                if field == "body_quat_w":
                    norms = np.linalg.norm(value, axis=-1, keepdims=True)
                    value = value / np.clip(norms, 1e-8, None)
                arrays[field].append(torch.from_numpy(value).to(device=self.device))
            names.append(name)
            fps_values.append(fps)

        if len(set(fps_values)) != 1:
            raise ValueError(f"All clips must share one fps, got {sorted(set(fps_values))}.")
        self._fps = float(fps_values[0])

        num_frames = [int(t.shape[0]) for t in arrays["joint_pos"]]
        if min(num_frames) < 2:
            raise ValueError("Every clip must have at least two frames.")
        starts = np.cumsum([0, *num_frames[:-1]], dtype=np.int64)

        self._num_unique_motions = len(names)
        self._motion_data_keys = np.array(names)
        self._motion_joints = tuple(target_joints)
        self._motion_bodies = tuple(target_bodies)
        self._motion_num_frames = torch.tensor(num_frames, dtype=torch.long, device=self.device)
        self._motion_lengths = (self._motion_num_frames - 1).float() / self._fps
        self._motion_dt = torch.full((self._num_unique_motions,), 1.0 / self._fps, device=self.device)
        self._motion_starts = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self._arrays = {
            field: torch.cat(values, dim=0) for field, values in arrays.items()
        }
        self.all_motions_loaded = True

        self._base_sampling_weights = torch.ones(len(names), device=self.device)
        # Overlap adds boundary coverage. Correct its window-count bias so each
        # source retains its original UFO sampling mass, before difficulty weights.
        clip_seconds = float(motion_lib_cfg.get("clip_seconds", 0) or 0)
        if clip_seconds > 0:
            clip_frames = max(2, int(round(clip_seconds * self._fps)))
            overlap = float(motion_lib_cfg.get("clip_overlap_seconds", 0.0))
            if not 0 <= overlap < clip_seconds:
                raise ValueError("Clip overlap must be non-negative and smaller than clip duration")
            stride = max(1, clip_frames - int(round(overlap * self._fps)))
            min_frames = max(2, int(round(float(motion_lib_cfg.get("min_clip_seconds", 1.0)) * self._fps)))
            win_src, win_start, win_len, win_names = [], [], [], []
            original_counts = []
            for m in range(len(names)):
                n = num_frames[m]
                original_counts.append(sum(1 for _ in _window_ranges(n, clip_frames, min_frames, clip_frames)))
                for start, end in _window_ranges(n, clip_frames, min_frames, stride):
                    win_src.append(m)
                    win_start.append(start)
                    win_len.append(end - start)
                    win_names.append(f"{names[m]}__clip{start // stride:03d}")
            if not win_names:
                raise ValueError("clip windowing produced no windows; check clip_seconds.")
            window_sources = torch.tensor(win_src, dtype=torch.long, device=self.device)
            original_counts = torch.tensor(original_counts, dtype=torch.float32, device=self.device)
            new_counts = torch.bincount(window_sources, minlength=len(names))
            self._base_sampling_weights = (original_counts / new_counts)[window_sources]
            window_starts = torch.tensor(win_start, dtype=torch.long, device=self.device)
            self._motion_starts = self._motion_starts[window_sources] + window_starts
            self._num_unique_motions = len(win_names)
            self._motion_data_keys = np.array(win_names)
            self._motion_num_frames = torch.tensor(win_len, dtype=torch.long, device=self.device)
            self._motion_lengths = (self._motion_num_frames - 1).float() / self._fps
            self._motion_dt = torch.full((self._num_unique_motions,), 1.0 / self._fps, device=self.device)
            print(
                f"[MotionLibNPZ] clip windowing: {len(names)} motions -> "
                f"{self._num_unique_motions} clips of <= {clip_seconds:.1f}s"
            )
        self._default_sampling_prob = (self._base_sampling_weights / self._base_sampling_weights.sum()
                                       if bool((self._base_sampling_weights != 1).any()) else None)
        self._sampling_prob = self._default_sampling_prob
        self.phase_curriculum = None
        if motion_lib_cfg.get("phase_curriculum", False):
            from robolab.utils.motion_phase_curriculum import MotionPhaseCurriculum

            self.phase_curriculum = MotionPhaseCurriculum(self._motion_lengths)

    # ------------------------------------------------------- original UFO API

    def load_motions_for_training(self, max_num_seqs=None):
        # Every clip is loaded eagerly in the constructor.
        print("All motions already loaded!!! No need to resample.")

    def sample_motions(self, n):
        if self._sampling_prob is not None:
            return torch.multinomial(self._sampling_prob, n, replacement=True)
        return torch.randint(0, self._num_unique_motions, (n,), device=self.device)

    def update_sampling_weights(self, probabilities) -> bool:
        """Set per-clip sampling probabilities (original UFO EMD prioritization).

        Accepts non-negative per-window difficulty weights. The source-mass
        correction is applied once here; None restores that baseline.
        """
        if probabilities is None:
            self._sampling_prob = self._default_sampling_prob
            return True
        prob = torch.as_tensor(probabilities, dtype=torch.float32, device=self.device).reshape(-1)
        if (prob.numel() != self._num_unique_motions or not torch.isfinite(prob).all()
                or bool((prob < 0).any())):
            return False
        prob = prob * self._base_sampling_weights
        total = float(prob.sum())
        if total <= 0:
            return False
        self._sampling_prob = prob / total
        return True

    def sample_time(self, motion_ids, truncate_time=None):
        if truncate_time is not None and truncate_time < 0:
            raise ValueError("truncate_time must be non-negative")
        if self.phase_curriculum is not None:
            return self.phase_curriculum.sample_time(motion_ids, truncate_time)
        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len = motion_len - truncate_time
        return (phase * motion_len.clamp_min(0.0)).to(self.device)

    def update_phase_errors(self, motion_id, per_frame_errors, dt):
        if self.phase_curriculum is not None:
            self.phase_curriculum.update(motion_id, per_frame_errors, dt)

    def phase_probabilities(self):
        return None if self.phase_curriculum is None else self.phase_curriculum.probabilities()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.as_tensor(motion_times, dtype=torch.float32, device=self.device).reshape(-1)

        lengths = self._motion_lengths[ids]
        num_frames = self._motion_num_frames[ids]
        dt = self._motion_dt[ids]

        phase = torch.clip(times / lengths, 0.0, 1.0)
        frame0 = (phase * (num_frames - 1).float()).long()
        frame1 = torch.minimum(frame0 + 1, num_frames - 1)
        blend = torch.clip((times - frame0.float() * dt) / dt, 0.0, 1.0).unsqueeze(-1)

        starts = self._motion_starts[ids]
        index0 = starts + frame0
        index1 = starts + frame1
        pos0 = self._arrays["body_pos_w"][index0]
        pos1 = self._arrays["body_pos_w"][index1]
        rot0 = self._arrays["body_quat_w"][index0]
        rot1 = self._arrays["body_quat_w"][index1]
        lin0 = self._arrays["body_lin_vel_w"][index0]
        lin1 = self._arrays["body_lin_vel_w"][index1]
        ang0 = self._arrays["body_ang_vel_w"][index0]
        ang1 = self._arrays["body_ang_vel_w"][index1]
        dof_pos0 = self._arrays["joint_pos"][index0]
        dof_pos1 = self._arrays["joint_pos"][index1]
        dof_vel0 = self._arrays["joint_vel"][index0]
        dof_vel1 = self._arrays["joint_vel"][index1]

        blend_exp = blend.unsqueeze(-1)
        rg_pos_t = (1.0 - blend_exp) * pos0 + blend_exp * pos1
        if offset is not None:
            rg_pos_t = rg_pos_t + offset.unsqueeze(-2)
        rg_rot_t = slerp(rot0, rot1, blend_exp)
        body_vel_t = (1.0 - blend_exp) * lin0 + blend_exp * lin1
        body_ang_vel_t = (1.0 - blend_exp) * ang0 + blend_exp * ang1
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        return {
            "root_pos": rg_pos_t[:, 0].clone(),
            "root_rot": rg_rot_t[:, 0].clone(),
            "root_vel": body_vel_t[:, 0].clone(),
            "root_ang_vel": body_ang_vel_t[:, 0].clone(),
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.clone(),
            "rg_pos": rg_pos_t.clone(),
            "rb_rot": rg_rot_t.clone(),
            "body_vel": body_vel_t.clone(),
            "body_ang_vel": body_ang_vel_t.clone(),
            "rg_pos_t": rg_pos_t.clone(),
            "rg_rot_t": rg_rot_t.clone(),
            "body_vel_t": body_vel_t.clone(),
            "body_ang_vel_t": body_ang_vel_t.clone(),
        }
