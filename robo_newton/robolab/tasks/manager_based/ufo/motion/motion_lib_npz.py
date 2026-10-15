"""Manifest-backed NPZ motion library for the UFO task.

Data management follows the WBC ``MotionStore`` conventions: the clip manifest
is the single source of truth, clip boundaries are explicit, all clips share
one fps, quaternions are XYZW, and every read stays inside one clip.

Sampling uses uniform clip weights initially and a uniform phase start time.
Evaluation can update clip weights through the original UFO priority API.

Continuous-time queries blend the two neighbouring frames exactly like the
vendored UFO ``_calc_frame_blend`` path: linear for positions and velocities,
SLERP for quaternions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .torch_utils import slerp

MOTION_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class MotionLibNPZ:
    """Serves the UFO motion API from a BeyondMimic-format NPZ dataset."""

    def __init__(self, motion_lib_cfg, num_envs, device):
        self.device = torch.device(device)
        self.num_envs = num_envs

        manifest_path = Path(str(motion_lib_cfg["motion_file"])).expanduser()
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Motion manifest does not exist: {manifest_path}")
        self.manifest_path = manifest_path

        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        records = manifest.get("clips")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Manifest must contain a non-empty 'clips' list: {manifest_path}")
        selected = motion_lib_cfg.get("motion_names")
        if selected is not None:
            selected = set(selected)
            missing = selected - {str(record.get("name")) for record in records}
            if not selected or missing:
                raise ValueError(f"Invalid motion selection; missing names: {sorted(missing)}")
            records = [record for record in records if str(record.get("name")) in selected]

        npz_meta = manifest.get("npz", {})
        if str(npz_meta.get("body_quaternion_order", "xyzw")).lower() != "xyzw":
            raise ValueError("MotionLibNPZ expects XYZW quaternions in the NPZ.")
        joint_order = npz_meta.get("joint_order")
        if joint_order is None:
            raise ValueError("Manifest 'npz.joint_order' is required to map motions to the robot.")
        expected_fps = npz_meta.get("output_fps")

        names: list[str] = []
        fps_values: list[float] = []
        arrays: dict[str, list[torch.Tensor]] = {field: [] for field in MOTION_FIELDS}
        for clip_id, record in enumerate(records):
            name = str(record.get("name", clip_id))
            raw_path = record.get("npz", record.get("path"))
            if raw_path is None:
                raise ValueError(f"Clip {name!r} has no 'npz' path in the manifest.")
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"NPZ for clip {name!r} does not exist: {path}")

            with np.load(path, allow_pickle=False) as loaded:
                missing = [field for field in MOTION_FIELDS + ("fps",) if field not in loaded]
                if missing:
                    raise ValueError(f"Clip {name!r} is missing NPZ keys: {missing}")
                fps_array = np.asarray(loaded["fps"]).reshape(-1)
                if fps_array.size != 1 or not np.isfinite(fps_array[0]):
                    raise ValueError(f"Clip {name!r} has invalid fps metadata.")
                fps = float(fps_array[0])
                if expected_fps is not None and abs(fps - float(expected_fps)) > 1e-3:
                    raise ValueError(
                        f"Clip {name!r} fps={fps} does not match manifest output_fps={expected_fps}."
                    )
                for field in MOTION_FIELDS:
                    value = np.asarray(loaded[field], dtype=np.float32)
                    if not np.isfinite(value).all():
                        raise ValueError(f"Clip {name!r}: {field} contains NaN/Inf.")
                    if field == "body_quat_w":
                        norms = np.linalg.norm(value, axis=-1, keepdims=True)
                        if np.abs(norms - 1.0).max() > 1e-2:
                            raise ValueError(f"Clip {name!r}: body_quat_w is not close to unit norm.")
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
        self._motion_joints = tuple(str(name) for name in joint_order)
        self._motion_num_frames = torch.tensor(num_frames, dtype=torch.long, device=self.device)
        self._motion_lengths = (self._motion_num_frames - 1).float() / self._fps
        self._motion_dt = torch.full((self._num_unique_motions,), 1.0 / self._fps, device=self.device)
        self._motion_starts = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self._arrays = {
            field: torch.cat(values, dim=0) for field, values in arrays.items()
        }
        self.all_motions_loaded = True

        # Original-UFO clip windowing (humanoidverse utils/motion_data/clip.py):
        # each source motion is split into fixed-duration non-overlapping
        # windows (clip_seconds=10, min_clip_seconds=1). The public motion API
        # then operates on window entries, exactly like the original 902-clip
        # training/eval granularity.
        clip_seconds = float(motion_lib_cfg.get("clip_seconds", 0) or 0)
        if clip_seconds > 0:
            clip_frames = max(2, int(round(clip_seconds * self._fps)))
            min_frames = max(2, int(round(float(motion_lib_cfg.get("min_clip_seconds", 1.0)) * self._fps)))
            win_src, win_start, win_len, win_names = [], [], [], []
            for m in range(len(names)):
                n = num_frames[m]
                start = 0
                while start < n:
                    end = min(start + clip_frames, n)
                    if start > 0 and (end - start) < min_frames:
                        break  # drop tiny tails; the first window is always kept
                    win_src.append(m)
                    win_start.append(start)
                    win_len.append(end - start)
                    win_names.append(f"{names[m]}__clip{start // clip_frames:03d}")
                    start = end
            if not win_names:
                raise ValueError("clip windowing produced no windows; check clip_seconds.")
            window_sources = torch.tensor(win_src, dtype=torch.long, device=self.device)
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
        self._sampling_prob = None

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

        Accepts an unnormalized non-negative probability vector of length
        num clips; uniform sampling is restored by passing None.
        """
        if probabilities is None:
            self._sampling_prob = None
            return True
        prob = torch.as_tensor(probabilities, dtype=torch.float32, device=self.device).reshape(-1)
        if prob.numel() != self._num_unique_motions or not torch.isfinite(prob).all():
            return False
        total = float(prob.sum())
        if total <= 0:
            return False
        self._sampling_prob = prob / total
        return True

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self.device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len = motion_len - truncate_time
        return (phase * motion_len).to(self.device)

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
