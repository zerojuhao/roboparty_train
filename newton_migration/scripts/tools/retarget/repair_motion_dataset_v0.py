#!/usr/bin/env python3
"""Resample RP1 lab motions to control rate, then repair spikes and smooth.

Parkour/AMP uses sim.dt=0.005 and decimation=4, so actions run at 50 Hz.
The loader interpolates by time, but storing 50 fps makes each demo frame
line up with one policy step (disc history of 8 = 8 consecutive actions).

Pipeline (per file):
  1. resample to TARGET_FPS with linear / quat-slerp (no cubic overshoot)
  2. replace impulse-like joint/root jumps
  3. Savitzky-Golay smooth
  4. inset joints that sit on URDF hard limits; clamp to limits
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

TARGET_FPS = 50.0
SG_WINDOW = 5  # 80 ms at 50 Hz; keeps stair contacts
SG_POLY = 2
LIMIT_INSET = 0.01  # rad, ~0.57 deg off the hard stop
HARD_EPS = 1e-4

LAB_DOF = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_yaw_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_joint",
    "right_wrist_joint",
]

REPO = Path("/home/zym/roboparty_train/robolab")
SRC_DIRS = [
    REPO / "data/motions/rp1_24dof_lab",
    REPO / "data/motions/stairs_flat_rp1_24dof_120fps_lab",
]
OUT_DIR = REPO / "data/motions/v0"
URDF = REPO / "data/robots/roboparty/rp1.3/urdf/rp1_24dof.urdf"


def load_urdf_limits(path: Path):
    text = path.read_text()
    blocks = re.findall(r'<joint name="([^"]+)" type="([^"]+)">(.*?)</joint>', text, re.S)
    lim = {}
    for name, _typ, body in blocks:
        m = re.search(
            r'lower="([^"]+)"\s+upper="([^"]+)"\s+effort="([^"]+)"\s+velocity="([^"]+)"',
            body,
        )
        if m:
            lim[name] = (float(m.group(1)), float(m.group(2)), float(m.group(4)))
    lo = np.array([lim[n][0] for n in LAB_DOF])
    hi = np.array([lim[n][1] for n in LAB_DOF])
    vel = np.array([lim[n][2] for n in LAB_DOF])
    return lo, hi, vel


def quat_hemisphere(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] *= -1.0
    return out


def slerp_vec(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """SLERP rows of (w,x,y,z). u shape (n,)."""
    q0 = q0 / np.clip(np.linalg.norm(q0, axis=1, keepdims=True), 1e-12, None)
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=1, keepdims=True), 1e-12, None)
    dot = np.clip(np.sum(q0 * q1, axis=1), -1.0, 1.0)
    q1 = np.where(dot[:, None] < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    u = u[:, None]
    w0 = np.where(small[:, None], 1.0 - u, np.sin((1.0 - u) * theta[:, None]) / np.clip(sin_t[:, None], 1e-12, None))
    w1 = np.where(small[:, None], u, np.sin(u * theta[:, None]) / np.clip(sin_t[:, None], 1e-12, None))
    out = w0 * q0 + w1 * q1
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def resample_linear(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    flat = x.reshape(n, -1)
    out = np.empty((t_out.shape[0], flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_out, t_in, flat[:, j])
    return out.reshape((t_out.shape[0],) + x.shape[1:])


def resample_quat(q: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    q = quat_hemisphere(q)
    idx = np.searchsorted(t_in, t_out, side="right") - 1
    idx = np.clip(idx, 0, len(t_in) - 2)
    span = t_in[idx + 1] - t_in[idx]
    u = np.clip((t_out - t_in[idx]) / np.clip(span, 1e-12, None), 0.0, 1.0)
    return slerp_vec(q[idx], q[idx + 1], u)


def resample_motion(data: dict, fps_out: float) -> dict:
    fps_in = float(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    key_body = np.asarray(data["key_body_pos"], dtype=np.float64)
    n_in = dof_pos.shape[0]
    duration = (n_in - 1) / fps_in
    n_out = max(int(round(duration * fps_out)) + 1, 2)
    t_in = np.linspace(0.0, duration, n_in)
    t_out = np.linspace(0.0, duration, n_out)
    if abs(fps_in - fps_out) < 1e-6 and n_out == n_in:
        return {
            "fps": fps_out,
            "root_pos": root_pos,
            "root_rot": quat_hemisphere(root_rot),
            "dof_pos": dof_pos,
            "key_body_pos": key_body,
            "loop_mode": int(data["loop_mode"]),
        }
    return {
        "fps": fps_out,
        "root_pos": resample_linear(root_pos, t_in, t_out),
        "root_rot": resample_quat(root_rot, t_in, t_out),
        "dof_pos": resample_linear(dof_pos, t_in, t_out),
        "key_body_pos": resample_linear(key_body, t_in, t_out),
        "loop_mode": int(data["loop_mode"]),
    }


def _dilate(mask: np.ndarray, k: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(k):
        nxt = out.copy()
        nxt[1:] |= out[:-1]
        nxt[:-1] |= out[1:]
        out = nxt
    return out


def _keep_short_runs(bad: np.ndarray, max_run: int) -> np.ndarray:
    """Drop contiguous bad blocks longer than max_run; those are motion, not glitches."""
    out = np.zeros_like(bad)
    i, n = 0, len(bad)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i + 1
        while j < n and bad[j]:
            j += 1
        if (j - i) <= max_run:
            out[i:j] = True
        i = j
    return out


def repair_jumps_1d(x: np.ndarray, spike: np.ndarray, max_run: int = 6) -> tuple[np.ndarray, int]:
    """Linear-fill short spike samples. spike is length n-1 (on diffs).

    Long contiguous 'bad' stretches are left untouched. That avoids collapsing
    a fast prefix (e.g. run-then-stop) when a global threshold flags all
    running frames and np.interp clamps them to the first slow sample.
    """
    n = x.shape[0]
    bad = np.zeros(n, dtype=bool)
    bad[1:] |= spike
    bad[:-1] |= spike
    bad = _keep_short_runs(_dilate(bad, 1), max_run=max_run)
    if not bad.any() or bad.all():
        return x, 0
    t = np.arange(n, dtype=np.float64)
    good = ~bad
    x_out = x.copy()
    x_out[bad] = np.interp(t[bad], t[good], x[good])
    return x_out, int(bad.sum())


def repair_jumps(data: dict, vel_lim: np.ndarray) -> tuple[dict, dict]:
    fps = float(data["fps"])
    dt = 1.0 / fps
    q = data["dof_pos"]
    stats = {"dof_bad": 0, "root_bad": 0, "kb_bad": 0, "joints": []}
    q_new = q.copy()
    dq = np.abs(np.diff(q, axis=0))
    for j in range(q.shape[1]):
        v = dq[:, j]
        pad = np.pad(v, 1, mode="edge")
        neigh = np.maximum(pad[:-2], pad[2:])
        ratio = v / np.maximum(neigh, 1e-6)
        spike = ((v > 0.10) & (ratio > 3.0)) | (v > 0.18) | (v / dt > 1.2 * vel_lim[j])
        if not spike.any():
            continue
        q_new[:, j], n_bad = repair_jumps_1d(q[:, j], spike)
        stats["dof_bad"] += n_bad
        stats["joints"].append((LAB_DOF[j], n_bad, float(v.max()), float((v / dt).max())))

    root = data["root_pos"].copy()
    step = np.linalg.norm(np.diff(root, axis=0), axis=1)
    pad = np.pad(step, 1, mode="edge")
    root_ratio = step / np.maximum(np.maximum(pad[:-2], pad[2:]), 1e-6)
    # Teleport only: >12 cm/frame (~6 m/s at 50 Hz) and much larger than neighbors.
    # Do not use a global median — a long standing tail would flag the whole run.
    root_spike = (step > 0.12) & (root_ratio > 4.0)
    if root_spike.any():
        for j in range(3):
            root[:, j], n_bad = repair_jumps_1d(root[:, j], root_spike)
            stats["root_bad"] += n_bad

    kb = data["key_body_pos"].copy()
    # World-frame key bodies ride the root; subtract root so running translation
    # is not treated as a foot teleport.
    rel = kb - root[:, None, :]
    drel = np.linalg.norm(np.diff(rel, axis=0), axis=2)
    kb_spike = np.zeros(drel.shape[0], dtype=bool)
    for b in range(drel.shape[1]):
        v = drel[:, b]
        pad_b = np.pad(v, 1, mode="edge")
        ratio_b = v / np.maximum(np.maximum(pad_b[:-2], pad_b[2:]), 1e-6)
        kb_spike |= (v > 0.08) & (ratio_b > 4.0)
    if kb_spike.any():
        for b in range(kb.shape[1]):
            for j in range(3):
                kb[:, b, j], n_bad = repair_jumps_1d(kb[:, b, j], kb_spike)
                stats["kb_bad"] += n_bad

    data = dict(data)
    data["dof_pos"] = q_new
    data["root_pos"] = root
    data["key_body_pos"] = kb
    return data, stats


def savgol_nd(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = x.shape[0]
    w = window if window % 2 == 1 else window + 1
    if n < w:
        w = n if n % 2 == 1 else n - 1
    if w < poly + 2:
        return x
    flat = x.reshape(n, -1)
    out = savgol_filter(flat, window_length=w, polyorder=poly, axis=0, mode="interp")
    return out.reshape(x.shape)


def smooth_motion(data: dict) -> dict:
    data = dict(data)
    data["dof_pos"] = savgol_nd(data["dof_pos"], SG_WINDOW, SG_POLY)
    data["root_pos"] = savgol_nd(data["root_pos"], SG_WINDOW, SG_POLY)
    data["key_body_pos"] = savgol_nd(data["key_body_pos"], SG_WINDOW, SG_POLY)
    q = quat_hemisphere(data["root_rot"])
    q = savgol_nd(q, SG_WINDOW, SG_POLY)
    data["root_rot"] = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    return data


def clamp_and_inset(data: dict, lo: np.ndarray, hi: np.ndarray) -> tuple[dict, list[str]]:
    q = data["dof_pos"].copy()
    notes = []
    rng = q.max(axis=0) - q.min(axis=0)
    at_lo = np.abs(q.mean(axis=0) - lo) < 0.02
    at_hi = np.abs(q.mean(axis=0) - hi) < 0.02
    locked = rng < 0.01
    for j, name in enumerate(LAB_DOF):
        if locked[j] and (at_lo[j] or at_hi[j]):
            interior = lo[j] + LIMIT_INSET if at_lo[j] else hi[j] - LIMIT_INSET
            q[:, j] = interior
            notes.append(f"{name}: locked at limit -> inset {interior:.3f}")
    q = np.clip(q, lo + LIMIT_INSET, hi - LIMIT_INSET)
    data = dict(data)
    data["dof_pos"] = q
    return data, notes


def collect_inputs() -> list[Path]:
    seen = set()
    files = []
    for d in SRC_DIRS:
        for p in sorted(d.glob("*.pkl")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)
    return files


def to_lab_dict(data: dict) -> dict:
    return {
        "fps": float(data["fps"]),
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "root_rot": np.asarray(data["root_rot"], dtype=np.float32),
        "dof_pos": np.asarray(data["dof_pos"], dtype=np.float32),
        "loop_mode": int(data["loop_mode"]),
        "key_body_pos": np.asarray(data["key_body_pos"], dtype=np.float32),
    }


def main():
    lo, hi, vel_lim = load_urdf_limits(URDF)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_inputs()
    print(f"Output: {OUT_DIR}")
    print(f"Target fps: {TARGET_FPS:g}  (action rate = 1/(0.005*4))")
    print(f"Unique source files: {len(files)}\n")

    rows = []
    for src in files:
        with open(src, "rb") as f:
            raw = pickle.load(f)
        fps0 = float(raw["fps"])
        n0 = np.asarray(raw["dof_pos"]).shape[0]
        data = resample_motion(raw, TARGET_FPS)
        n1 = data["dof_pos"].shape[0]
        data, jump_stats = repair_jumps(data, vel_lim)
        data = smooth_motion(data)
        data, inset_notes = clamp_and_inset(data, lo, hi)
        out = to_lab_dict(data)
        dst = OUT_DIR / src.name
        with open(dst, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        peak_v = float(np.abs(np.diff(out["dof_pos"], axis=0)).max() * TARGET_FPS)
        rows.append(
            (
                src.name,
                fps0,
                n0,
                n1,
                jump_stats["dof_bad"],
                ",".join(j for j, *_ in jump_stats["joints"]) or "-",
                "; ".join(inset_notes) or "-",
                peak_v,
            )
        )
        print(
            f"{src.name:<52} {fps0:5.0f}->{TARGET_FPS:g}  "
            f"N {n0:4d}->{n1:4d}  jump_frames={jump_stats['dof_bad']:3d}  "
            f"peak_vel={peak_v:5.2f}"
        )
        if jump_stats["joints"]:
            for name, n_bad, dq, v in jump_stats["joints"]:
                print(f"    jump {name}: {n_bad} frames  maxΔq={dq:.3f}  maxvel={v:.1f}")
        if inset_notes:
            for note in inset_notes:
                print(f"    {note}")

    print(f"\nWrote {len(rows)} motions to {OUT_DIR}")


if __name__ == "__main__":
    main()
