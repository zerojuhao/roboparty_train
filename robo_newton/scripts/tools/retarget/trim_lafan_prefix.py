"""Conservatively trim manifest-backed LAFAN NPZ prefixes, with full backup."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if set(arrays) != {*FIELDS, "fps"}:
        raise ValueError(f"Unexpected NPZ schema: {path}")
    fps = arrays["fps"].reshape(-1)
    if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
        raise ValueError(f"Invalid FPS: {path}")
    n = len(arrays["joint_pos"])
    for key in FIELDS:
        if len(arrays[key]) != n or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid frame data: {path}: {key}")
    if n < 2:
        raise ValueError(f"Too few frames: {path}")
    quats = arrays["body_quat_w"]
    if np.max(np.abs(np.linalg.norm(quats, axis=-1) - 1)) > 1e-3:
        raise ValueError(f"Non-unit quaternion: {path}")
    return arrays


def tilt_degrees(quats):
    """Angle between a body's local Z axis and world Z; quaternions are XYZW."""
    q = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(1 - 2 * (q[..., 0] ** 2 + q[..., 1] ** 2), -1, 1)))


def choose_start(arrays, waist_ids, upright_ids, *, limit_deg=3.0,
                 hold_seconds=0.5, max_seconds=2.0, tilt_limit_deg=10.0):
    fps = float(arrays["fps"].reshape(-1)[0])
    # Include both endpoints so the accepted interval spans at least hold_seconds.
    hold = math.ceil(hold_seconds * fps) + 1
    waist = np.rad2deg(arrays["joint_pos"][:, waist_ids])
    tilt = tilt_degrees(arrays["body_quat_w"][:, upright_ids])
    good = (np.abs(waist) <= limit_deg).all(axis=1) & (tilt <= tilt_limit_deg).all(axis=1)
    last = min(math.floor(max_seconds * fps), len(good) - hold)
    for start in range(last + 1):
        if good[start:start + hold].all():
            return start
    return None


def trim_arrays(arrays, start):
    n = len(arrays["joint_pos"])
    if not 0 <= start <= n - 2:
        raise ValueError("Trim must leave at least two frames")
    return {key: value[start:].copy() if key in FIELDS else value.copy()
            for key, value in arrays.items()}


def window_count(n, fps):
    width, minimum = round(10 * fps), round(fps)
    return sum(start == 0 or min(width, n - start) >= minimum
               for start in range(0, n, width))


def inspect(root, policy):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("prefix_trim"):
        raise ValueError("Dataset was already prefix-trimmed; refusing a second trim")
    meta = manifest["npz"]
    if meta["body_quaternion_order"] != "xyzw":
        raise ValueError("Expected XYZW quaternions")
    waist_ids = [meta["joint_order"].index(n) for n in ("waist_roll_joint", "waist_yaw_joint")]
    upright_ids = [meta["body_order"].index(n) for n in ("base_link", "waist_yaw_link")]
    rows = []
    paths = set()
    for record in manifest["clips"]:
        relative = Path(record["npz"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"NPZ path must be relative and contained: {relative}")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path in paths:
            raise ValueError(f"Unsafe or duplicate NPZ path: {path}")
        paths.add(path)
        digest = sha256(path)
        if digest != record["npz_sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {path}")
        arrays = read_arrays(path)
        n = len(arrays["joint_pos"])
        fps = float(arrays["fps"].reshape(-1)[0])
        if n != record["npz_frames"] or fps != meta["output_fps"]:
            raise ValueError(f"Frame count/FPS mismatch: {path}")
        if arrays["joint_pos"].shape != (n, len(meta["joint_order"])):
            raise ValueError(f"Joint shape mismatch: {path}")
        for key in FIELDS:
            expected = (n, len(meta["joint_order"])) if key.startswith("joint") else (
                n, len(meta["body_order"]), 4 if key == "body_quat_w" else 3)
            if arrays[key].shape != expected:
                raise ValueError(f"Shape mismatch: {path}: {key}")
        start = choose_start(arrays, waist_ids, upright_ids, **policy)
        cut = start or 0
        rows.append({
            "name": record["name"], "npz": record["npz"], "source_sha256": digest,
            "status": "manual_review" if start is None else "trimmed" if cut else "unchanged",
            "removed_frames": cut, "removed_seconds": cut / fps,
            "old_frames": n, "new_frames": n - cut,
            "old_waist_deg": np.rad2deg(arrays["joint_pos"][0, waist_ids]).tolist(),
            "new_waist_deg": np.rad2deg(arrays["joint_pos"][cut, waist_ids]).tolist(),
            "new_upright_tilt_deg": tilt_degrees(arrays["body_quat_w"][cut, upright_ids]).tolist(),
            "new_root_position_m": arrays["body_pos_w"][cut, 0].tolist(),
            "old_10s_windows": window_count(n, fps),
            "new_10s_windows": window_count(n - cut, fps),
        })
    if sum(r["old_frames"] for r in rows) != meta["total_frames"]:
        raise ValueError("Manifest total frame count mismatch")
    report = {"source": str(root), "source_manifest_sha256": sha256(root / "manifest.json"),
              "policy": policy, "clips": rows}
    return manifest, report


def apply(root, backup, manifest, report):
    root, backup = root.resolve(), backup.absolute()
    if backup.parent.resolve() != root.parent or backup.exists() or backup.is_symlink():
        raise ValueError("Backup must be a new sibling directory")
    if sha256(root / "manifest.json") != report["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after inspection")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}_trim_stage_", dir=root.parent))
    updated = deepcopy(manifest)
    try:
        shutil.copytree(root, stage, dirs_exist_ok=True, symlinks=True)
        for record, row in zip(updated["clips"], report["clips"], strict=True):
            source = root / row["npz"]
            if sha256(source) != row["source_sha256"]:
                raise ValueError(f"Source changed after inspection: {source}")
            old = read_arrays(source)
            cut = row["removed_frames"]
            target = stage / row["npz"]
            if cut:
                np.savez(target, **trim_arrays(old, cut))
            new = read_arrays(target)
            for key in old:
                expected = old[key][cut:] if key in FIELDS else old[key]
                if new[key].dtype != old[key].dtype or not np.array_equal(new[key], expected):
                    raise ValueError(f"Retained data changed: {source}: {key}")
            record["npz_frames"] = row["new_frames"]
            record["npz_sha256"] = sha256(target)
            record["npz_prefix_trim"] = deepcopy(row)
            row["output_sha256"] = record["npz_sha256"]
            row["all_retained_arrays_exact"] = True
        updated["dataset"] += "_trimmed_prefix_v1"
        updated["npz"]["total_frames"] = sum(r["new_frames"] for r in report["clips"])
        updated["npz"]["center_root_xy"] = False
        report.update({"backup": str(backup), "created_utc": datetime.now(timezone.utc).isoformat(),
                       "note": "All retained arrays are exact source suffixes; no recentering or velocity recomputation."})
        updated["prefix_trim"] = {k: v for k, v in report.items() if k != "clips"}
        updated["prefix_trim"]["source_metadata_note"] = (
            "CSV/BVH paths, checksums, source frame counts, retarget_quality and original root summaries "
            "describe the untrimmed source exports. Current NPZ initial root is in each npz_prefix_trim record. "
            "Source files have not been regenerated."
        )
        (stage / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
        (stage / "prefix_trim_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if sha256(root / "manifest.json") != report["source_manifest_sha256"]:
            raise ValueError("Source manifest changed during staging")
        # Keep the complete original directory, then publish the validated replacement.
        os.rename(root, backup)
        try:
            os.rename(stage, root)
        except BaseException:
            os.rename(backup, root)
            raise
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if args.apply and args.backup is None:
        parser.error("--apply requires an explicit --backup directory")
    policy = dict(limit_deg=3.0, hold_seconds=0.5, max_seconds=2.0, tilt_limit_deg=10.0)
    manifest, report = inspect(args.dataset, policy)
    if args.apply:
        report = apply(args.dataset, args.backup, manifest, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
