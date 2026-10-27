"""Physical tracking diagnostics without assigning an acrobatic success label."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _aligned_kinematics(
    body_position,
    root_quaternion,
    body_link_velocity,
    target_body_position,
    target_root_quaternion,
    target_body_link_velocity,
):
    positions, velocities, targets, target_velocities = (
        np.asarray(value, dtype=np.float64)
        for value in (body_position, body_link_velocity, target_body_position, target_body_link_velocity)
    )
    quaternions = np.asarray(root_quaternion, dtype=np.float64)
    target_quaternions = np.asarray(target_root_quaternion, dtype=np.float64)
    shape = positions.shape
    if len(shape) != 3 or shape[0] < 2 or shape[1] < 1 or shape[2] != 3:
        raise ValueError("Tracking kinematics require [time >= 2, bodies >= 1, xyz] positions")
    if any(value.shape != shape for value in (velocities, targets, target_velocities)):
        raise ValueError("Tracking kinematics must share time and body ordering")
    if quaternions.shape != (shape[0], 4) or target_quaternions.shape != quaternions.shape:
        raise ValueError("Tracking root quaternions must be [time, xyzw]")
    if not all(np.isfinite(value).all() for value in (
        positions, velocities, targets, target_velocities, quaternions, target_quaternions
    )):
        raise FloatingPointError("Tracking kinematics contain non-finite values")
    actual_rotation = Rotation.from_quat(quaternions)
    target_rotation = Rotation.from_quat(target_quaternions)
    actual_initial = actual_rotation[0].as_matrix()
    target_initial = target_rotation[0].as_matrix()
    yaw = (np.arctan2(actual_initial[1, 0], actual_initial[0, 0]) -
           np.arctan2(target_initial[1, 0], target_initial[0, 0]))
    alignment = Rotation.from_rotvec([0.0, 0.0, yaw])
    aligned_targets = alignment.apply(targets.reshape(-1, 3)).reshape(shape)
    aligned_targets[..., :2] += positions[0, 0, :2] - aligned_targets[0, 0, :2]
    aligned_velocities = alignment.apply(target_velocities.reshape(-1, 3)).reshape(shape)
    aligned_rotation = alignment * target_rotation

    return (positions, velocities, aligned_targets, aligned_velocities,
            actual_rotation, target_rotation, aligned_rotation)


def tracking_phase_errors(
    body_position, root_quaternion, body_link_velocity,
    target_body_position, target_root_quaternion, target_body_link_velocity,
):
    """Dimensionless, time-aligned curriculum errors; never used as rewards.

    Average body-position / 0.3 m, root-height / 0.3 m and rotation / 0.4 rad
    errors. The fixed scales keep all three physical quantities influential.
    """
    positions, _, targets, _, actual_rotation, _, aligned_rotation = _aligned_kinematics(
        body_position, root_quaternion, body_link_velocity,
        target_body_position, target_root_quaternion, target_body_link_velocity,
    )
    position_error = np.linalg.norm(positions - targets, axis=-1).mean(axis=-1)
    height_error = np.abs(positions[:, 0, 2] - targets[:, 0, 2])
    rotation_error = (actual_rotation.inv() * aligned_rotation).magnitude()
    return (position_error / 0.3 + height_error / 0.3 + rotation_error / 0.4) / 3.0


def tracking_kinematics_metrics(
    body_position, root_quaternion, body_link_velocity,
    target_body_position, target_root_quaternion, target_body_link_velocity,
    *, contact_free=None, clearance_threshold_m=0.1,
):
    """Measure physical tracking after only initial heading/XY alignment.

    Inputs use XYZW root quaternions and root body at index zero. Height and
    later pose drift remain visible. Body-origin clearance does not establish
    collision-surface clearance or flight; contact_free measures forces below
    1 N over the sensor history after each real step.
    """
    if not np.isfinite(clearance_threshold_m) or clearance_threshold_m <= 0:
        raise ValueError("Body-origin clearance threshold must be finite and positive")
    positions, velocities, aligned_targets, aligned_velocities, actual_rotation, target_rotation, aligned_rotation = (
        _aligned_kinematics(body_position, root_quaternion, body_link_velocity,
                            target_body_position, target_root_quaternion, target_body_link_velocity)
    )
    shape = positions.shape

    rotation_error = (actual_rotation.inv() * aligned_rotation).magnitude()
    actual_tilt = np.arccos(np.clip(actual_rotation.as_matrix()[:, 2, 2], -1.0, 1.0))
    target_tilt = np.arccos(np.clip(target_rotation.as_matrix()[:, 2, 2], -1.0, 1.0))
    position_error = positions - aligned_targets
    relative_error = ((positions - positions[:, :1]) -
                      (aligned_targets - aligned_targets[:, :1]))
    vector_rmse = lambda error: float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))
    metrics = {
        "root_xy_rmse_m": vector_rmse(position_error[:, 0, :2]),
        "root_height_rmse_m": float(np.sqrt(np.mean(position_error[:, 0, 2] ** 2))),
        "root_rotation_mean_rad": float(rotation_error.mean()),
        "root_rotation_p95_rad": float(np.quantile(rotation_error, 0.95)),
        "root_tilt_mae_rad": float(np.abs(actual_tilt - target_tilt).mean()),
        "root_tilt_max_rad": float(actual_tilt.max()),
        "reference_root_tilt_max_rad": float(target_tilt.max()),
        "body_position_rmse_m": vector_rmse(position_error),
        "body_root_relative_position_rmse_m": vector_rmse(relative_error),
        "body_link_velocity_rmse_m_s": vector_rmse(velocities - aligned_velocities),
        "body_origin_clearance_threshold_m": float(clearance_threshold_m),
    }
    for name, actual, reference in (
        ("inverted", actual_tilt > np.pi / 2.0, target_tilt > np.pi / 2.0),
        ("body_origin_clearance", positions[:, :, 2].min(axis=1) > clearance_threshold_m,
         aligned_targets[:, :, 2].min(axis=1) > clearance_threshold_m),
    ):
        metrics[f"{name}_fraction"] = float(actual.mean())
        metrics[f"reference_{name}_fraction"] = float(reference.mean())
        metrics[f"{name}_overlap_fraction"] = float((actual & reference).mean())
        metrics[f"{name}_mismatch_fraction"] = float((actual != reference).mean())
    if contact_free is not None:
        contact_free = np.asarray(contact_free)
        if contact_free.shape != (shape[0] - 1,) or contact_free.dtype != np.bool_:
            raise ValueError("Measured contact_free must be boolean with one entry per simulated step")
        metrics["contact_force_below_1n_fraction"] = float(contact_free.mean())
    return metrics


def write_tracking_kinematics(log_dir, global_step, motion_keys, clips):
    """Keep physical per-clip detail separately from the legacy joint EMD CSV."""
    if not clips:
        return
    expected = set(next(iter(clips.values())))
    if any(set(values) != expected for values in clips.values()):
        raise ValueError("Tracking kinematics metrics differ between clips")
    metric_keys = sorted(expected)
    metric_keys = [key for key in metric_keys if key not in ("emd", "obs_state_emd")]
    if not metric_keys:
        return
    path = Path(log_dir) / "tracking_kinematics_clips.csv"
    header = ["motion_name", "motion_id", "timestep", *metric_keys]
    exists = path.exists()
    if exists:
        with path.open(newline="") as stream:
            if next(csv.reader(stream), None) != header:
                raise ValueError("Tracking kinematics CSV schema differs from the current evaluator")
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(header)
        for motion_id in sorted(clips):
            writer.writerow([str(motion_keys[motion_id]), motion_id, global_step,
                             *(clips[motion_id][key] for key in metric_keys)])
