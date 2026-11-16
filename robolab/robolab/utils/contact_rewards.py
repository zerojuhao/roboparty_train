"""Shared stumble reduction for terrain tasks."""

import math

import torch


def feet_stumble(env, sensor_cfg, horizontal_ratio=1.0, force_threshold=20.0,
                 force_scale=100.0, max_penalty=1.0):
    """Maximum individual-contact penalty over one complete policy step.

    PhysX retains its body-net-force fallback. Newton terrain tasks supply
    per-contact maxima from TerrainContactSensor instead.
    """
    if not all(math.isfinite(v) for v in (horizontal_ratio, force_threshold, force_scale, max_penalty)):
        raise ValueError("Stumble parameters must be finite")
    if horizontal_ratio < 0 or force_threshold < 0 or force_scale <= 0 or max_penalty <= 0:
        raise ValueError("Invalid stumble ratio, threshold, scale or maximum")
    sensor = env.scene[sensor_cfg.name]
    data = sensor.data
    history = getattr(data, "contact_excess_history", None)
    if history is not None:
        if horizontal_ratio != sensor.stumble_horizontal_ratio:
            raise ValueError("Reward horizontal_ratio must match the terrain contact sensor")
        if history.ndim != 3 or history.shape[1] != env.cfg.decimation:
            raise ValueError("Stumble history must cover exactly one complete policy step")
        excess = history[:, :, sensor_cfg.body_ids]
    else:
        from isaaclab_newton.sensors.contact_sensor.contact_sensor import ContactSensor
        if isinstance(sensor, ContactSensor):
            raise RuntimeError("Newton stumble requires the shared terrain contact sensor")
        forces = data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        excess = torch.linalg.vector_norm(forces[..., :2], dim=-1) - horizontal_ratio * forces[..., 2].abs()
    if excess.shape[2] == 0:
        raise ValueError("Stumble must select at least one body")
    penalty = ((excess - force_threshold).clamp(min=0) / force_scale).clamp(max=max_penalty)
    # Preserve invalid physics signals for the environment's finite-value checks.
    return torch.where(torch.isfinite(excess), penalty, excess).amax(dim=(1, 2))
