from __future__ import annotations

import math

import torch


def heading_slowdown_scale(
    heading_error: torch.Tensor,
    full_speed_angle: float,
    stop_angle: float,
) -> torch.Tensor:
    """Return a linear speed scale that falls from one to zero between two heading errors."""
    if full_speed_angle < 0.0 or stop_angle <= full_speed_angle:
        raise ValueError("heading slowdown angles must satisfy 0 <= full_speed_angle < stop_angle.")
    return torch.clamp(
        (stop_angle - heading_error.abs()) / (stop_angle - full_speed_angle),
        min=0.0,
        max=1.0,
    )


def apply_abs_deadband(values: torch.Tensor, deadband: float) -> torch.Tensor:
    """Set values with magnitude strictly below the deadband to zero."""
    if deadband < 0.0:
        raise ValueError("deadband must be non-negative.")
    return values * (values.abs() >= deadband)


def sample_uniform_with_deadband(
    ranges: torch.Tensor,
    deadband: float,
) -> torch.Tensor:
    """Sample one value per range, then apply an absolute deadband."""
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("ranges must have shape (batch, 2).")
    if torch.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("range upper bounds must be greater than or equal to lower bounds.")
    sampled = ranges[:, 0] + torch.rand(
        len(ranges), device=ranges.device, dtype=ranges.dtype
    ) * (ranges[:, 1] - ranges[:, 0])
    return apply_abs_deadband(sampled, deadband)


def clamp_to_ranges(values: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
    """Clamp each scalar value to its corresponding lower and upper bounds."""
    if values.ndim != 1 or ranges.shape != (len(values), 2):
        raise ValueError("values and ranges must have shapes (batch,) and (batch, 2).")
    if torch.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("range upper bounds must be greater than or equal to lower bounds.")
    return torch.maximum(torch.minimum(values, ranges[:, 1]), ranges[:, 0])


def target_distance_reached(
    target_xy: torch.Tensor,
    root_xy: torch.Tensor,
    distance_threshold: float,
) -> torch.Tensor:
    """Return whether each root is within the inclusive distance threshold of its target."""
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    return torch.norm(target_xy - root_xy, dim=1) <= distance_threshold


def signed_target_heading_offset(
    axis: torch.Tensor,
    signed_speed: torch.Tensor,
) -> torch.Tensor:
    """Return the body-heading offset that points the selected signed body axis at a target.

    Axis values are ``1`` for body x and ``2`` for body y. A zero speed has no
    offset; the caller should hold the current heading while that command is idle.
    """
    if axis.shape != signed_speed.shape:
        raise ValueError("axis and signed_speed must have matching shapes.")
    if torch.any((axis != 1) & (axis != 2)):
        raise ValueError("axis values must be 1 (x) or 2 (y).")

    offset = torch.zeros_like(signed_speed)
    offset[(axis == 1) & (signed_speed < 0.0)] = math.pi
    offset[(axis == 2) & (signed_speed > 0.0)] = -0.5 * math.pi
    offset[(axis == 2) & (signed_speed < 0.0)] = 0.5 * math.pi
    return offset


def signed_target_linear_velocity(
    axis: torch.Tensor,
    signed_speed: torch.Tensor,
    target_distance: torch.Tensor,
    heading_error: torch.Tensor,
    velocity_stiffness: float,
    full_speed_angle: float,
    stop_angle: float,
) -> torch.Tensor:
    """Compute axis-aligned signed commands with distance and heading slowdown."""
    if not (
        axis.shape
        == signed_speed.shape
        == target_distance.shape
        == heading_error.shape
    ):
        raise ValueError("signed-target inputs must have matching shapes.")
    if torch.any((axis != 1) & (axis != 2)):
        raise ValueError("axis values must be 1 (x) or 2 (y).")
    if velocity_stiffness < 0.0:
        raise ValueError("velocity_stiffness must be non-negative.")

    magnitude = torch.minimum(
        target_distance.clamp_min(0.0) * velocity_stiffness,
        signed_speed.abs(),
    )
    magnitude *= heading_slowdown_scale(
        heading_error, full_speed_angle, stop_angle
    )
    command = torch.zeros(len(axis), 2, device=axis.device, dtype=signed_speed.dtype)
    command[axis == 1, 0] = signed_speed[axis == 1].sign() * magnitude[axis == 1]
    command[axis == 2, 1] = signed_speed[axis == 2].sign() * magnitude[axis == 2]
    return command


def select_distant_target_indices(
    targets_xy: torch.Tensor,
    root_xy: torch.Tensor,
    distance_threshold: float,
    random_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose a random target beyond the threshold, or the farthest if none qualify."""
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative.")
    if targets_xy.ndim != 3 or targets_xy.shape[-1] != 2:
        raise ValueError("targets_xy must have shape (batch, targets, 2).")
    if root_xy.shape != (targets_xy.shape[0], 2):
        raise ValueError("root_xy must have shape (batch, 2).")
    if targets_xy.shape[1] == 0:
        raise ValueError("targets_xy must contain at least one candidate per batch.")
    if random_scores is None:
        random_scores = torch.rand(
            targets_xy.shape[:2], device=targets_xy.device, dtype=targets_xy.dtype
        )
    elif random_scores.shape != targets_xy.shape[:2]:
        raise ValueError("random_scores must have shape (batch, targets).")

    distances = torch.norm(targets_xy - root_xy.unsqueeze(1), dim=-1)
    valid = distances > distance_threshold
    random_choice = random_scores.masked_fill(~valid, float("-inf")).argmax(dim=1)
    farthest_choice = distances.argmax(dim=1)
    return torch.where(valid.any(dim=1), random_choice, farthest_choice)

