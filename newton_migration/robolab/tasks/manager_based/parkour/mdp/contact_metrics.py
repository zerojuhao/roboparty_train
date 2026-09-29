# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-force metrics shared by parkour rewards and foothold labels."""

from __future__ import annotations

import torch


def horizontal_force_excess(
    forces_w: torch.Tensor,
    horizontal_ratio: float,
    force_threshold: float,
) -> torch.Tensor:
    """Horizontal force above the scaled vertical force and noise floor."""
    force_xy = torch.linalg.norm(forces_w[..., :2], dim=-1)
    force_z = torch.abs(forces_w[..., 2])
    return torch.relu(force_xy - horizontal_ratio * force_z - force_threshold)


def vertical_contact_mask(
    forces_w: torch.Tensor,
    vertical_force_ratio: float,
    vertical_force_min: float,
) -> torch.Tensor:
    """Whether contact is strong enough and dominated by vertical force."""
    force_xy = torch.linalg.norm(forces_w[..., :2], dim=-1)
    force_z = torch.abs(forces_w[..., 2])
    return (force_z >= vertical_force_min) & (
        force_z > vertical_force_ratio * force_xy
    )


def capped_swing_clearance_reward(
    clearance: torch.Tensor,
    contact_time: torch.Tensor,
    command: torch.Tensor,
    target_clearance: float,
    angular_velocity_threshold: float,
) -> torch.Tensor:
    """Reward single-support swing-foot clearance, capped at the target height."""
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be positive.")
    if angular_velocity_threshold < 0.0:
        raise ValueError("angular_velocity_threshold must be non-negative.")
    if clearance.shape != contact_time.shape or clearance.ndim != 2:
        raise ValueError(
            "clearance and contact_time must have matching [N, feet] shapes."
        )
    if (
        command.ndim != 2
        or command.shape[0] != clearance.shape[0]
        or command.shape[1] < 3
    ):
        raise ValueError("command must have shape [N, >=3].")

    in_contact = contact_time > 0.0
    single_stance = in_contact.sum(dim=1) == 1
    swing_foot = ~in_contact
    clearance_score = torch.clamp(
        clearance / target_clearance, min=0.0, max=1.0
    )
    swing_score = torch.sum(clearance_score * swing_foot.float(), dim=1)

    turning = torch.abs(command[:, 2]) > angular_velocity_threshold
    active = turning & single_stance
    return swing_score * active.float()


def directional_joint_lift_reward(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    swing_foot: torch.Tensor,
    flexion_signs: tuple[float, ...],
    target_flexion: float,
) -> torch.Tensor:
    """Reward directional hip/knee flexion for the currently swinging foot.

    ``flexion_signs`` maps each selected joint to its positive flexion direction.
    The output is the mean score over selected joints, capped at the target
    angle, so larger joint excursions do not increase the reward indefinitely.
    """
    if joint_pos.ndim != 2 or default_joint_pos.shape != joint_pos.shape:
        raise ValueError("joint_pos and default_joint_pos must have shape [N, joints].")
    if swing_foot.ndim != 2 or swing_foot.shape[0] != joint_pos.shape[0]:
        raise ValueError("swing_foot must have shape [N, feet].")
    if joint_pos.shape[1] != swing_foot.shape[1]:
        raise ValueError("The selected joints must correspond one-to-one with feet.")
    if len(flexion_signs) != joint_pos.shape[1]:
        raise ValueError("flexion_signs must match the number of selected joints.")
    if target_flexion <= 0.0:
        raise ValueError("target_flexion must be positive.")

    signs = torch.as_tensor(
        flexion_signs, dtype=joint_pos.dtype, device=joint_pos.device
    )
    flexion = torch.relu((joint_pos - default_joint_pos) * signs)
    score = torch.clamp(flexion / target_flexion, min=0.0, max=1.0)
    return torch.sum(score * swing_foot.to(score.dtype), dim=1) / swing_foot.shape[1]


def bounded_flight_phase_reward(
    air_time: torch.Tensor,
    contact_time: torch.Tensor,
    command_xy: torch.Tensor,
    velocity_xy: torch.Tensor,
    command_speed_threshold: float,
    flight_time_range: tuple[float, float],
    tracking_std: float,
    min_tracking_score: float,
    transition_time: float,
) -> torch.Tensor:
    """Reward a bounded two-foot flight phase while tracking planar velocity.

    The duration score ramps up just inside the lower bound and ramps down just
    before the upper bound. Consequently, longer or more violent jumps cannot
    increase the reward without limit.
    """
    flight_time_min, flight_time_max = flight_time_range
    if command_speed_threshold < 0.0:
        raise ValueError("command_speed_threshold must be non-negative.")
    if flight_time_min < 0.0 or flight_time_max <= flight_time_min:
        raise ValueError("flight_time_range must satisfy 0 <= min < max.")
    if transition_time <= 0.0:
        raise ValueError("transition_time must be positive.")
    if 2.0 * transition_time > flight_time_max - flight_time_min:
        raise ValueError("transition_time is too large for flight_time_range.")
    if tracking_std <= 0.0:
        raise ValueError("tracking_std must be positive.")
    if not 0.0 <= min_tracking_score < 1.0:
        raise ValueError("min_tracking_score must be in [0, 1).")
    if air_time.shape != contact_time.shape or air_time.ndim != 2:
        raise ValueError("air_time and contact_time must have matching [N, feet] shapes.")
    if command_xy.shape != velocity_xy.shape or command_xy.shape != (air_time.shape[0], 2):
        raise ValueError("command_xy and velocity_xy must have shape [N, 2].")

    both_airborne = torch.all(contact_time <= 0.0, dim=1)
    shared_air_time = torch.amin(air_time, dim=1)
    ramp_up = torch.clamp(
        (shared_air_time - flight_time_min) / transition_time, min=0.0, max=1.0
    )
    ramp_down = torch.clamp(
        (flight_time_max - shared_air_time) / transition_time,
        min=0.0,
        max=1.0,
    )
    duration_score = ramp_up * ramp_down

    tracking_error = torch.sum(torch.square(command_xy - velocity_xy), dim=1)
    tracking_score = torch.exp(-tracking_error / tracking_std**2)
    tracking_gate = tracking_score >= min_tracking_score
    command_gate = torch.linalg.norm(command_xy, dim=1) >= command_speed_threshold
    return (
        duration_score
        * tracking_score
        * both_airborne.float()
        * tracking_gate.float()
        * command_gate.float()
    )
