# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


def dc_motor_envelope(
    joint_vel: torch.Tensor,
    saturation_effort: torch.Tensor,
    velocity_limit: torch.Tensor,
    max_motor_effort: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upstream four-quadrant bounds, including the peak-torque plateau."""
    corner = velocity_limit * (1.0 + max_motor_effort / saturation_effort)
    velocity = joint_vel.clamp(min=-corner, max=corner)
    upper = torch.minimum(saturation_effort * (1.0 - velocity / velocity_limit), max_motor_effort)
    lower = torch.maximum(saturation_effort * (-1.0 - velocity / velocity_limit), -max_motor_effort)
    return lower, upper


class RoboPartyActuator(DelayedPDActuator):
    """Delayed PD actuator clipped by a measured peak T-N motor envelope.

    ``actuator_effort_limit`` and ``actuator_velocity_limit`` define the motor
    model's peak torque and no-load speed. ``keep_velocity`` is the maximum
    speed that can still provide peak torque. ``rated_effort`` and
    ``rated_velocity`` are retained as continuous operating-point parameters,
    but this actuator does not use them for the current peak hard clip.
    Joint friction should be configured with Isaac Lab's native friction fields.
    """

    cfg: RoboPartyActuatorCfg

    def __init__(self, cfg: RoboPartyActuatorCfg, *args, **kwargs):
        # Keep the explicit actuator's internal hard clip aligned with the
        # simulator peak limit unless a caller intentionally overrides it.
        if cfg.effort_limit is None:
            cfg.effort_limit = cfg.effort_limit_sim
        if cfg.velocity_limit is None:
            cfg.velocity_limit = cfg.velocity_limit_sim
        if not 0.0 <= cfg.rfi_lim <= 1.0:
            raise ValueError(f"RoboPartyActuator rfi_lim must be in [0, 1], got {cfg.rfi_lim}.")
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._rfi_noise = torch.zeros_like(self.computed_effort)
        self._keep_velocity = resolve_joint_parameter(
            cfg.keep_velocity, None, self.joint_names, self._num_envs, self._device
        )
        self._rated_effort = resolve_joint_parameter(
            cfg.rated_effort, None, self.joint_names, self._num_envs, self._device
        )
        self._rated_velocity = resolve_joint_parameter(
            cfg.rated_velocity, None, self.joint_names, self._num_envs, self._device
        )

        if torch.any(self._keep_velocity <= 0.0) or torch.any(
            self._keep_velocity >= self.velocity_limit
        ):
            raise ValueError("RoboPartyActuator keep_velocity must be within (0, velocity_limit(_sim)).")
        if torch.any(self._rated_effort <= 0.0):
            raise ValueError("RoboPartyActuator requires positive rated_effort values.")
        if torch.any(self._rated_effort > self.effort_limit):
            raise ValueError("RoboPartyActuator rated_effort must not exceed effort_limit(_sim).")
        if torch.any(self._rated_velocity <= 0.0) or torch.any(
            self._rated_velocity > self.velocity_limit
        ):
            raise ValueError("RoboPartyActuator rated_velocity must be within (0, velocity_limit(_sim)].")
        self._saturation_effort = self.actuator_effort_limit / (
            1.0 - self._keep_velocity / self.actuator_velocity_limit
        )

    @property
    def rated_effort(self) -> torch.Tensor:
        return self._rated_effort

    @property
    def rated_velocity(self) -> torch.Tensor:
        return self._rated_velocity

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            self._rfi_noise.zero_()
        else:
            self._rfi_noise[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action = super().compute(control_action, joint_pos, joint_vel)
        if self.cfg.rfi_lim > 0.0:
            self._rfi_noise[:] = (
                2.0 * torch.rand_like(self._rfi_noise) - 1.0
            ) * self.cfg.rfi_lim * self.effort_limit
            self.computed_effort += self._rfi_noise
            self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)
            control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor | None = None) -> torch.Tensor:
        if joint_vel is None:
            joint_vel = self._joint_vel
        lower, upper = dc_motor_envelope(
            joint_vel, self._saturation_effort, self.actuator_velocity_limit, self.actuator_effort_limit
        )
        return torch.clip(effort, min=lower, max=upper)


@configclass
class RoboPartyActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for RoboParty explicit actuator dynamics."""

    class_type: type = RoboPartyActuator

    keep_velocity: dict[str, float] | float = MISSING
    """Maximum speed that can still provide peak torque, in rad/s."""

    rated_effort: dict[str, float] | float = MISSING
    """Continuous rated torque of the actuator joints in N-m."""

    rated_velocity: dict[str, float] | float = MISSING
    """Rated speed of the continuous operating point in rad/s."""

    rfi_lim: float = 0.0
    """Per-step random feed-forward torque as a fraction of effort limits."""


# Upstream spelling; retain existing names for saved local configurations.
RobopartyActuator = RoboPartyActuator
RobopartyActuatorCfg = RoboPartyActuatorCfg
