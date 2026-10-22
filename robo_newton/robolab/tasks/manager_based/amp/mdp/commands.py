"""AMP command generators."""

from collections.abc import Sequence
import math

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _sample_uniform_with_abs_min(
    lower: float, upper: float, count: int, device: torch.device, minimum: float
) -> torch.Tensor:
    """Sample uniformly from a range after removing its low-magnitude interval."""
    if lower > upper:
        raise ValueError("Command range lower bound must not exceed its upper bound.")
    if minimum < 0.0:
        raise ValueError("Minimum command magnitude must be non-negative.")

    negative_end = min(upper, -minimum)
    negative_length = max(0.0, negative_end - lower)
    positive_start = max(lower, minimum)
    positive_length = max(0.0, upper - positive_start)
    total_length = negative_length + positive_length
    if total_length <= 0.0:
        raise ValueError(
            f"Command range ({lower}, {upper}) has no values with magnitude >= {minimum}."
        )

    sample = torch.rand(count, device=device) * total_length
    negative = sample < negative_length
    values = torch.empty(count, device=device)
    values[negative] = lower + sample[negative]
    values[~negative] = positive_start + (sample[~negative] - negative_length)
    return values


class FilteredUniformVelocityCommand(UniformVelocityCommand):
    """Sample all velocity axes uniformly while excluding low magnitudes.

    Each non-degenerate axis follows the original ``UniformVelocityCommand``
    behavior.  The configured absolute-minimum thresholds only remove the
    central low-speed interval from the corresponding axis.

    Standing and single-axis commands are mutually exclusive shares of all
    samples: ``rel_standing_envs`` stand still, ``single_axis_probability``
    keep only one axis, and the remainder keep every sampled axis.
    """

    def __init__(self, cfg: "FilteredUniformVelocityCommandCfg", env):
        super().__init__(cfg, env)
        for name in ("min_abs_lin_velocity", "min_abs_ang_velocity"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0.0 <= cfg.single_axis_probability <= 1.0:
            raise ValueError("single_axis_probability must be in [0, 1].")
        if cfg.rel_standing_envs + cfg.single_axis_probability > 1.0:
            raise ValueError("rel_standing_envs + single_axis_probability must be <= 1.")
        self.segment_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.segment_start_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.segment_duration = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int] | slice | None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(env_ids):
            return
        ranges = (
            self.cfg.ranges.lin_vel_x,
            self.cfg.ranges.lin_vel_y,
            self.cfg.ranges.ang_vel_z,
        )
        active_axes = [axis for axis, (lower, upper) in enumerate(ranges) if upper > lower]
        if not active_axes:
            raise ValueError("FilteredUniformVelocityCommand requires at least one non-degenerate range.")

        self.vel_command_b[env_ids] = 0.0
        for index in active_axes:
            lower, upper = ranges[index]
            minimum = self.cfg.min_abs_ang_velocity if index == 2 else self.cfg.min_abs_lin_velocity
            self.vel_command_b[env_ids, index] = _sample_uniform_with_abs_min(
                lower, upper, len(env_ids), self.device, minimum
            )
        if self.cfg.heading_command:
            heading_samples = torch.empty(len(env_ids), device=self.device)
            self.heading_target[env_ids] = heading_samples.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = (
                heading_samples.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        standing_prob = self.cfg.rel_standing_envs
        single_prob = self.cfg.single_axis_probability
        mode = torch.rand(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = mode < standing_prob
        if single_prob > 0.0 and len(active_axes) > 1:
            single_axis = (mode >= standing_prob) & (mode < standing_prob + single_prob)
            if single_axis.any():
                axis_ids = torch.tensor(active_axes, device=self.device, dtype=torch.long)
                chosen = axis_ids[torch.randint(len(active_axes), (int(single_axis.sum()),), device=self.device)]
                selected = env_ids[single_axis]
                kept = self.vel_command_b[selected].gather(1, chosen.unsqueeze(1)).squeeze(1)
                self.vel_command_b[selected] = 0.0
                self.vel_command_b[selected, chosen] = kept
        self.vel_command_b[env_ids[self.is_standing_env[env_ids]]] = 0.0
        self.segment_start_pos_w[env_ids] = _as_torch(self.robot.data.root_pos_w)[env_ids]
        self.segment_start_heading_w[env_ids] = _as_torch(self.robot.data.heading_w)[env_ids]
        self.segment_duration[env_ids] = self.time_left[env_ids]

    @property
    def _segment_displacement_b(self) -> torch.Tensor:
        """Return segment displacement in the heading frame at command start."""
        root_pos_w = _as_torch(self.robot.data.root_pos_w)
        displacement = root_pos_w[:, :2] - self.segment_start_pos_w[:, :2]
        heading = self.segment_start_heading_w
        displacement_b = torch.empty_like(displacement)
        displacement_b[:, 0] = torch.cos(heading) * displacement[:, 0]
        displacement_b[:, 0] += torch.sin(heading) * displacement[:, 1]
        displacement_b[:, 1] = -torch.sin(heading) * displacement[:, 0]
        displacement_b[:, 1] += torch.cos(heading) * displacement[:, 1]

        elapsed = torch.clamp(self.segment_duration - self.time_left, min=0.0)
        expected_displacement_b = self.vel_command_b[:, :2] * elapsed.unsqueeze(-1)
        return displacement_b - expected_displacement_b

    @property
    def lateral_offset_error(self) -> torch.Tensor:
        """Signed lateral displacement error from the segment's start pose."""
        return self._segment_displacement_b[:, 1]

    @property
    def planar_offset_error(self) -> torch.Tensor:
        """Forward/lateral displacement error from the segment's start pose."""
        return self._segment_displacement_b


@configclass
class FilteredUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type[FilteredUniformVelocityCommand] = FilteredUniformVelocityCommand
    min_abs_lin_velocity: float = 0.0
    """Minimum magnitude for sampled linear velocity commands."""
    min_abs_ang_velocity: float = 0.0
    """Minimum magnitude for sampled angular velocity commands."""
    single_axis_probability: float = 0.1
    """Share of all commands that keep only one sampled axis and zero the rest."""
