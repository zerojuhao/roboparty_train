"""AMP command generators."""

from collections.abc import Sequence
import math

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def validate_braking_duration_range(duration, name="braking_duration_range"):
    if duration is not None and (
        len(duration) != 2 or not all(math.isfinite(x) for x in duration)
        or not 0.0 < duration[0] <= duration[1]
    ):
        raise ValueError(f"{name} must be finite, positive and ordered.")


class SingleAxisVelocityCommand(UniformVelocityCommand):
    """Sample one non-degenerate velocity axis per environment.

    Axes with a zero-width range are ignored. This keeps a disabled lateral
    command from turning an otherwise moving command into an all-zero sample.
    """

    def __init__(self, cfg: "SingleAxisVelocityCommandCfg", env):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.stop_sequence_probability <= 1.0:
            raise ValueError("stop_sequence_probability must be in [0, 1].")
        if cfg.stop_min_speed <= 0.0:
            raise ValueError("stop_min_speed must be positive.")
        for duration in (cfg.stop_cruise_time_range, cfg.stop_hold_time_range):
            if not 0.0 < duration[0] <= duration[1]:
                raise ValueError("Stop sequence durations must be positive ordered ranges.")
        validate_braking_duration_range(cfg.braking_duration_range)
        validate_braking_duration_range(cfg.acceleration_duration_range, "acceleration_duration_range")
        validate_braking_duration_range(cfg.stop_start_time_range, "stop_start_time_range")
        if not 0.0 <= cfg.stop_peak_speed_probability <= 1.0:
            raise ValueError("stop_peak_speed_probability must be in [0, 1].")
        self._stop_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._start_pending = torch.zeros_like(self._stop_pending)
        self._sequence_speed = torch.zeros(self.num_envs, device=self.device)
        self._braking_start_velocity = torch.zeros(self.num_envs, 2, device=self.device)
        self._braking_duration = torch.zeros(self.num_envs, device=self.device)
        self.segment_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.segment_start_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.segment_duration = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(env_ids):
            return
        # reset() clears command_counter before resampling: never resume a stale
        # cruise-to-stop transition after a fall or episode timeout.
        stopping = self._stop_pending[env_ids] & (self.command_counter[env_ids] > 0)
        launching = self._start_pending[env_ids] & (self.command_counter[env_ids] > 0)
        self._stop_pending[env_ids] = False
        self._start_pending[env_ids] = False
        ranges = (
            self.cfg.ranges.lin_vel_x,
            self.cfg.ranges.lin_vel_y,
            self.cfg.ranges.ang_vel_z,
        )
        active_axes = [axis for axis, (lower, upper) in enumerate(ranges) if upper > lower]
        if not active_axes:
            raise ValueError("SingleAxisVelocityCommand requires at least one non-degenerate range.")

        self.vel_command_b[env_ids] = 0.0
        active_axes_tensor = torch.tensor(active_axes, device=self.device, dtype=torch.long)
        axis = active_axes_tensor[torch.randint(len(active_axes), (len(env_ids),), device=self.device)]
        for index in active_axes:
            selected_mask = axis == index
            selected = env_ids[selected_mask]
            if len(selected):
                lower, upper = ranges[index]
                self.vel_command_b[selected, index] = torch.empty(
                    len(selected), device=self.device
                ).uniform_(lower, upper)
        self.is_standing_env[env_ids] = (
            torch.rand(len(env_ids), device=self.device) <= self.cfg.rel_standing_envs
        )
        if self.cfg.stop_sequence_probability > 0.0:
            upper_speed = self.cfg.ranges.lin_vel_x[1]
            if upper_speed >= self.cfg.stop_min_speed:
                cruise = ~stopping & ~launching & (
                    torch.rand(len(env_ids), device=self.device) < self.cfg.stop_sequence_probability
                )
                cruise_ids = env_ids[cruise]
                self.vel_command_b[cruise_ids] = 0.0
                self.vel_command_b[cruise_ids, 0] = torch.empty(
                    len(cruise_ids), device=self.device
                ).uniform_(max(self.cfg.stop_min_speed, ranges[0][0]), upper_speed)
                peak = torch.rand(len(cruise_ids), device=self.device) < self.cfg.stop_peak_speed_probability
                self.vel_command_b[cruise_ids[peak], 0] = upper_speed
                self.is_standing_env[cruise_ids] = False
                self._stop_pending[cruise_ids] = True
                self.time_left[cruise_ids] = torch.empty(
                    len(cruise_ids), device=self.device
                ).uniform_(*self.cfg.stop_cruise_time_range)
                if self.cfg.stop_start_time_range is not None:
                    self._sequence_speed[cruise_ids] = self.vel_command_b[cruise_ids, 0]
                    self.is_standing_env[cruise_ids] = True
                    self._stop_pending[cruise_ids] = False
                    self._start_pending[cruise_ids] = True
                    self.time_left[cruise_ids] = torch.empty(
                        len(cruise_ids), device=self.device
                    ).uniform_(*self.cfg.stop_start_time_range)
            stop_ids = env_ids[stopping]
            self.is_standing_env[stop_ids] = True
            self.time_left[stop_ids] = torch.empty(
                len(stop_ids), device=self.device
            ).uniform_(*self.cfg.stop_hold_time_range)
        # Complete scheduled starts even if the curriculum lowered its limit
        # during the short standing phase, just like an already scheduled stop.
        launch_ids = env_ids[launching]
        self.vel_command_b[launch_ids] = 0.0
        self.vel_command_b[launch_ids, 0] = self._sequence_speed[launch_ids]
        self.is_standing_env[launch_ids] = False
        self._stop_pending[launch_ids] = True
        self.time_left[launch_ids] = torch.empty(
            len(launch_ids), device=self.device
        ).uniform_(*self.cfg.stop_cruise_time_range)
        self.vel_command_b[env_ids[self.is_standing_env[env_ids]]] = 0.0
        self.segment_start_pos_w[env_ids] = _as_torch(self.robot.data.root_pos_w)[env_ids]
        self.segment_start_heading_w[env_ids] = _as_torch(self.robot.data.heading_w)[env_ids]
        self.segment_duration[env_ids] = self.time_left[env_ids]

    def _resample(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(env_ids):
            return
        # Capture before the base class replaces time_left with the next segment timer.
        previous_reference = self.tracking_command[env_ids, :2].clone()
        continuing_episode = self.command_counter[env_ids] > 0
        super()._resample(env_ids)
        self._braking_start_velocity[env_ids] = previous_reference
        self._braking_duration[env_ids] = 0.0
        if self.cfg.braking_duration_range is not None:
            target = self.command[env_ids, :2]
            previous_speed = torch.linalg.vector_norm(previous_reference, dim=1)
            target_speed = torch.linalg.vector_norm(target, dim=1)
            reversing = (previous_reference * target).sum(dim=1) < 0.0
            braking = ((target_speed < previous_speed) | reversing) & continuing_episode
            duration = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.braking_duration_range)
            # Scale the 4 m/s braking duration for smaller changes, with a short minimum.
            delta = torch.linalg.vector_norm(target - previous_reference, dim=1)
            self._braking_duration[env_ids] = torch.where(
                braking, torch.clamp(duration * delta / 4.0, min=0.1), 0.0,
            )
        if self.cfg.acceleration_duration_range is not None:
            target = self.command[env_ids, :2]
            accelerating = (
                torch.linalg.vector_norm(target, dim=1)
                > torch.linalg.vector_norm(previous_reference, dim=1)
            ) & continuing_episode & ((previous_reference * target).sum(dim=1) >= 0.0)
            duration = torch.empty(len(env_ids), device=self.device).uniform_(
                *self.cfg.acceleration_duration_range
            )
            delta = torch.linalg.vector_norm(target - previous_reference, dim=1)
            self._braking_duration[env_ids] = torch.where(
                accelerating, torch.clamp(duration * delta / 4.0, min=0.1),
                self._braking_duration[env_ids],
            )

    @property
    def tracking_command(self) -> torch.Tensor:
        """Reward-only transition reference; actor observations keep the raw command.

        Configured command changes follow a cubic smoothstep with zero endpoint slope.
        Reading this property never advances time or changes state. A new command
        starts from the current reference; episode resets discard old transitions.
        """
        reference = self.command.clone()
        elapsed = torch.clamp(self.segment_duration - self.time_left, min=0.0)
        phase = torch.clamp(elapsed / self._braking_duration.clamp_min(1.0e-6), 0.0, 1.0)
        blend = phase.square() * (3.0 - 2.0 * phase)
        velocity = torch.lerp(self._braking_start_velocity, reference[:, :2], blend[:, None])
        reference[:, :2] = torch.where((self._braking_duration > 0.0)[:, None], velocity, reference[:, :2])
        return reference

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
class SingleAxisVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type[SingleAxisVelocityCommand] = SingleAxisVelocityCommand
    stop_sequence_probability: float = 0.0
    stop_min_speed: float = 3.0
    stop_cruise_time_range: tuple[float, float] = (4.0, 6.0)
    stop_hold_time_range: tuple[float, float] = (2.0, 3.0)
    braking_duration_range: tuple[float, float] | None = None
    """Reward-only duration for a 4 m/s command drop; None preserves step tracking."""
    acceleration_duration_range: tuple[float, float] | None = None
    """Reward-only duration for a 4 m/s increase; the actor still sees an instant step."""
    stop_start_time_range: tuple[float, float] | None = None
    """Optional zero-command hold before each scheduled run-to-stop sequence."""
    stop_peak_speed_probability: float = 0.0
    """Fraction of scheduled sequences targeting the current curriculum upper speed."""
