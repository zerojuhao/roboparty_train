"""Command curricula driven by completed-episode tracking performance."""

import torch

from isaaclab.managers import ManagerTermBase


class VelocityTrackingCurriculum(ManagerTermBase):
    """Adjust forward speed from globally synchronized episode tracking scores."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        self.lower, initial_upper = self.command.cfg.ranges.lin_vel_x
        minimum = cfg.params["min_velocity"]
        maximum = cfg.params["max_velocity"]
        if not self.lower <= minimum <= initial_upper <= maximum:
            raise ValueError("Expected command lower <= min_velocity <= initial upper <= max_velocity.")
        if cfg.params["step_size"] <= 0:
            raise ValueError("step_size must be positive.")
        if not 0 <= cfg.params["down_threshold"] < cfg.params["up_threshold"] <= 1:
            raise ValueError("Expected 0 <= down_threshold < up_threshold <= 1.")
        up_required_windows = cfg.params.get(
            "up_required_windows", cfg.params.get("required_windows", 1)
        )
        down_required_windows = cfg.params.get(
            "down_required_windows", cfg.params.get("required_windows", 1)
        )
        if up_required_windows < 1 or down_required_windows < 1:
            raise ValueError("up/down required windows must be at least one.")
        self.local_score_sum = torch.zeros((), device=env.device)
        self.local_episode_count = torch.zeros((), device=env.device)
        self.window_score_sum = torch.zeros((), device=env.device)
        self.window_episode_count = torch.zeros((), device=env.device)
        self.last_change_step = 0
        self.last_update_step = 0
        self.last_score = 0.0
        self.consecutive_up = 0
        self.consecutive_down = 0

    def __call__(
        self,
        env,
        env_ids,
        command_name,
        reward_term_name,
        min_velocity,
        max_velocity,
        step_size,
        down_threshold,
        up_threshold,
        up_required_windows=None,
        down_required_windows=None,
        required_windows=None,
    ):
        del (
            command_name,
            min_velocity,
            max_velocity,
            step_size,
            down_threshold,
            up_threshold,
            up_required_windows,
            down_required_windows,
            required_windows,
        )
        lengths = env.episode_length_buf[env_ids]
        # Include every command mode, but exclude initialization and episodes
        # containing samples from before the most recent speed-level change.
        valid = (lengths > 0) & (env.common_step_counter - lengths >= self.last_change_step)
        weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        if weight <= 0:
            raise ValueError("Velocity curriculum requires a positive tracking reward weight.")
        scores = env.reward_manager._episode_sums[reward_term_name][env_ids]
        scores = scores / (weight * env.max_episode_length_s)
        self.local_score_sum += scores[valid].sum()
        self.local_episode_count += valid.sum()
        return self.state()

    def synchronize(self) -> dict[str, float]:
        """Merge rank-local samples and update the curriculum on a common boundary."""
        contribution = torch.stack((self.local_score_sum, self.local_episode_count))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(contribution, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        self.window_score_sum += contribution[0]
        self.window_episode_count += contribution[1]
        self.local_score_sum.zero_()
        self.local_episode_count.zero_()

        env = self._env
        enough_time = env.common_step_counter - self.last_update_step >= env.max_episode_length
        enough_episodes = self.window_episode_count >= env.num_envs * world_size
        if enough_time and enough_episodes:
            self.last_score = (self.window_score_sum / self.window_episode_count).item()
            self._update_level()
            self.window_score_sum.zero_()
            self.window_episode_count.zero_()
            self.last_update_step = env.common_step_counter
        return self.state()

    def _update_level(self) -> None:
        params = self.cfg.params
        if self.last_score > params["up_threshold"]:
            self.consecutive_up += 1
            self.consecutive_down = 0
        elif self.last_score < params["down_threshold"]:
            self.consecutive_down += 1
            self.consecutive_up = 0
        else:
            self.consecutive_up = 0
            self.consecutive_down = 0

        upper = self.command.cfg.ranges.lin_vel_x[1]
        new_upper = upper
        up_required_windows = params.get(
            "up_required_windows", params.get("required_windows", 1)
        )
        down_required_windows = params.get(
            "down_required_windows", params.get("required_windows", 1)
        )
        if self.consecutive_up >= up_required_windows:
            new_upper = min(params["max_velocity"], upper + params["step_size"])
        elif self.consecutive_down >= down_required_windows:
            new_upper = max(params["min_velocity"], upper - params["step_size"])
        if new_upper != upper:
            self.command.cfg.ranges.lin_vel_x = (self.lower, new_upper)
            self.last_change_step = self._env.common_step_counter
            self.consecutive_up = 0
            self.consecutive_down = 0

    def state(self) -> dict[str, float]:
        return {
            "max_velocity": self.command.cfg.ranges.lin_vel_x[1],
            "tracking_score": self.last_score,
            "consecutive_up_windows": self.consecutive_up,
            "consecutive_down_windows": self.consecutive_down,
        }

    def checkpoint_state(self) -> dict[str, float]:
        """Return the dynamic command range needed to resume this curriculum."""
        return {"max_velocity": float(self.command.cfg.ranges.lin_vel_x[1])}

    def restore_checkpoint_state(self, state: dict[str, float]) -> None:
        """Restore the command range while starting a fresh score window."""
        max_velocity = float(state["max_velocity"])
        minimum = self.cfg.params["min_velocity"]
        maximum = self.cfg.params["max_velocity"]
        # Clamp when resuming into a tighter curriculum ceiling (e.g. survival-first vmax=2.5
        # from a speed4 ckpt that stored max_velocity=4.0). Prefer clamp over hard fail.
        if max_velocity > maximum or max_velocity < minimum:
            clamped = min(maximum, max(minimum, max_velocity))
            print(
                f"[WARN] Curriculum checkpoint max_velocity {max_velocity} outside "
                f"[{minimum}, {maximum}]; clamping to {clamped}."
            )
            max_velocity = clamped
        self.command.cfg.ranges.lin_vel_x = (self.lower, max_velocity)
        self.local_score_sum.zero_()
        self.local_episode_count.zero_()
        self.window_score_sum.zero_()
        self.window_episode_count.zero_()
        self.last_change_step = self._env.common_step_counter
        self.last_update_step = self._env.common_step_counter
        self.consecutive_up = 0
        self.consecutive_down = 0
