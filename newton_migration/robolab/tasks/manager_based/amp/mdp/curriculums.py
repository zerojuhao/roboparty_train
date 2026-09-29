"""Command curricula driven by completed-episode tracking performance."""

import torch

from isaaclab.managers import ManagerTermBase


class VelocityTrackingCurriculum(ManagerTermBase):
    """Adjust forward speed after a batch of episodes at the current level."""

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
        self.score_sum = torch.zeros((), device=env.device)
        self.episode_count = 0
        self.last_change_step = 0
        self.last_update_step = 0
        self.last_score = 0.0

    def __call__(
        self, env, env_ids, command_name, reward_term_name,
        min_velocity, max_velocity, step_size, down_threshold, up_threshold,
    ):
        lengths = env.episode_length_buf[env_ids]
        # Ignore initialization and episodes that began before the last level change.
        valid = (lengths > 0) & (env.common_step_counter - lengths >= self.last_change_step)
        weight = env.reward_manager.get_term_cfg(reward_term_name).weight
        if weight <= 0:
            raise ValueError("Velocity curriculum requires a positive tracking reward weight.")
        scores = env.reward_manager._episode_sums[reward_term_name][env_ids]
        scores = scores / (weight * env.max_episode_length_s)
        self.score_sum += scores[valid].sum()
        self.episode_count += int(valid.sum().item())
        upper = self.command.cfg.ranges.lin_vel_x[1]
        # One decision per episode horizon, based on at least num_envs completed episodes.
        if (
            self.episode_count >= env.num_envs
            and env.common_step_counter - self.last_update_step >= env.max_episode_length
        ):
            self.last_score = (self.score_sum / self.episode_count).item()
            new_upper = upper
            if self.last_score > up_threshold:
                new_upper = min(max_velocity, upper + step_size)
            elif self.last_score < down_threshold:
                new_upper = max(min_velocity, upper - step_size)
            self.command.cfg.ranges.lin_vel_x = (self.lower, new_upper)
            if new_upper != upper:
                self.last_change_step = env.common_step_counter
            upper = new_upper
            self.score_sum.zero_()
            self.episode_count = 0
            self.last_update_step = env.common_step_counter
        return {"max_velocity": upper, "tracking_score": self.last_score}
