"""Multi-critic AMP for locomotion, foothold, and style rewards."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP


class MultiRewardPPOAMP(PPOAMP):
    def __init__(
        self,
        *args,
        num_reward_heads: int = 3,
        advantage_weights: list[float] = (1.0, 0.5, 0.5),
        reward_head_names: list[str] = ("locomotion", "foothold", "style"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if num_reward_heads != len(advantage_weights) or num_reward_heads != len(reward_head_names):
            raise ValueError("Reward heads, weights, and names must have equal lengths")
        self.num_reward_heads = num_reward_heads
        self.advantage_weights = torch.as_tensor(advantage_weights, device=self.device)
        self.reward_head_names = reward_head_names
        self.foothold_env = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs["foothold_teacher"] = torch.zeros(self.storage.num_envs, 6, device=self.device)
        guidance = self.foothold_env.foothold_guidance if self.foothold_env is not None else None
        if guidance is not None:
            ready = guidance.trainer is not None and guidance.trainer.reward_enabled
            self.actor.set_foothold_teacher_ready(ready)
            self.actor.set_foothold_actor_mask(guidance.foothold_active_mask())
        actions = super().act(obs)
        if guidance is not None:
            self.foothold_env.prepare_foothold_prediction_step(obs, actions)
            obs["foothold_teacher"] = guidance.latest_mu_flat().detach()
        return actions

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if rewards.ndim != 2 or rewards.shape[-1] != self.num_reward_heads - 1:
            raise ValueError("SSR requires one environment reward per non-style critic head")
        self.style_rewards, self.disc_score = self._collect_style_reward(obs)
        training_rewards = torch.cat((rewards, self.style_rewards.unsqueeze(-1)), dim=-1)
        self.task_rewards = (rewards * self.advantage_weights[:-1]).sum(dim=-1)
        self.rewards_lerp = (training_rewards * self.advantage_weights).sum(dim=-1)
        PPO.process_env_step(self, obs, training_rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)
        advantages = self.storage.returns - self.storage.values
        centered = advantages - advantages.mean(dim=(0, 1), keepdim=True)
        normalized = centered / (advantages.std(dim=(0, 1), keepdim=True) + 1e-8)
        self.storage.advantages = (normalized * self.advantage_weights).sum(dim=-1, keepdim=True)

    def update(self) -> dict[str, float]:
        losses = super().update()
        if self.foothold_env is not None:
            losses.update(self.foothold_env.update_foothold_predictor())
        return losses

    def save(self) -> dict:
        saved = super().save()
        if self.foothold_env is not None:
            saved["foothold_guidance"] = self.foothold_env.foothold_guidance.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if self.foothold_env is not None and "foothold_guidance" in loaded_dict:
            self.foothold_env.foothold_guidance.load_state_dict(loaded_dict["foothold_guidance"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> MultiRewardPPOAMP:
        obs["foothold_teacher"] = torch.zeros(env.num_envs, 6, device=device)
        alg = PPOAMP.construct_algorithm(obs, env, cfg, device)
        predictor_cfg = cfg.get("foothold_imagination")
        if predictor_cfg and predictor_cfg.get("enabled"):
            env.unwrapped.configure_foothold_imagination(predictor_cfg)
            alg.foothold_env = env.unwrapped
        return alg
