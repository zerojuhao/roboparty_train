# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour AMP environment wiring :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg` to MultiReward."""

from __future__ import annotations

import torch

from isaaclab.envs import VecEnvStepReturn
from isaaclab.utils.buffers import CircularBuffer

from robolab.tasks.manager_based.amp.amp_env import AmpEnv
from robolab.tasks.manager_based.parkour.managers import (
    DummyRewardCfg,
    MultiRewardCfg,
    MultiRewardManager,
)


def _reward_dict_to_vector(rew: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack multi-group rewards in declaration order for RSL-RL.

    Single group → ``(num_envs,)``; multiple groups → ``(num_envs, num_groups)``.
    """
    if not rew:
        raise ValueError("MultiRewardManager produced an empty reward dict.")
    out = torch.stack(tuple(rew.values()), dim=-1)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


class ParkourEnv(AmpEnv):
    """Same as :class:`~robolab.tasks.manager_based.amp.amp_env.AmpEnv` but swaps in
    :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardManager` when ``cfg.rewards``
    is a :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg`.

    Enables curriculum helpers that adjust per-environment reward weights via
    ``get_per_env_term_weights`` / ``set_term_weight_for_envs``.

    RSL-RL expects tensor rewards from ``step``; ``MultiRewardManager.compute`` returns a dict, so we
    convert it here without touching ``rsl_rl``.
    """

    def load_managers(self):
        reward_group_cfg = None
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()

        super().load_managers()

        # Action history for smoothness rewards (same layout as direct BaseEnv).
        self.action_buffer = CircularBuffer(
            max_len=3, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(
            torch.zeros(
                self.num_envs,
                self.action_manager.total_action_dim,
                dtype=torch.float,
                device=self.device,
            )
        )

        if reward_group_cfg is not None:
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi Reward Manager: ", self.reward_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_buffer.append(action.to(self.device))
        obs, rew, terminated, truncated, extras = super().step(action)
        reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.action_buffer.reset(reset_env_ids)
        if isinstance(rew, dict):
            rew = _reward_dict_to_vector(rew)
            self.reward_buf = rew
        return obs, rew, terminated, truncated, extras
