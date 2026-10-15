"""Tests for AMP rollout fast paths."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_amp import PPOAMP


class _FixedDiscriminator:
    def style_reward(self, obs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        score = obs.sum(dim=-1)
        return dt * (score + 2.0), score


def _single_route_amp(weight: float = 1.0) -> PPOAMP:
    amp = object.__new__(PPOAMP)
    route = {
        "policy_obs_set": "discriminator",
        "demonstration_obs_set": "discriminator_demonstration",
        "style_reward_weight": weight,
    }
    amp.device = "cpu"
    amp.routes = {"general": route}
    amp.amp_cfg = {
        "route_obs_group": None,
        "step_dt": 0.02,
        "turn_style_reward_multiplier": 1.0,
    }
    amp.amp_obs_groups = {
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    amp.discriminators = {"general": _FixedDiscriminator()}
    amp.disc_buffers = {"general": (deque(), deque())}
    amp.storage = SimpleNamespace(num_envs=3)
    amp._rollout_distributions = {}
    return amp


def test_single_global_route_fast_path_preserves_rewards_and_buffers() -> None:
    amp = _single_route_amp(weight=2.5)
    disc = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-2.0, 1.0]])
    demo = disc + 10.0
    obs = TensorDict({"disc": disc, "disc_demo": demo}, batch_size=[3])

    style, score = amp._collect_style_reward(obs)
    raw_score = disc.sum(dim=-1)
    raw_reward = 0.02 * (raw_score + 2.0)
    weight = torch.full_like(raw_reward, 2.5)

    torch.testing.assert_close(style, 2.5 * raw_reward / weight)
    torch.testing.assert_close(score, 2.5 * raw_score / weight)
    torch.testing.assert_close(amp.disc_buffers["general"][0][0], disc)
    torch.testing.assert_close(amp.disc_buffers["general"][1][0], demo)


def test_single_global_route_rejects_zero_weight() -> None:
    amp = _single_route_amp(weight=0.0)
    obs = TensorDict(
        {"disc": torch.ones(3, 2), "disc_demo": torch.ones(3, 2)}, batch_size=[3]
    )
    with pytest.raises(RuntimeError, match="cover every environment"):
        amp._collect_style_reward(obs)


def test_rollout_distribution_allocates_once_per_name_and_preserves_statistics() -> None:
    amp = _single_route_amp()
    first = torch.tensor([1.0, 3.0, -2.0], requires_grad=True)
    second = torch.tensor([2.0, 4.0, -1.0], requires_grad=True)
    with patch.object(amp, "_new_distribution_stats", wraps=amp._new_distribution_stats) as allocate:
        amp._record_rollout_distribution("style", first)
        amp._record_rollout_distribution("style", second)
        amp._record_rollout_distribution("score", first)
        assert allocate.call_count == 2
        stats = amp._rollout_distributions["style"]
        values = torch.cat((first, second)).detach()
        for key, expected in (("sum", values.sum()), ("sum_sq", values.square().sum()),
                              ("count", torch.tensor(float(values.numel()))),
                              ("min", values.min()), ("max", values.max())):
            torch.testing.assert_close(stats[key], expected)
            assert not stats[key].requires_grad
        amp._rollout_distributions.clear()
        amp._record_rollout_distribution("style", first)
        assert allocate.call_count == 3
