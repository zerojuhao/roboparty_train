"""Tests for AMP rollout fast paths."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.amp_model import AMPModel


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


def _two_route_amp(route_group=None, device="cpu"):
    amp = _single_route_amp()
    amp.device = device
    amp.routes["general"]["all_envs"] = True
    amp.routes["upper"] = {
        "all_envs": True,
        "policy_obs_set": "upper_discriminator",
        "demonstration_obs_set": "upper_demo",
        "style_reward_weight": 3.0,
        "route_value": 1,
    }
    amp.amp_cfg["route_obs_group"] = route_group
    amp.amp_obs_groups.update({"upper_discriminator": ["upper"], "upper_demo": ["upper_demo"]})
    amp.discriminators["upper"] = _FixedDiscriminator()
    amp.disc_buffers["upper"] = (deque(), deque())
    policy = torch.tensor([[1., 2.], [3., 4.], [-2., 1.]])
    obs = TensorDict({"disc": policy, "disc_demo": policy + 10,
                      "upper": policy + 2, "upper_demo": policy + 20,
                      "route": torch.tensor([[0], [1], [0]])}, batch_size=[3])
    return amp, obs.to(device)


def _legacy_style_reward(amp, obs):
    route_group = amp.amp_cfg.get("route_obs_group")
    route_values = obs[route_group].reshape(-1).long() if route_group else None
    style = torch.zeros(amp.storage.num_envs, device=amp.device)
    scores, weights = torch.zeros_like(style), torch.zeros_like(style)
    for name, route in amp.routes.items():
        mask = (torch.ones_like(style, dtype=torch.bool)
                if route.get("all_envs", False) or route_values is None
                else route_values == route["route_value"])
        if not mask.any():
            continue
        policy = amp._group_obs(obs, amp.amp_obs_groups[route["policy_obs_set"]])[mask]
        demo = amp._group_obs(obs, amp.amp_obs_groups[route["demonstration_obs_set"]])[mask]
        reward, score = amp.discriminators[name].style_reward(policy, amp.amp_cfg["step_dt"])
        amp._record_rollout_distribution(name, reward)
        weight = float(route.get("style_reward_weight", 1.0))
        style[mask] += weight * reward
        scores[mask] += weight * score
        weights[mask] += weight
        amp.disc_buffers[name][0].append(policy.detach())
        amp.disc_buffers[name][1].append(demo.detach())
    if (weights == 0).any():
        raise RuntimeError("AMP routes do not cover every environment")
    style /= weights
    if amp.amp_cfg.get("turn_style_reward_multiplier", 1.0) != 1.0:
        style = amp._scale_turn_style_reward(style, obs)
    amp._record_rollout_distribution("combined", style)
    return style, scores / weights


@pytest.mark.parametrize("route_group", [None, "route"])
@pytest.mark.parametrize("upper_weight", [0.0, 0.7, 3.0])
@pytest.mark.parametrize("turn_multiplier", [1.0, 0.5])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_global_routes_match_legacy_rewards_buffers_and_statistics(route_group, upper_weight, turn_multiplier, device):
    actual, obs = _two_route_amp(route_group, device)
    reference, _ = _two_route_amp(route_group, device)
    for amp in (actual, reference):
        amp.routes["upper"]["style_reward_weight"] = upper_weight
        amp.amp_cfg["turn_style_reward_multiplier"] = turn_multiplier
        amp._scale_turn_style_reward = lambda style, _: style * turn_multiplier
    expected = _legacy_style_reward(reference, obs)
    result = actual._collect_style_reward(obs)
    for left, right in zip(result, expected):
        torch.testing.assert_close(left, right)
    for name in actual.routes:
        for left, right in zip(actual.disc_buffers[name], reference.disc_buffers[name]):
            torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
    for name, stats in actual._rollout_distributions.items():
        for key, value in stats.items():
            torch.testing.assert_close(value, reference._rollout_distributions[name][key])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_global_routes_have_no_host_scalar_reads_or_boolean_indexing(device):
    amp, obs = _two_route_amp(device=device)
    amp._collect_style_reward(obs)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        amp._collect_style_reward(obs)
    operations = {event.key for event in prof.key_averages()}
    assert not {"aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::index"} & operations


def test_global_routes_reject_zero_total_weight():
    amp, obs = _two_route_amp()
    amp.routes["upper"]["style_reward_weight"] = -1.0
    with pytest.raises(RuntimeError, match="cover every environment"):
        amp._collect_style_reward(obs)


def test_partial_routes_still_use_masks_and_match_legacy():
    actual, obs = _two_route_amp("route")
    reference, _ = _two_route_amp("route")
    for amp in (actual, reference):
        amp.routes["upper"]["all_envs"] = False
    for left, right in zip(actual._collect_style_reward(obs), _legacy_style_reward(reference, obs)):
        torch.testing.assert_close(left, right)
    assert actual.disc_buffers["upper"][0][0].shape[0] == 1


@pytest.mark.parametrize("actor_class", [
    "rsl_rl.models.amp_model:AMPModel", "rsl_rl.models.amp_model.AMPModel", AMPModel,
])
def test_construct_resolves_velocity_targets_for_qualified_amp_models(actor_class) -> None:
    manager = SimpleNamespace(active_terms={"critic": ["joint_pos", "base_lin_vel"]},
                              group_obs_term_dim={"critic": [(24,), (9,)]})
    raw = SimpleNamespace(observation_manager=manager, step_dt=0.02)
    env = SimpleNamespace(unwrapped=raw, env=SimpleNamespace(unwrapped=raw))
    cfg = {
        "actor": {"class_name": actor_class, "enable_velocity_estimation": True},
        "algorithm": {"amp_cfg": {"discriminators": {"general": {
            "policy_obs_set": "discriminator", "demonstration_obs_set": "demo",
        }}}},
        "obs_groups": {"discriminator": ["disc"], "demo": ["disc_demo"]},
    }
    obs = TensorDict({"disc": torch.zeros(2, 3, 1), "disc_demo": torch.zeros(2, 3, 1)},
                     batch_size=[2])
    with patch.object(PPO, "construct_algorithm") as construct:
        PPOAMP.construct_algorithm(obs, env, cfg, "cpu")
    assert cfg["actor"]["velocity_target_indices"] == (30, 31)
    construct.assert_called_once_with(obs, env, cfg, "cpu")
