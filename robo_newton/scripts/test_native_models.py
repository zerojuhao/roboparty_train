"""Small CPU checks for the structured models and native rollout storage."""

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "rsl_rl"))

from rsl_rl.models import ParkourModel
from rsl_rl.algorithms.ppo_amp import PPOAMP
from rsl_rl.storage import RolloutStorage


def test_parkour_model() -> None:
    envs = 4
    policy = TensorDict(
        {
            "base_ang_vel": torch.randn(envs, 24),
            "joint_pos": torch.randn(envs, 192),
            "depth_image": torch.randn(envs, 1, 32, 32),
        },
        batch_size=[envs],
    )
    critic = TensorDict(
        {"base_lin_vel": torch.randn(envs, 24), "joint_pos": torch.randn(envs, 192)},
        batch_size=[envs],
    )
    obs = TensorDict({"policy": policy, "critic": critic}, batch_size=[envs])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    encoder_cfg = {
        "channels": [4], "kernel_sizes": [3], "strides": [1],
        "hidden_sizes": [32], "output_size": 16, "paddings": [1],
        "nonlinearity": "ReLU", "use_maxpool": True, "last_activation": "ReLU",
    }
    actor = ParkourModel(
        obs, groups, "actor", 23, hidden_dims=[32, 16], encoder_cfg=encoder_cfg,
        num_moe_experts=3, enable_velocity_estimation=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    value = ParkourModel(obs, groups, "critic", 1, hidden_dims=[32, 16])
    actions = actor(obs, stochastic_output=True)
    values = value(obs)
    assert actions.shape == (envs, 23)
    assert values.shape == (envs, 1)
    assert actor.get_aux_loss() is not None
    (actor.get_aux_loss() + values.square().mean()).backward()

    storage = RolloutStorage("rl", envs, 2, obs, [23])
    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.actions = actions.detach()
    transition.rewards = torch.zeros(envs)
    transition.dones = torch.zeros(envs)
    transition.values = values.detach()
    transition.actions_log_prob = actor.get_output_log_prob(actions).detach()
    transition.distribution_params = tuple(param.detach() for param in actor.output_distribution_params)
    storage.add_transition(transition)
    assert storage.observations[0]["policy"]["depth_image"].shape == (envs, 1, 32, 32)


def test_ssr_model() -> None:
    envs = 4
    policy = TensorDict(
        {
            "base_ang_vel": torch.randn(envs, 3 * 8),
            "joint_pos": torch.randn(envs, 24 * 8),
            "depth_image": torch.randn(envs, 1, 32, 32),
        }, batch_size=[envs],
    )
    critic = TensorDict(
        {
            "base_lin_vel": torch.randn(envs, 3 * 8),
            "joint_pos": torch.randn(envs, 24 * 8),
            "left_foot_height_map": torch.randn(envs, 120),
            "right_foot_height_map": torch.randn(envs, 120),
        }, batch_size=[envs],
    )
    obs = TensorDict({"policy": policy, "critic": critic, "foothold_teacher": torch.randn(envs, 6)}, batch_size=[envs])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    encoder_cfg = {
        "channels": [4], "kernel_sizes": [3], "strides": [1],
        "hidden_sizes": [32], "output_size": 16, "paddings": [1],
        "nonlinearity": "ReLU", "use_maxpool": True, "last_activation": "ReLU",
    }
    actor = ParkourModel(
        obs, groups, "actor", 24, hidden_dims=[32, 16], encoder_cfg=encoder_cfg,
        num_moe_experts=3, enable_ssr_estimation=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic_model = ParkourModel(obs, groups, "critic", 3, hidden_dims=[32, 16], num_moe_experts=3)
    actions = actor(obs, stochastic_output=True)
    values = critic_model(obs)
    assert actions.shape == (envs, 24)
    assert values.shape == (envs, 3)
    loss = actor.get_aux_loss()
    assert loss is not None and torch.isfinite(loss)
    (loss + values.square().mean()).backward()


def test_discriminator_routes() -> None:
    class FixedDiscriminator:
        def __init__(self, score):
            self.score = score

        def style_reward(self, observations, step_dt):
            value = torch.full((observations.shape[0],), self.score * step_dt)
            return value, value

    routes = {
        "general": {"route_value": 0, "policy_obs_set": "discriminator", "demonstration_obs_set": "general_demo"},
        "special": {"route_value": 1, "policy_obs_set": "discriminator", "demonstration_obs_set": "special_demo"},
    }
    algorithm = SimpleNamespace(
        amp_cfg={"route_obs_group": "route", "step_dt": 1.0},
        amp_obs_groups={"discriminator": ["policy"], "general_demo": ["demo_a"], "special_demo": ["demo_b"]},
        storage=SimpleNamespace(num_envs=4),
        device="cpu",
        routes=routes,
        discriminators={"general": FixedDiscriminator(1), "special": FixedDiscriminator(2)},
        disc_buffers={name: (deque(), deque()) for name in routes},
        _group_obs=PPOAMP._group_obs,
    )
    observations = TensorDict(
        {
            "route": torch.tensor([[0], [1], [1], [0]]),
            "policy": torch.arange(4.0).reshape(4, 1, 1),
            "demo_a": torch.ones(4, 1, 1),
            "demo_b": torch.full((4, 1, 1), 2.0),
        }, batch_size=[4],
    )
    style, _ = PPOAMP._collect_style_reward(algorithm, observations)
    assert style.tolist() == [1.0, 2.0, 2.0, 1.0]
    assert algorithm.disc_buffers["general"][0][0].flatten().tolist() == [0.0, 3.0]
    assert algorithm.disc_buffers["special"][0][0].flatten().tolist() == [1.0, 2.0]
    assert algorithm.disc_buffers["general"][1][0].unique().tolist() == [1.0]
    assert algorithm.disc_buffers["special"][1][0].unique().tolist() == [2.0]


if __name__ == "__main__":
    test_parkour_model()
    test_ssr_model()
    test_discriminator_routes()
    print("native models, rollout storage, and discriminator routes: ok")
