"""Reuse preserves actor gradients, optimizer updates and normalizer statistics."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent
from rsl_rl.algorithms.fb_networks import ObsNormalizer, ResidualForwardMap, SharedCriticMap, VisualResidualActor
from rsl_rl.algorithms.pufo import PufoAgent


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DIMS = dict(state=4, last_action=2, history_actor=3, heading=2, velocity_command=3, velocity_target_yaw=3)
KEYS = tuple(k for k in DIMS if k != "velocity_target_yaw")


def make_agent(device, updates_encoder=True):
    agent = object.__new__(PufoAgent)
    agent.cfg = SimpleNamespace(actor_std=.05, stddev_clip=.3, actor_pessimism_penalty=.5,
                                scale_reg=True, reg_coeff=.01, reg_coeff_aux=.02,
                                reg_coeff_heading=.02, reg_coeff_velocity=.03,
                                velocity_estimator_loss_coef=1.0)
    agent.device = device
    agent.visual_enabled = True
    agent.use_bf16_amp = False
    agent.shared_value_net = True
    agent.actor_keys = agent.full_keys = KEYS
    agent.obs_dims = DIMS
    model = dict(hidden_dim=8, hidden_layers=1, embedding_layers=2, activation="elu", layer_norm=False)
    agent._actor = VisualResidualActor(
        DIMS, KEYS, 4, 2, **model, depth_shape=(2, 8, 8),
        encoder_cfg=dict(channels=[4], kernel_sizes=[3], strides=[1], paddings=[1],
                         pool_shape=(2, 2), hidden_dims=[8], activation="relu", output_size=4, gru_layers=1),
        estimator_cfg=dict(hidden_dims=[8], activation="elu", first_activation="elu", layer_norm=False),
        estimate_velocity=True, velocity_estimator_updates_encoder=updates_encoder,
    ).to(device)
    full_dim = sum(DIMS[k] for k in KEYS)
    agent._critic_shared = SharedCriticMap(full_dim, 4, 2, num_parallel=2, **model).to(device)
    agent._context_critic_shared = SharedCriticMap(
        full_dim, 4, 2, num_parallel=2, head_names=("heading", "velocity"), **model
    ).to(device)
    agent._forward_map = ResidualForwardMap(full_dim, 4, 2, num_parallel=2, **model).to(device)
    agent.actor_optimizer = torch.optim.Adam(agent._actor.parameters(), lr=3e-4)
    return agent


def inputs(device, batch_size=8):
    obs = {k: torch.randn(batch_size, dim, device=device) for k, dim in DIMS.items()}
    obs["depth_image"] = torch.randint(256, (batch_size, 2, 8, 8), device=device, dtype=torch.uint8)
    context = {**obs, "heading": obs["heading"] + .3, "velocity_command": obs["velocity_command"] - .5}
    return dict(actor_obs=obs, full_obs=torch.cat([obs[k] for k in KEYS], -1),
                z=torch.randn(batch_size, 4, device=device), context_actor_obs=context,
                context_full_obs=torch.cat([context[k] for k in KEYS], -1),
                context_z=torch.randn(batch_size, 4, device=device),
                heading_valid=torch.arange(batch_size, device=device) % 2 == 0,
                velocity_valid=torch.arange(batch_size, device=device) % 3 == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("updates_encoder", [True, False])
def test_reused_encoding_preserves_actions_gradients_and_adam_steps(device, updates_encoder, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(11)
    shared = make_agent(device, updates_encoder)
    reference = copy.deepcopy(shared)
    batch = inputs(device)
    separate = {**batch, "context_actor_obs": {
        **batch["context_actor_obs"], "depth_image": batch["actor_obs"]["depth_image"].clone(),
    }}
    for step in range(3):
        means = [[], []]
        handles = [agent._actor.register_forward_hook(
            lambda module, args, out, dest=dest: dest.append(out.mean.detach().clone())
        ) for agent, dest in zip((shared, reference), means)]
        try:
            torch.manual_seed(31 + step)
            with patch.object(shared._actor.depth_encoder, "forward", wraps=shared._actor.depth_encoder.forward) as encode:
                actual = shared.update_actor(**batch)
                assert encode.call_count == 1
            torch.manual_seed(31 + step)
            with patch.object(reference._actor.depth_encoder, "forward", wraps=reference._actor.depth_encoder.forward) as encode:
                expected = reference.update_actor(**separate)
                assert encode.call_count == 2
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(means[0], means[1], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for p, q in zip(shared._actor.parameters(), reference._actor.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(shared.actor_optimizer.state_dict(), reference.actor_optimizer.state_dict(),
                                   atol=2e-6, rtol=2e-5)
        assert all(p.requires_grad for p in shared._critic_shared.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA precision/compile coverage")
@pytest.mark.parametrize("precision,compiled", [("tf32", False), ("bf16", False), ("bf16", True)])
def test_cuda_reuse_roundoff_and_compiled_backward(precision, compiled, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    torch.manual_seed(11)
    shared = make_agent("cuda")
    shared.use_bf16_amp = precision == "bf16"
    reference = copy.deepcopy(shared)
    for agent in (shared, reference):
        agent._actor.depth_encoder.gru.flatten_parameters()
        if compiled:
            agent._actor.forward = torch.compile(agent._actor.forward, mode="default")
    batch = inputs("cuda")
    separate = {**batch, "context_actor_obs": {
        **batch["context_actor_obs"], "depth_image": batch["actor_obs"]["depth_image"].clone(),
    }}
    for step in range(3):
        torch.manual_seed(31 + step)
        actual = shared.update_actor(**batch)
        torch.manual_seed(31 + step)
        expected = reference.update_actor(**separate)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
        gradients = [torch.cat([p.grad.flatten() for p in agent._actor.parameters() if p.grad is not None])
                     for agent in (shared, reference)]
        # Shared backward changes the summation order in low-precision kernels.
        assert (gradients[0] - gradients[1]).norm() / gradients[1].norm() < 1e-3
        for p, q in zip(shared._actor.parameters(), reference._actor.parameters()):
            torch.testing.assert_close(p, q, atol=3e-5, rtol=1e-4)


def test_diagnostic_update_keeps_separate_encoder_branches():
    agent = make_agent("cpu")
    with patch.object(agent._actor.depth_encoder, "forward", wraps=agent._actor.depth_encoder.forward) as encode:
        result = agent.update_actor(**inputs("cpu"), depth_gradient_diagnostics=True)
    assert encode.call_count == 4  # Actor, two diagnostic ablations, context.
    assert all(torch.isfinite(value) for value in result.values())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("visual", [False, True])
def test_relabel_normalization_preserves_values_counters_and_original_context(device, visual):
    agent = object.__new__(PufoAgent if visual else FbCprAuxAgent)
    agent.obs_dims, agent.visual_enabled = DIMS, visual
    agent._obs_normalizer = ObsNormalizer(DIMS, identity_keys=("heading", "velocity_command")).to(device)
    obs = inputs(device)["actor_obs"]
    for _ in range(3):
        agent._obs_normalizer.update_dict(agent._parts(obs))
    before = copy.deepcopy(agent._obs_normalizer.state_dict())
    normalized = agent._norm_parts(obs)
    original = {k: v.clone() for k, v in normalized.items()}
    relabelled = {**obs, "heading": obs["heading"] + 1, "velocity_command": obs["velocity_command"] - 2} if visual else obs
    expected = agent._norm_parts(relabelled)
    with patch.object(agent._obs_normalizer, "normalize_dict", wraps=agent._obs_normalizer.normalize_dict) as normalize:
        actual = agent._norm_relabelled_parts(relabelled, normalized)
    assert normalize.call_count == int(visual)
    if visual:
        assert set(normalize.call_args.args[0]) == {"heading", "velocity_command"}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(normalized, original, rtol=0, atol=0)
    torch.testing.assert_close(agent._obs_normalizer.state_dict(), before, rtol=0, atol=0)
