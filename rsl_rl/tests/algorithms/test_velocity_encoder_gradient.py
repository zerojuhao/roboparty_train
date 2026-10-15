"""The estimator stop-gradient ablation must preserve inference and resume layout."""

import pytest
import torch

from rsl_rl.algorithms.fb_networks import VisualResidualActor
from rsl_rl.algorithms.gradient_audit import parameter_gradient_metrics


def actor(updates_encoder):
    return VisualResidualActor(
        dict(state=4, last_action=2, history_actor=3, velocity_target_yaw=3),
        ("state", "last_action", "history_actor"), 4, 2,
        hidden_dim=8, hidden_layers=1, embedding_layers=2,
        activation="elu", layer_norm=False, depth_shape=(2, 8, 8),
        encoder_cfg=dict(channels=[4], kernel_sizes=[3], strides=[1], paddings=[1],
                         pool_shape=(2, 2), hidden_dims=[8], activation="relu",
                         output_size=4, gru_layers=1),
        estimator_cfg=dict(hidden_dims=[8], activation="elu", first_activation="elu", layer_norm=False),
        estimate_velocity=True, velocity_estimator_updates_encoder=updates_encoder,
    )


def inputs():
    return dict(state=torch.randn(3, 4), last_action=torch.randn(3, 2),
                history_actor=torch.randn(3, 3),
                depth_image=torch.randint(256, (3, 2, 8, 8), dtype=torch.uint8)), torch.randn(3, 4)


@pytest.mark.parametrize("updates_encoder", [True, False])
def test_estimator_loss_updates_head_and_only_allowed_encoder(updates_encoder):
    torch.manual_seed(3)
    net = actor(updates_encoder)
    obs, z = inputs()
    dist = net(obs, z, .05)
    dist.velocity_estimate.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.velocity_estimator.parameters())
    encoder_grad = sum(float(p.grad.abs().sum()) for p in net.depth_encoder.parameters() if p.grad is not None)
    assert (encoder_grad > 0) == updates_encoder
    assert all(p.grad is None for p in net.policy.parameters())


def test_inference_and_optimizer_layout_are_unchanged():
    shared, stopped = actor(True), actor(False)
    stopped.load_state_dict(shared.state_dict(), strict=True)
    obs, z = inputs()
    with torch.no_grad():
        left, right = shared(obs, z, .05), stopped(obs, z, .05)
    torch.testing.assert_close(left.mean, right.mean, rtol=0, atol=0)
    torch.testing.assert_close(left.velocity_estimate, right.velocity_estimate, rtol=0, atol=0)
    first = torch.optim.Adam(shared.parameters())
    shared(obs, z, .05).velocity_estimate.square().mean().backward()
    first.step()
    second = torch.optim.Adam(stopped.parameters())
    second.load_state_dict(first.state_dict())
    assert len(first.state) == len(second.state)


@pytest.mark.parametrize("updates_encoder", [True, False])
def test_actor_loss_still_updates_encoder_without_updating_estimator(updates_encoder):
    net = actor(updates_encoder)
    obs, z = inputs()
    net(obs, z, .05).mean.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.depth_encoder.parameters())
    assert all(p.grad is None for p in net.velocity_estimator.parameters())


def test_detached_branch_parameter_audit_reports_zero():
    net = actor(False)
    obs, z = inputs()
    dist = net.forward_with_depth_gradient_diagnostics(obs, z, .05)
    terms = dict(fb=dist.mean.square().mean(), estimator=dist.velocity_estimate.square().mean())
    result = parameter_gradient_metrics(terms, net.depth_encoder.parameters())
    assert result["depth_gradient_local/estimator_actor_ratio"] == 0
    assert all(torch.isfinite(value) for value in result.values())
    dist.actor_depth_latent.retain_grad()
    dist.velocity_depth_latent.retain_grad()
    sum(terms.values()).backward()
    assert dist.actor_depth_latent.grad is not None
    assert dist.velocity_depth_latent.grad is None
