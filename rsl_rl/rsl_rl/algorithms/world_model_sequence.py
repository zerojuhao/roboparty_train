"""FB sequence returns with a critic conditioned on a shifted future window.

Tracking rows use freshly encoded contiguous expert goals. The finite context
window predicts conditional continuation; it is not an exact motion phase.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .world_model import frozen, mlp, quat_apply_inverse, yaw_quat
from .world_model_tracking import tracking_pose_loss
from .world_model_imagination import (
    accepted_count, actor_return_loss, context_masks, quality_metrics,
    return_metrics, reward_projection, rollout, score_backward,
    score_discriminator, discriminator_weight, discriminator_return_metrics,
)


class SequenceValue(nn.Module):
    def __init__(self, state_dim, xi_dim, actor_obs_dim, z_dim, context_length, hidden_dims,
                 output_dim=1, velocity_dim=2):
        super().__init__()
        self.input_dim = state_dim + xi_dim + actor_obs_dim + context_length * (z_dim + 4 + velocity_dim)
        self.net = mlp(self.input_dim, hidden_dims, output_dim)

    def forward(self, features):
        return self.net(features)


def sequence_value_input(model, agent, state, xi, observation, batch, step):
    length = model.cfg.sequence_context_length
    future = slice(step, step + length)
    z, heading, velocity, valid = (batch[key][:, future].detach().float()
                                    for key in ("z", "heading", "velocity", "valid"))
    if z.shape[1] != length:
        raise ValueError("Sequence value requires a complete shifted K-step context")
    frame = yaw_quat(state[:, 1:5])[:, None].expand(-1, length, -1)
    heading_local = quat_apply_inverse(frame, F.pad(heading, (0, 1)))[..., :2] * valid[..., :1]
    dim = velocity.shape[-1]
    velocity_local = quat_apply_inverse(frame, F.pad(velocity, (0, 3 - dim)))[..., :dim] * valid[..., 1:2]
    context = torch.cat((z, heading_local, velocity_local, valid), -1).flatten(1)
    return torch.cat((model.codec.local_state(state) / model.codec.state_scale,
                      xi, agent._actor_input(observation), context), -1)


@torch.no_grad()
def sequence_lambda_returns(rewards, values, discount, value_lambda):
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1, *rewards.shape[2:]):
        raise ValueError("Sequence lambda returns need H rewards and H+1 shifted values")
    total = values[:, -1]
    targets = []
    for step in reversed(range(rewards.shape[1])):
        total = rewards[:, step] + discount * ((1 - value_lambda) * values[:, step + 1]
                                               + value_lambda * total)
        targets.append(total)
    return torch.stack(targets[::-1], dim=1)


def fb_sequence_imagination_loss(model, agent, batch, *, total_batch_size=None,
                                 actor_weight=None, update_counters=True, discriminator_scale=1.0):
    """Return actor/value losses; external management performs one whole-batch step.

    Both losses use the original trajectory denominator, including rejected rows.
    The value loss also averages over H. No collective or counter update occurs
    when update_counters=False, including a completely rejected microbatch.
    """
    cfg = model.cfg
    ready, metrics = quality_metrics(model)
    if not ready:
        return None, None, metrics
    same_context, no_push = context_masks(model, batch, cfg.horizon + cfg.sequence_context_length, "H+K")
    latents = batch["z"][:, :cfg.horizon + 1].detach()
    projection = reward_projection(model, latents[:, :cfg.horizon])
    if projection is None:
        return None, None, metrics
    weights, _, residual = projection
    projection_ok = (torch.isfinite(residual) & (residual <= cfg.fb_max_projection_residual)).all(-1)
    trusted = same_context & no_push & projection_ok
    actor_ready = int(model.sequence_value_updates) >= cfg.value_warmup_updates
    metrics.update({"wm/context_fraction": same_context.float().mean(),
                    "wm/no_push_fraction": no_push.float().mean(),
                    "wm/fb_projection_fraction": projection_ok.float().mean(),
                    "wm/fb_projection_residual": residual.mean(),
                    "wm/sequence_value_ready": residual.new_tensor(float(actor_ready)),
                    "wm/sequence_value_updates": model.sequence_value_updates.float().clone()})
    xi = batch["xi"][:, 0].detach()
    use_discriminator = getattr(cfg, "discriminator_reward", False)
    tracking = getattr(cfg, "tracking_coef", 0) > 0
    with frozen(model.members, agent._target_backward_map, model.target_sequence_value,
                getattr(agent._actor, "velocity_estimator", None),
                agent._discriminator if use_discriminator else None), torch.set_grad_enabled(actor_ready):
        states, observations, trusted = rollout(model, agent, batch, latents, trusted, metrics)
        backward = score_backward(model, agent, states, observations)
        rewards = (backward * weights).sum(-1, keepdim=True)
        if use_discriminator:
            disc_rewards = score_discriminator(model, agent, states, observations, latents)
            rewards = torch.cat((rewards, disc_rewards.unsqueeze(-1)), dim=-1)
        discounts = agent.cfg.discount ** torch.arange(cfg.horizon, device=rewards.device)
        if tracking:
            poses = torch.stack([obs["world_model_pose"] for obs in observations[1:]], 1)
            pose_cost, pose_errors = tracking_pose_loss(poses, batch["reference_pose"][:, :cfg.horizon])
            tracking_mask = batch["tracking_mask"][:, :cfg.horizon, 0].bool()
            pose_cost = torch.where(tracking_mask, pose_cost, 0.)
            accepted_tracking = tracking_mask & trusted[:, None]
            metrics["wm/tracking_fraction"] = accepted_tracking.float().mean()
            metrics["wm/tracking_pose_loss"] = torch.where(accepted_tracking, pose_cost, 0.).detach().mean()
            for index, name in enumerate(("body_position_mse", "rotation_chordal_mse", "height_mse")):
                metrics[f"wm/tracking_{name}"] = torch.where(accepted_tracking, pose_errors[..., index], 0.).detach().mean()
        inputs = [sequence_value_input(model, agent, state, xi, observations[step], batch, step)
                  for step, state in enumerate(states)]
        detached_inputs = torch.stack([features.detach() for features in inputs], dim=1)
        with torch.no_grad():
            values = model.target_sequence_value(detached_inputs.flatten(0, 1)).reshape(
                rewards.shape[0], cfg.horizon + 1, rewards.shape[-1])
            targets = sequence_lambda_returns(rewards, values, agent.cfg.discount, cfg.value_lambda)
        reward_sum = (rewards * discounts[None, :, None]).sum(1)
        terminal = model.target_sequence_value(inputs[-1]) if actor_ready else values[:, -1]
        return_metrics(metrics, reward_sum[:, 0], terminal[:, 0], agent.cfg.discount ** cfg.horizon, weights)
        returns = reward_sum[:, 0] + agent.cfg.discount ** cfg.horizon * terminal[:, 0]
        metrics["wm/fb_return"] = returns.detach().mean()
        if use_discriminator:
            coefficient = discriminator_weight(agent, discriminator_scale, returns)
            disc_returns = discriminator_return_metrics(
                metrics, reward_sum[:, 1], terminal[:, 1], agent.cfg.discount ** cfg.horizon, coefficient)
            returns = returns + coefficient * disc_returns
        if tracking:
            pose_returns = -(pose_cost * discounts[None]).sum(-1) / discounts.sum()
            scale = torch.as_tensor(discriminator_scale, device=returns.device).detach().float().clamp_min(1.)
            coefficient = cfg.tracking_coef * scale
            returns = returns + coefficient * pose_returns
            metrics.update({"wm/tracking_return": pose_returns.detach().mean(), "wm/tracking_weight": coefficient})
    count = accepted_count(trusted, metrics, update_counters)
    if update_counters and int(count) == 0:
        return None, None, metrics
    predicted = model.sequence_value(detached_inputs[:, :cfg.horizon].flatten(0, 1)).reshape_as(targets)
    errors = torch.where(trusted[:, None, None], (predicted - targets).square(), 0.0)
    denominator = rewards.shape[0] if total_batch_size is None else total_batch_size
    value_loss = 0.5 * errors.sum() / (denominator * cfg.horizon)
    if not torch.isfinite(value_loss):
        raise FloatingPointError("Non-finite FB sequence value loss")
    metrics.update({"wm/sequence_value_loss": value_loss.detach(),
                    "wm/sequence_value_target": torch.where(trusted[:, None, None], targets, 0.0).mean()})
    actor_loss = None
    if actor_ready:
        actor_loss = actor_return_loss(model, returns, trusted, metrics, total_batch_size=total_batch_size,
                                      actor_weight=actor_weight, update_counters=update_counters)
    return actor_loss, value_loss, metrics
