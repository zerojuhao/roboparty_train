"""FB sequence returns with a critic conditioned on a shifted future window.

Stored latents are behavior-task coordinates. The finite context window predicts
the conditional continuation beyond that window; it is not an exact motion phase.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .world_model import frozen, mlp, quat_apply_inverse, yaw_quat
from .world_model_imagination import (
    accepted_count, actor_return_loss, context_masks, quality_metrics,
    return_metrics, reward_projection, rollout, score_backward,
)


class SequenceValue(nn.Module):
    def __init__(self, state_dim, xi_dim, actor_obs_dim, z_dim, context_length, hidden_dims):
        super().__init__()
        self.input_dim = state_dim + xi_dim + actor_obs_dim + context_length * (z_dim + 6)
        self.net = mlp(self.input_dim, hidden_dims, 1)

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
    velocity_local = quat_apply_inverse(frame, F.pad(velocity, (0, 1)))[..., :2] * valid[..., 1:2]
    context = torch.cat((z, heading_local, velocity_local, valid), -1).flatten(1)
    return torch.cat((model.codec.local_state(state) / model.codec.state_scale,
                      xi, agent._actor_input(observation), context), -1)


@torch.no_grad()
def sequence_lambda_returns(rewards, values, discount, value_lambda):
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("Sequence lambda returns need H rewards and H+1 shifted values")
    total = values[:, -1]
    targets = []
    for step in reversed(range(rewards.shape[1])):
        total = rewards[:, step] + discount * ((1 - value_lambda) * values[:, step + 1]
                                               + value_lambda * total)
        targets.append(total)
    return torch.stack(targets[::-1], dim=1)


def fb_sequence_imagination_loss(model, agent, batch, *, total_batch_size=None,
                                 actor_weight=None, update_counters=True):
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
    with frozen(model.members, agent._target_backward_map, model.target_sequence_value,
                getattr(agent._actor, "velocity_estimator", None)), torch.set_grad_enabled(actor_ready):
        states, observations, trusted = rollout(model, agent, batch, latents, trusted, metrics)
        backward = score_backward(model, agent, states, observations)
        rewards = (backward * weights).sum(-1)
        inputs = [sequence_value_input(model, agent, state, xi, observations[step], batch, step)
                  for step, state in enumerate(states)]
        detached_inputs = torch.stack([features.detach() for features in inputs], dim=1)
        with torch.no_grad():
            values = model.target_sequence_value(detached_inputs.flatten(0, 1)).reshape(rewards.shape[0], -1)
            targets = sequence_lambda_returns(rewards, values, agent.cfg.discount, cfg.value_lambda)
        discounts = agent.cfg.discount ** torch.arange(cfg.horizon, device=rewards.device)
        reward_sum = (rewards * discounts).sum(-1)
        terminal = (model.target_sequence_value(inputs[-1]).reshape(-1) if actor_ready else values[:, -1])
        return_metrics(metrics, reward_sum, terminal, agent.cfg.discount ** cfg.horizon, weights)
        returns = reward_sum + agent.cfg.discount ** cfg.horizon * terminal
    count = accepted_count(trusted, metrics, update_counters)
    if update_counters and int(count) == 0:
        return None, None, metrics
    predicted = model.sequence_value(detached_inputs[:, :cfg.horizon].flatten(0, 1)).reshape_as(targets)
    errors = torch.where(trusted[:, None], (predicted - targets).square(), 0.0)
    denominator = rewards.shape[0] if total_batch_size is None else total_batch_size
    value_loss = 0.5 * errors.sum() / (denominator * cfg.horizon)
    if not torch.isfinite(value_loss):
        raise FloatingPointError("Non-finite FB sequence value loss")
    metrics.update({"wm/sequence_value_loss": value_loss.detach(),
                    "wm/sequence_value_target": torch.where(trusted[:, None], targets, 0.0).mean()})
    actor_loss = None
    if actor_ready:
        actor_loss = actor_return_loss(model, returns, trusted, metrics, total_batch_size=total_batch_size,
                                      actor_weight=actor_weight, update_counters=update_counters)
    return actor_loss, value_loss, metrics
