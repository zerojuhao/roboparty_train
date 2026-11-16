"""Ordered-latent value expansion for FB rewards, without a pose tracking loss."""

import torch
from collections.abc import Sequence
from torch import nn
from torch.nn import functional as F

from .world_model import frozen, mlp, quat_apply, quat_conjugate, yaw_quat
from .world_model_mbfb_diagnostics import return_statistics
from .world_model_sequence import sequence_lambda_returns


class FBSequenceValue(nn.Module):
    """Estimate terminal returns conditioned on an ordered window of latents."""

    def __init__(self, base_dim: int, z_dim: int, length: int, hidden_dims: Sequence[int], heads: int) -> None:
        super().__init__()
        self.net = mlp(base_dim + length * z_dim, hidden_dims, heads, zero_output=True)

    def forward(self, base: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Predict one value per reward head from state features and latent order."""
        return self.net(torch.cat((base, latents.flatten(1)), -1))


def value_base(model, agent, state, context, obs, commands):
    """Combine local physical state, actor observations and heading-frame commands."""
    heading, velocity, valid = commands
    inverse = quat_conjugate(yaw_quat(state[:, 3:7]))
    heading = quat_apply(inverse, F.pad(heading, (0, 1)))[:, :2] * valid[:, :1]
    dim = velocity.shape[-1]
    velocity = quat_apply(inverse, F.pad(velocity, (0, 3 - dim)))[:, :dim] * valid[:, 1:2]
    return torch.cat(
        (
            model.codec.local_state(state[:, 2:61]) / model.codec.state_scale,
            context,
            agent._actor_input(obs),
            heading,
            velocity,
            valid,
        ),
        -1,
    )


def value_chunks(model, bases, latents, steps):
    """Materialize only a minibatch of ordered windows, not a B x H x K copy."""
    count = bases.shape[1]
    for start in range(0, steps * count, model.cfg.batch_size):
        index = torch.arange(start, min(start + model.cfg.batch_size, steps * count), device=bases.device)
        time, row = index // count, index % count
        window = time[:, None] + torch.arange(model.cfg.sequence_context_length, device=bases.device)
        yield time, row, bases[time, row], latents[row[:, None], window]


@torch.no_grad()
def sequence_directions(agent, goal, latents, horizon, *, ridge=0.0):
    """Convert latent targets to reward directions using the backward covariance."""
    with agent._autocast_context():
        embeddings = agent._backward_map(goal).float()
    covariance = embeddings.T @ embeddings / len(embeddings)
    if ridge:
        diagonal = covariance.diagonal().mean().clamp_min(torch.finfo(covariance.dtype).eps)
        covariance = covariance + ridge * diagonal * torch.eye(covariance.shape[0], device=covariance.device)
    directions = torch.linalg.solve(covariance, latents[:, :horizon].flatten(0, 1).T).T
    return directions.reshape(len(latents), horizon, -1)


def sequence_svg_loss(
    model, agent, state, context, obs, latents, goal, commands, directions=None, *, trajectory_weights=None
):
    """Backpropagate imagined rewards and terminal value through the actor rollout."""
    horizon, gamma, length = model.cfg.horizon, agent.cfg.discount, model.cfg.sequence_context_length
    if latents.shape[1] != horizon + length:
        raise ValueError("FB sequence value requires H+K latents, including a complete terminal window")
    model._value_batch = None

    def commands_at(step):
        return tuple(value[:, step] if value.ndim == 3 else value for value in commands)

    if directions is None:
        directions = sequence_directions(agent, goal, latents, horizon, ridge=model.cfg.fb_covariance_ridge)
    bases, privileged, physical_states, normalized_states = [], [], [], []
    privileged_keys = tuple(getattr(agent, "privileged_keys", ()))
    deferred_geometry = privileged_keys in (
        ("privileged_state",),
        ("state", "privileged_state"),
    ) and "privileged_state" not in getattr(agent, "actor_keys", ("privileged_state",))
    with frozen(
        model.dynamics,
        agent._backward_map,
        agent._discriminator,
        model.sequence_value,
        getattr(agent._actor, "velocity_estimator", None),
    ):
        normalized = agent._obs_normalizer.normalize_dict(obs)
        privileged.append(agent._privileged_input(normalized))
        for step in range(horizon):
            with torch.no_grad():
                bases.append(value_base(model, agent, state, context, normalized, commands_at(step)).detach())
            action = model.action(agent, obs, latents[:, step], svg=True, normalized=normalized)
            state = model.dynamics(state, model.codec.env_action(action), context)
            if deferred_geometry:
                obs = model.observe(state, obs, *commands_at(step + 1), privileged=False)
                physical_states.append(state[:, 2:61])
            else:
                obs = model.observe(state, obs, *commands_at(step + 1))
            normalized = agent._obs_normalizer.normalize_dict(obs)
            if deferred_geometry and "state" in privileged_keys:
                normalized_states.append(normalized["state"])
            elif not deferred_geometry:
                privileged.append(agent._privileged_input(normalized))
        terminal_base = value_base(model, agent, state, context, normalized, commands_at(horizon))
        bases.append(terminal_base.detach())
        terminal = model.sequence_value(terminal_base, latents[:, horizon:])
        # The actor needs proprioception only; reconstruct full-body states after rollout.
        if deferred_geometry:
            physical_states = torch.cat(physical_states)
            normalized_states = torch.cat(normalized_states) if normalized_states else None
            next_inputs = []
            for start in range(0, len(physical_states), model.cfg.scoring_chunk_size):
                part = slice(start, start + model.cfg.scoring_chunk_size)
                parts = agent._obs_normalizer.normalize_dict({
                    "privileged_state": model.kinematics(physical_states[part])
                })
                if normalized_states is not None:
                    parts["state"] = normalized_states[part]
                next_inputs.append(agent._privileged_input(parts))
            next_inputs = torch.cat(next_inputs)
        else:
            next_inputs = torch.cat(privileged[1:])
        current_inputs = (
            torch.cat((privileged[0], next_inputs[: -len(state)])) if model.cfg.discriminator_reward else None
        )
        directions = directions.transpose(0, 1).flatten(0, 1)
        reward_latents = latents[:, :horizon].transpose(0, 1).flatten(0, 1)
        rewards = []
        # Preserve BF16 network batch shapes; only FP32 geometry uses larger chunks.
        for start in range(0, len(next_inputs), len(state)):
            part = slice(start, start + len(state))
            with agent._autocast_context():
                backward = agent._backward_map(next_inputs[part]).float()
            reward = (backward * directions[part]).sum(-1, keepdim=True)
            if model.cfg.discriminator_reward:
                with agent._autocast_context():
                    logits = agent._discriminator.compute_logits(current_inputs[part], reward_latents[part])
                # Convert before sigmoid/log odds: BF16 rounds sigmoid(7) to one.
                disc = agent.discriminator_reward_from_logits(logits)
                reward = torch.cat((reward, disc.float().reshape(-1, 1)), -1)
            rewards.append(reward)
        rewards = torch.cat(rewards).reshape(horizon, len(state), -1).transpose(0, 1)
    discounts = gamma ** torch.arange(horizon, device=state.device)
    returns = (rewards * discounts[None, :, None]).sum(1) + gamma**horizon * terminal
    fb_return = returns[:, 0]
    disc_return = returns[:, 1] if model.cfg.discriminator_reward else torch.zeros_like(fb_return)
    scale = fb_return.abs().mean().detach() if agent.cfg.scale_reg else 1.0
    loss = -fb_return.mean() - agent.cfg.reg_coeff * scale * disc_return.mean()
    if trajectory_weights is not None:
        # Probe subsets without changing network batch shapes or the full-batch reward scale.
        weights = trajectory_weights / trajectory_weights.sum()
        loss = -(fb_return * weights).sum() - agent.cfg.reg_coeff * scale * (disc_return * weights).sum()
    model._value_batch = (torch.stack(bases), latents.detach(), rewards.detach())
    metrics = {
        "mb/svg_Q_fb": fb_return.detach().mean(),
        "mb/svg_Q_discriminator": disc_return.detach().mean(),
        "wm/sequence_value_terminal": terminal.detach().mean(),
    }
    if getattr(model.cfg, "diagnostics", True) and trajectory_weights is None:
        metrics.update(return_statistics(returns, terminal, gamma**horizon))
    return loss, metrics


def sequence_value_gradients(model, discount, batch):
    """Accumulate value gradients in chunks without retaining all window activations."""
    bases, latents, rewards = batch
    horizon, count, heads = model.cfg.horizon, bases.shape[1], rewards.shape[-1]
    target_value = getattr(model, "target_sequence_value", model.sequence_value)
    with torch.no_grad():
        values = bases.new_empty(horizon + 1, count, heads)
        for time, row, base, window in value_chunks(model, bases, latents, horizon + 1):
            values[time, row] = target_value(base, window)
        targets = sequence_lambda_returns(rewards, values.transpose(0, 1), discount, model.cfg.value_lambda).transpose(
            0, 1
        )
    total_loss = bases.new_zeros(())
    parameters = tuple(model.sequence_value.parameters())
    gradients = tuple(torch.zeros_like(p) for p in parameters)
    for time, row, base, window in value_chunks(model, bases, latents, horizon):
        prediction = model.sequence_value(base, window)
        loss = 0.5 * (prediction - targets[time, row]).square().sum() / (horizon * count * heads)
        chunk = torch.autograd.grad(loss, parameters)
        gradients = tuple(total + value for total, value in zip(gradients, chunk))
        total_loss += loss.detach()
    return total_loss, gradients


def fit_sequence_value(model, agent):
    """Fit detached imagination targets for several epochs with a frozen target critic."""
    from .world_model_mbfb import all_ranks_true

    batch, model._value_batch = model._value_batch, None
    if batch is None:
        return {}
    bases, latents, rewards = batch
    optimizer = model.sequence_value_optimizer
    finite = torch.isfinite(bases).all() & torch.isfinite(rewards).all() & torch.isfinite(latents).all()
    if not all_ranks_true(finite):
        return {"wm/sequence_value_skipped": bases.new_ones(())}
    graph = None
    if getattr(model.cfg, "cuda_graph_value", False) and model.device.type == "cuda":
        from .world_model_mbfb_graph import ValueGraph

        if model._value_graph is None or not model._value_graph.matches(dict(batch=batch)):
            model._value_graph = None
            model._value_graph = ValueGraph(model, agent.cfg.discount, batch)
        graph = model._value_graph
    losses = []
    success = True
    for _ in range(model.cfg.value_fit_epochs):
        optimizer.zero_grad(set_to_none=True)
        if graph is not None:
            total_loss, gradients = graph.replay(dict(batch=batch))
        else:
            total_loss, gradients = sequence_value_gradients(model, agent.cfg.discount, batch)
        for parameter, gradient in zip(model.sequence_value.parameters(), gradients):
            parameter.grad = gradient
        agent._avg_grad(model.sequence_value.parameters())
        norm = nn.utils.clip_grad_norm_(model.sequence_value.parameters(), 1.0)
        success = all_ranks_true(torch.isfinite(total_loss) & torch.isfinite(norm))
        if not success:
            optimizer.zero_grad(set_to_none=True)
            break
        optimizer.step()
        model.sequence_value_updates.add_(1)
        losses.append(total_loss.detach())
    if success and losses:
        with torch.no_grad():
            for source, target in zip(model.sequence_value.parameters(), model.target_sequence_value.parameters()):
                target.lerp_(source, model.cfg.value_tau)
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.stack(losses).mean() if losses else bases.new_zeros(())
    return {
        "wm/sequence_value_loss": torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0),
        "wm/sequence_value_updates": model.sequence_value_updates.float().clone(),
        "wm/sequence_value_skipped": bases.new_tensor(float(not success)),
    }
