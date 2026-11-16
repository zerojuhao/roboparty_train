"""FB value expansion with batched geometry and microbatch-safe reductions."""

import torch

from .world_model import frozen


def quality_metrics(model):
    cfg = model.cfg
    validated = int(model.validated_at) >= 0
    fresh = validated and 0 <= int(model.updates - model.validated_at) <= cfg.validation_interval
    ready = fresh and bool(model.validation_p95 <= cfg.validation_max_p95)
    quality = model.validation_p95.detach().clone() if validated else model.validation_p95.new_zeros(())
    metrics = {key: model.validation_p95.new_zeros(()) for key in (
        "wm/context_fraction", "wm/no_push_fraction", "wm/fb_projection_fraction",
        "wm/fb_projection_residual", "wm/recorded_z_relative_drift", "wm/trusted_fraction",
        "wm/action_radius_fraction", "wm/trusted_prefix_steps_mean",
        "wm/imagination_action_distance", "wm/accepted_count",
        "wm/actor_loss", "wm/actor_weight", "wm/imagined_return", "wm/imagination_updates",
        "wm/reward_sum", "wm/terminal_value", "wm/terminal_contribution", "wm/reward_weights_norm",
        "wm/sequence_value_ready", "wm/sequence_value_updates", "wm/sequence_value_loss",
        "wm/sequence_value_target",
        "wm/discriminator_reward_sum", "wm/discriminator_terminal_value",
        "wm/discriminator_return", "wm/discriminator_weight", "wm/fb_return",
        "wm/tracking_fraction", "wm/tracking_pose_loss", "wm/tracking_return", "wm/tracking_weight",
        "wm/tracking_body_position_mse", "wm/tracking_rotation_chordal_mse", "wm/tracking_height_mse",
    )}
    for step in {1, 4, 8, 16, 32, cfg.horizon}:
        if step <= cfg.horizon:
            metrics[f"wm/trusted_fraction_h{step}"] = model.validation_p95.new_zeros(())
    metrics.update({
        "wm/quality_ready": model.updates.new_tensor(float(ready), dtype=torch.float32),
        "wm/quality_validated": model.updates.new_tensor(float(validated), dtype=torch.float32),
        "wm/quality_p95": quality,
        "wm/imagination_horizon": model.updates.new_tensor(cfg.horizon, dtype=torch.float32),
    })
    return ready, metrics


def context_masks(model, batch, expected_length, description):
    if "context_id" not in batch:
        raise ValueError("FB imagination requires replay context IDs; collect fresh replay")
    if batch["z"].shape[1] != expected_length:
        raise ValueError(f"FB imagination needs {description} aligned states and command contexts")
    same_context = (batch["context_id"] == batch["context_id"][:, :1]).flatten(1).all(-1)
    no_push = ~batch["disturbance_mask"][:, :model.cfg.horizon].bool().flatten(1).any(-1)
    return same_context, no_push


@torch.no_grad()
def reward_projection(model, latents):
    covariance = model.fb_covariance.detach().float()
    mean_diagonal = covariance.diagonal().mean()
    if not torch.isfinite(covariance).all() or not bool(mean_diagonal > 0):
        return None
    ridge = model.cfg.fb_covariance_ridge * mean_diagonal
    matrix = covariance + ridge * torch.eye(covariance.shape[0], device=covariance.device)
    flat = latents.detach().float().reshape(-1, latents.shape[-1])
    weights = torch.linalg.solve(matrix, flat.T).T
    effective = weights @ covariance.T
    residual = torch.linalg.vector_norm(effective - flat, dim=-1)
    residual /= torch.linalg.vector_norm(flat, dim=-1).clamp_min(1e-8)
    return weights.reshape_as(latents), effective.reshape_as(latents), residual.reshape(latents.shape[:-1])


def rollout(model, agent, batch, latents, trusted, metrics):
    """Retain one noisy observation per state and the complete physical graph."""
    cfg = model.cfg
    state, xi, previous_action, history = (
        batch[key][:, 0].detach() for key in ("state", "xi", "previous_action", "history")
    )
    member = int(model.updates) % cfg.ensemble_size
    action_ok = torch.ones_like(trusted)
    prefix = torch.zeros_like(trusted, dtype=torch.float32)
    action_distances = []
    states, observations = [state], []
    obs = model.codec.observation(state, previous_action, history, batch["heading"][:, 0].detach(),
                                  batch["velocity"][:, 0].detach(), batch["valid"][:, 0].detach(), noisy=True)
    normalized = agent._obs_normalizer.normalize_dict(obs)
    observations.append(normalized)
    for step in range(cfg.horizon):
        with agent._autocast_context():
            distribution = agent._actor_forward(normalized, latents[:, step], imagination=True)
            action = distribution.sample(clip=agent.cfg.stddev_clip).float()
        prediction = model.predict(state, action, xi, member)
        with torch.no_grad():
            distance = ((model.codec.env_action(action) - model.codec.env_action(batch["action"][:, step]))
                        / model.codec.action_rescale).abs().amax(-1)
            action_ok &= distance <= cfg.imagination_action_radius
            trusted = trusted & action_ok
            prefix += trusted.float()
            if step + 1 in {1, 4, 8, 16, 32, cfg.horizon}:
                metrics[f"wm/trusted_fraction_h{step + 1}"] = trusted.float().mean()
            action_distances.append(distance)
        history = model.codec.advance_history(history, obs)
        previous_action = model.codec.env_action(action)
        state = prediction
        states.append(state)
        obs = model.codec.observation(state, previous_action, history, batch["heading"][:, step + 1].detach(),
                                      batch["velocity"][:, step + 1].detach(),
                                      batch["valid"][:, step + 1].detach(), noisy=True)
        normalized = agent._obs_normalizer.normalize_dict(obs)
        observations.append(normalized)
    metrics.update({
        "wm/trusted_fraction": trusted.float().mean(),
        "wm/action_radius_fraction": action_ok.float().mean(),
        "wm/trusted_prefix_steps_mean": prefix.mean(),
        "wm/imagination_action_distance": torch.stack(action_distances).mean(),
    })
    return states, observations, trusted


def score_backward(model, agent, states, observations):
    """Batch FK/B over time without resampling observation noise or detaching."""
    batch_size, horizon = states[0].shape[0], len(states) - 1
    physical = torch.stack(states[1:], dim=1).flatten(0, 1)
    flat_obs = {key: torch.stack([obs[key] for obs in observations[1:]], dim=1).flatten(0, 1)
                for key in observations[0]}
    embeddings, privileged, poses = [], [], []
    chunk_size = int(getattr(model.cfg, "scoring_chunk_size", 2048))
    for start in range(0, physical.shape[0], chunk_size):
        stop = start + chunk_size
        parts = {key: value[start:stop] for key, value in flat_obs.items()}
        body = model.kinematics(physical[start:stop])
        if getattr(model.cfg, "tracking_coef", 0) > 0:
            from .world_model_tracking import POSE_DIM

            poses.append(body[:, :POSE_DIM])
        parts.update(agent._obs_normalizer.normalize_dict({"privileged_state": body}))
        with agent._autocast_context():
            embeddings.append(agent._target_backward_map(agent._privileged_input(parts)).float())
        privileged.append(parts["privileged_state"])
    body_observations = torch.cat(privileged).reshape(batch_size, horizon, -1)
    for step, obs in enumerate(observations[1:]):
        obs["privileged_state"] = body_observations[:, step]
    if poses:
        raw = torch.cat(poses).reshape(batch_size, horizon, -1)
        for step, obs in enumerate(observations[1:]):
            obs["world_model_pose"] = raw[:, step]
    return torch.cat(embeddings).reshape(batch_size, horizon, -1)


def score_discriminator(model, agent, states, observations, latents):
    """Score current states, matching the real discriminator critic's reward timing."""
    body = model.kinematics(states[0])
    observations[0].update(agent._obs_normalizer.normalize_dict({"privileged_state": body}))
    inputs = torch.stack([agent._privileged_input(obs) for obs in observations[:-1]], dim=1)
    batch_size, horizon = inputs.shape[:2]
    inputs, z = inputs.flatten(0, 1), latents[:, :horizon].flatten(0, 1)
    rewards = []
    for start in range(0, inputs.shape[0], model.cfg.scoring_chunk_size):
        stop = start + model.cfg.scoring_chunk_size
        with agent._autocast_context():
            logits = agent._discriminator.compute_logits(obs=inputs[start:stop], z=z[start:stop])
        rewards.append(agent.discriminator_reward_from_logits(logits))
    return torch.cat(rewards).reshape(batch_size, horizon)


def discriminator_weight(agent, scale, reference):
    # One detached real-batch FB scale is shared by every imagined microbatch.
    scale = scale if agent.cfg.scale_reg else 1.0
    return reference.new_tensor(agent.cfg.reg_coeff) * torch.as_tensor(
        scale, device=reference.device, dtype=reference.dtype).detach()


def discriminator_return_metrics(metrics, reward_sum, terminal, discount, coefficient):
    returns = reward_sum + discount * terminal
    metrics.update({"wm/discriminator_reward_sum": reward_sum.detach().mean(),
                    "wm/discriminator_terminal_value": terminal.detach().mean(),
                    "wm/discriminator_return": returns.detach().mean(),
                    "wm/discriminator_weight": coefficient.detach()})
    return returns


def accepted_count(trusted, metrics, update_counters):
    count = trusted.sum().detach()
    metrics["wm/accepted_count"] = count.float()
    if update_counters and torch.distributed.is_initialized():
        count = count.clone()
        torch.distributed.all_reduce(count)
    return count


def actor_return_loss(model, returns, trusted, metrics, *, total_batch_size=None,
                      actor_weight=None, update_counters=True):
    count = accepted_count(trusted, metrics, update_counters)
    if update_counters and int(count) == 0:
        return None
    next_update = int(model.imagination_updates) + 1
    weight = (model.cfg.actor_weight * min(1.0, next_update / model.cfg.ramp_updates)
              if actor_weight is None else actor_weight)
    if update_counters:
        model.imagination_updates.add_(1)
    accepted = torch.where(trusted, returns, 0.0)
    denominator = returns.shape[0] if total_batch_size is None else total_batch_size
    loss = -weight * accepted.sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite FB imagination loss")
    metrics.update({"wm/actor_loss": loss.detach(), "wm/actor_weight": returns.new_tensor(weight),
                    "wm/imagined_return": accepted.detach().mean(),
                    "wm/imagination_updates": model.imagination_updates.float().clone()})
    return loss


def return_metrics(metrics, reward_sum, terminal, discount, weights):
    metrics.update({"wm/reward_sum": reward_sum.detach().mean(),
                    "wm/terminal_value": terminal.detach().mean(),
                    "wm/terminal_contribution": discount * terminal.detach().mean(),
                    "wm/reward_weights_norm": weights.detach().norm(dim=-1).mean()})


def fb_imagination_loss(model, agent, batch, *, total_batch_size=None, actor_weight=None,
                        update_counters=True, discriminator_scale=1.0):
    """Fixed-z reward and its paired terminal F; optional externally managed update."""
    cfg = model.cfg
    ready, metrics = quality_metrics(model)
    if not ready:
        return None, metrics
    same_context, no_push = context_masks(model, batch, cfg.horizon + 1, "H+1")
    z = batch["z"][:, 0].detach()
    with torch.no_grad():
        drift = torch.linalg.vector_norm(batch["z"].detach().float() - z[:, None].float(), dim=-1)
        drift /= torch.linalg.vector_norm(z.float(), dim=-1).clamp_min(1e-8)[:, None]
        metrics["wm/recorded_z_relative_drift"] = drift.amax(-1).mean()
    projection = reward_projection(model, z)
    if projection is None:
        metrics["wm/fb_projection_fraction"] = z.new_zeros(())
        return None, metrics
    weights, effective_z, residual = projection
    projection_ok = torch.isfinite(residual) & (residual <= cfg.fb_max_projection_residual)
    trusted = same_context & no_push & projection_ok
    metrics.update({"wm/context_fraction": same_context.float().mean(), "wm/no_push_fraction": no_push.float().mean(),
                    "wm/fb_projection_fraction": projection_ok.float().mean(),
                    "wm/fb_projection_residual": residual.mean()})
    latents = z[:, None].expand(-1, cfg.horizon + 1, -1)
    use_discriminator = getattr(cfg, "discriminator_reward", False)
    critic = ((agent._target_critic_shared if agent.shared_value_net else agent._target_critic)
              if use_discriminator else None)
    with frozen(model.members, agent._target_backward_map, agent._target_forward_map,
                getattr(agent._actor, "velocity_estimator", None), critic,
                agent._discriminator if use_discriminator else None):
        states, observations, trusted = rollout(model, agent, batch, latents, trusted, metrics)
        backward = score_backward(model, agent, states, observations)
        rewards = (backward * weights[:, None]).sum(-1)
        discounts = agent.cfg.discount ** torch.arange(cfg.horizon, device=rewards.device)
        returns = (rewards * discounts).sum(-1)
        with frozen(agent._actor), agent._autocast_context():
            distribution = agent._actor_forward(observations[-1], z, imagination=True)
            terminal_action = distribution.sample(clip=agent.cfg.stddev_clip).float()
        with agent._autocast_context():
            forward = agent._target_forward_map(agent._full_input(observations[-1]), z, terminal_action)
        values = (forward.float() * effective_z).sum(-1)
        _, _, terminal = agent._targets_uncertainty(values, agent.cfg.actor_pessimism_penalty)
        terminal = terminal.reshape(-1)
        return_metrics(metrics, returns, terminal, agent.cfg.discount ** cfg.horizon, weights)
        returns = returns + agent.cfg.discount ** cfg.horizon * terminal
        metrics["wm/fb_return"] = returns.detach().mean()
        if use_discriminator:
            disc_rewards = score_discriminator(model, agent, states, observations, latents)
            with agent._autocast_context():
                qs = critic(agent._full_input(observations[-1]), z, terminal_action)
                if agent.shared_value_net:
                    qs = qs[0]
            _, _, disc_terminal = agent._targets_uncertainty(qs.float(), agent.cfg.actor_pessimism_penalty)
            coefficient = discriminator_weight(agent, discriminator_scale, returns)
            disc_returns = discriminator_return_metrics(
                metrics, (disc_rewards * discounts).sum(-1), disc_terminal.reshape(-1),
                agent.cfg.discount ** cfg.horizon, coefficient)
            returns = returns + coefficient * disc_returns
    loss = actor_return_loss(model, returns, trusted, metrics, total_batch_size=total_batch_size,
                             actor_weight=actor_weight, update_counters=update_counters)
    return loss, metrics
