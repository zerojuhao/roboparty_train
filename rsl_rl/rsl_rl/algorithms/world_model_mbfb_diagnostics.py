"""Read-only diagnostics for imagined returns, model error and sampling gradients."""

import torch

from .mbfb_dynamics import rollout_error


@torch.no_grad()
def return_statistics(returns: torch.Tensor, terminal: torch.Tensor, discount: float) -> dict[str, torch.Tensor]:
    """Summarize each reward head within one rank, without assuming independent trajectories."""
    finite = torch.isfinite(returns) & torch.isfinite(terminal)
    values = torch.nan_to_num(returns.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    bootstrap = discount * torch.nan_to_num(terminal.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    rollout = values - bootstrap
    levels = torch.linspace(0.1, 0.9, 3, device=values.device, dtype=values.dtype)
    quantiles = torch.quantile(values, levels, dim=0)
    spread = values.std(dim=0, unbiased=False)
    fraction = bootstrap.abs() / (rollout.abs() + bootstrap.abs()).clamp_min(1e-12)
    metrics = {}
    for head, name in enumerate(("fb", "discriminator")[: values.shape[1]]):
        prefix = f"wm/returns/{name}/"
        metrics.update({
            prefix + "std_local": spread[head],
            prefix + "p10_local": quantiles[0, head],
            prefix + "p50_local": quantiles[1, head],
            prefix + "p90_local": quantiles[2, head],
            prefix + "rollout_mean": rollout[:, head].mean(),
            prefix + "terminal_mean": bootstrap[:, head].mean(),
            prefix + "terminal_abs_fraction": fraction[:, head].mean(),
            prefix + "nonfinite_fraction": (~finite[:, head]).float().mean(),
        })
    return metrics


@torch.no_grad()
def validate_horizons(model) -> dict[str, torch.Tensor]:
    """Measure uninterrupted held-out rollouts without advancing the training RNG."""
    horizons = sorted({model.cfg.horizon, *model.cfg.validation_horizons})
    horizons = [step for step in horizons if step <= model.cfg.horizon]
    available = model.replay.size >= max(horizons)
    metrics = {"wm/validation/horizon_available": model.calls.new_tensor(float(available), dtype=torch.float32)}
    if not available:
        return metrics
    devices = [model.device.index] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        batch = model.replay.sample(model.cfg.validation_batch_size, max(horizons), validation=True)
        errors = rollout_error(model.dynamics, batch)
        baseline = rollout_error(model.dynamics, batch, baseline=True)
    rows = []
    for step in horizons:
        uninterrupted = batch["continues"][:, :step].all(dim=1)
        finite = torch.isfinite(errors[:, step - 1]) & torch.isfinite(baseline[:, step - 1])
        valid = uninterrupted & finite
        rows.append(
            torch.stack((
                torch.where(valid, errors[:, step - 1], 0.0).sum(),
                torch.where(valid, baseline[:, step - 1], 0.0).sum(),
                valid.float().sum(),
                uninterrupted.float().sum(),
                errors.new_tensor(len(errors)),
            ))
        )
    totals = torch.stack(rows)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    for step, (error, reference, count, uninterrupted, total) in zip(horizons, totals):
        prefix = f"wm/validation/h{step}/"
        metrics.update({
            prefix + "model_error": error / count.clamp_min(1),
            prefix + "baseline_error": reference / count.clamp_min(1),
            prefix + "ratio": error / reference.clamp_min(1e-12),
            prefix + "valid_samples_global": count,
            prefix + "uninterrupted_fraction": uninterrupted / total,
            prefix + "nonfinite_fraction": (uninterrupted - count) / uninterrupted.clamp_min(1),
        })
    return metrics


def sampling_gradient_probe(model, agent, svg) -> dict[str, torch.Tensor]:
    """Compare disjoint half-batch objectives at fixed weights, shapes and reward scale.

    This is a local sampling-noise diagnostic, not a comparison of independent
    full batches or the final distributed optimizer gradient.
    """
    from .world_model_mbfb_sequence import sequence_svg_loss

    count = len(svg["state"])
    if count < 2:
        return {"wm/probe/available": svg["state"].new_zeros(())}
    parameters = tuple(parameter for parameter in agent._actor.parameters() if parameter.requires_grad)
    modules = (model, agent._actor, agent._backward_map, agent._discriminator, agent._obs_normalizer)
    buffers = {id(buffer): (buffer, buffer.clone()) for module in modules for buffer in module.buffers()}
    previous_batch = model._value_batch
    devices = [model.device.index] if model.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            order = torch.randperm(count, device=model.device)
            first = svg["state"].new_zeros(count)
            first[order[: count // 2]] = 1
            gradients = []
            for weights in (first, 1 - first):
                # Both objectives see identical rollout noise and full-batch reward scaling.
                with torch.random.fork_rng(devices=devices):
                    loss, _ = sequence_svg_loss(model, agent, **svg, trajectory_weights=weights)
                    values = torch.autograd.grad(loss, parameters, allow_unused=True)
                gradients.append(tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)))
            left, right = gradients
            norm_left = torch.stack([value.square().sum() for value in left]).sum().sqrt()
            norm_right = torch.stack([value.square().sum() for value in right]).sum().sqrt()
            dot = torch.stack([(a * b).sum() for a, b in zip(left, right)]).sum()
            difference = torch.stack([(a - b).square().sum() for a, b in zip(left, right)]).sum().sqrt()
            finite = torch.isfinite(norm_left) & torch.isfinite(norm_right) & torch.isfinite(difference)
            nonzero = (norm_left > 1e-12) & (norm_right > 1e-12)
            return {
                "wm/probe/available": loss.new_ones(()),
                "wm/probe/finite": finite.float(),
                "wm/probe/nonzero": nonzero.float(),
                "wm/probe/half_batch_size_local": loss.new_tensor(count // 2),
                "wm/probe/other_half_batch_size_local": loss.new_tensor(count - count // 2),
                "wm/probe/gradient_cosine_local": torch.nan_to_num(
                    dot / (norm_left * norm_right).clamp_min(1e-12), nan=0.0, posinf=0.0, neginf=0.0
                ),
                "wm/probe/gradient_relative_difference_local": torch.nan_to_num(
                    difference / (0.5 * (norm_left + norm_right)).clamp_min(1e-12),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
            }
    finally:
        model._value_batch = previous_batch
        with torch.no_grad():
            for buffer, original in buffers.values():
                buffer.copy_(original)
