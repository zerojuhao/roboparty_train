"""Memory-bounded, single-step policy updates over complete imagined batches."""

import torch
from torch import nn


def add_imagination_gradients(model, agent):
    """Accumulate trajectories, then reduce/clip once and add to the UFO gradient."""
    from .world_model_imagination import fb_imagination_loss
    from .world_model_sequence import fb_sequence_imagination_loss

    cfg, batch = model.cfg, model._imagination_batch
    model._imagination_batch = None
    if batch is None or cfg.actor_weight == 0 or int(model.updates) <= cfg.warmup_updates:
        return {}
    if cfg.objective not in {"fb_value", "fb_sequence"}:
        model._imagination_batch = batch
        loss, metrics = model.actor_loss(agent)
        if loss is not None:
            metrics.update(model.add_actor_gradients(agent, loss))
        return metrics

    size = batch["state"].shape[0]
    parameters = tuple(parameter for parameter in agent._actor.parameters() if parameter.requires_grad)
    original = tuple(parameter.grad for parameter in parameters)
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    weight = cfg.actor_weight * min(1.0, (int(model.imagination_updates) + 1) / cfg.ramp_updates)
    sequence = cfg.objective == "fb_sequence"
    if sequence:
        model.sequence_value_optimizer.zero_grad(set_to_none=True)
    metrics = {}
    totals = batch["state"].new_zeros(5)
    for start in range(0, size, cfg.imagination_microbatch_size):
        end = min(start + cfg.imagination_microbatch_size, size)
        chunk = {key: value[start:end] for key, value in batch.items()}
        kwargs = dict(total_batch_size=size, actor_weight=weight, update_counters=False)
        if sequence:
            loss, value_loss, chunk_metrics = fb_sequence_imagination_loss(model, agent, chunk, **kwargs)
        else:
            loss, chunk_metrics = fb_imagination_loss(model, agent, chunk, **kwargs)
            value_loss = None
        for name, value in chunk_metrics.items():
            if name != "wm/accepted_count":
                metrics[name] = metrics.get(name, 0) + value.detach() * ((end - start) / size)
        accepted = chunk_metrics.get("wm/accepted_count", totals.new_zeros(()))
        totals[0] += accepted
        if loss is not None and bool(accepted > 0):
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            for accumulated_gradient, gradient in zip(accumulated, gradients):
                if gradient is not None:
                    accumulated_gradient.add_(gradient)
            totals[1] += 1
            totals[3] += loss.detach()
            del gradients
        if value_loss is not None and bool(accepted > 0):
            value_loss.backward()
            totals[2] += 1
            totals[4] += value_loss.detach()
        del loss, value_loss, chunk, chunk_metrics

    global_counts = totals[:3].clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_counts)
    if global_counts[1] > 0:
        try:
            for parameter, gradient in zip(parameters, accumulated):
                parameter.grad = gradient
            agent._avg_grad(parameters)
            norm = nn.utils.clip_grad_norm_(parameters, cfg.grad_clip, error_if_nonfinite=True)
            addition = tuple(parameter.grad for parameter in parameters)
        finally:
            for parameter, gradient in zip(parameters, original):
                parameter.grad = gradient
        with torch.no_grad():
            zero = totals.new_zeros(())
            base_square = sum((gradient.square().sum() for gradient in original if gradient is not None), zero)
            extra_square = sum((gradient.square().sum() for gradient in addition), zero)
            dot = sum(((base * extra).sum() for base, extra in zip(original, addition) if base is not None), zero)
            for parameter, extra in zip(parameters, addition):
                if parameter.grad is None:
                    parameter.grad = extra
                else:
                    parameter.grad.add_(extra)
        model.imagination_updates.add_(1)
        metrics.update({"wm/actor_grad_norm": norm.detach(),
                        "wm/actor_grad_ratio": extra_square.sqrt() / base_square.sqrt().clamp_min(1e-8),
                        "wm/actor_grad_cosine": dot / (base_square * extra_square).sqrt().clamp_min(1e-8),
                        "wm/actor_weight": totals.new_tensor(weight)})
    if sequence and global_counts[2] > 0:
        for parameter in model.sequence_value.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        agent._avg_grad(model.sequence_value.parameters())
        value_norm = nn.utils.clip_grad_norm_(model.sequence_value.parameters(), cfg.grad_clip,
                                           error_if_nonfinite=True)
        model.sequence_value_optimizer.step()
        model.sequence_value_updates.add_(1)
        with torch.no_grad():
            for source, target in zip(model.sequence_value.parameters(), model.target_sequence_value.parameters()):
                target.lerp_(source, cfg.value_tau)
        metrics["wm/sequence_value_grad_norm"] = value_norm.detach()
        metrics["wm/sequence_value_updates"] = model.sequence_value_updates.float().clone()
    metrics.update({"wm/actor_loss": totals[3], "wm/accepted_count": totals[0],
                    "wm/imagination_batch_size": totals.new_tensor(size),
                    "wm/imagination_microbatches": totals.new_tensor((size - 1) // cfg.imagination_microbatch_size + 1),
                    "wm/imagination_updates": model.imagination_updates.float().clone()})
    if sequence:
        metrics["wm/sequence_value_loss"] = totals[4]
    if torch.distributed.is_initialized():
        # Rejected ranks still participate in identical downstream metric reductions.
        keys = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(keys, tuple(metrics))
        for name in set().union(*keys):
            metrics.setdefault(name, totals.new_zeros(()))
    return metrics
