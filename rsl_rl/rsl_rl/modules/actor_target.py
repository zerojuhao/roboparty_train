"""Training-only supervision of position targets in a permitted effort interval."""

import torch


def actor_target_loss(mean, labels):
    """Labels: lower, upper, torque/action scale, loaded gate, return gate."""
    if mean.ndim != 2 or labels.shape != (*mean.shape[:-1], mean.shape[-1] * 5):
        raise ValueError('Actor target labels must have five fields per action')
    lower, upper, scale, loaded, returning = labels.detach().float().reshape(*mean.shape, 5).unbind(-1)
    residual = (mean.float() - mean.float().clamp(min=lower, max=upper)) * scale
    square = residual.square()
    cost = square / (torch.sqrt(1 + square) + 1)
    # Rare loaded joints must not be diluted by unrelated/returning environments.
    # Count samples, not gate weight: partial deadzone activation stays partial.
    count = (loaded > 0).sum(0)
    per_joint = (cost * loaded).sum(0) / count.clamp_min(1)
    loaded_loss = per_joint.sum() / (count > 0).sum().clamp_min(1)
    return_loss = ((cost * returning).sum(-1) / returning.sum(-1).clamp_min(1)).mean()
    return loaded_loss + return_loss, torch.maximum(loaded, returning).mean()
