"""Small tensor adapters for IsaacLab and Newton-backed scene data."""

from __future__ import annotations

import torch


def as_torch_tensor(value) -> torch.Tensor:
    """Return a scene data value as a Torch tensor without copying it."""
    return value if isinstance(value, torch.Tensor) else value.torch


def resolve_env_ids(env_ids, num_envs: int, device) -> torch.Tensor:
    """Convert a 3.0 reset selector (`None`, `slice`, tensor, or sequence) to indices."""
    if env_ids is None or env_ids == slice(None):
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    if torch.is_tensor(env_ids):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)
