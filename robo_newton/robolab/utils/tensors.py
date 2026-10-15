"""Small tensor adapters for IsaacLab and Newton-backed scene data."""

from __future__ import annotations

import torch


def as_torch_tensor(value) -> torch.Tensor:
    """Return a scene data value as a Torch tensor without copying it."""
    return value if isinstance(value, torch.Tensor) else value.torch
