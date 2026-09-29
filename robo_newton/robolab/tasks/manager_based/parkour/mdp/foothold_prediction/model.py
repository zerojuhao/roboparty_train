"""Foothold prediction network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FootholdPredictor(nn.Module):
    """MLP mapping privileged input to per-foot Gaussian parameters.

    Output layout is two feet × ``(μx, μy, μψ, σ_xy, σ_yaw)``.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ELU()))
        layers.append(nn.Linear(dims[-1], 10))
        self.network = nn.Sequential(*layers)

    def forward(self, predictor_input: torch.Tensor) -> torch.Tensor:
        """Return raw distribution parameters with shape ``[B, 2, 5]``."""
        return self.network(predictor_input).view(-1, 2, 5)
