# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Running normalization for foothold predictor inputs."""

from __future__ import annotations

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Distributed running normalization with checkpointed statistics."""

    def __init__(self, shape: tuple[int, ...], device: torch.device | str) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(shape, dtype=torch.float32, device=device))
        self.register_buffer("_var", torch.ones(shape, dtype=torch.float32, device=device))
        self.register_buffer("count", torch.zeros((), dtype=torch.float32, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        batch_sum = values.sum(dim=0)
        batch_sq_sum = values.square().sum(dim=0)
        batch_count = torch.tensor(
            float(values.shape[0]), dtype=torch.float32, device=values.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            feature_count = batch_sum.numel()
            packed = torch.cat(
                (batch_sum.flatten(), batch_sq_sum.flatten(), batch_count.view(1))
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            batch_sum = packed[:feature_count].view_as(batch_sum)
            batch_sq_sum = packed[feature_count : 2 * feature_count].view_as(
                batch_sq_sum
            )
            batch_count = packed[-1]

        batch_mean = batch_sum / batch_count
        batch_var = (batch_sq_sum / batch_count - batch_mean.square()).clamp_min(0.0)
        total_count = self.count + batch_count
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * (batch_count / total_count)
        old_m2 = self._var * self.count
        batch_m2 = batch_var * batch_count
        new_m2 = old_m2 + batch_m2 + delta.square() * self.count * batch_count / total_count
        self._mean.copy_(new_mean)
        self._var.copy_(new_m2 / total_count)
        self.count.copy_(total_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self._mean) / torch.sqrt(self._var + 1.0e-6)
        return normalized.clamp(-10.0, 10.0)
