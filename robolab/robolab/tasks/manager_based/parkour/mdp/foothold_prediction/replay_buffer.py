# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU replay storage for foothold prediction."""

from __future__ import annotations

import torch

# Touchdown label: base-frame ``(x, y, yaw)``.
_TARGET_DIM = 3
# Must match FOOTHOLD_GUIDANCE_FAMILY_IDS: discrete, stairs-down, stairs-up.
FOOTHOLD_REPLAY_FAMILY_IDS = (1, 2, 3)
_NUM_BUCKETS = len(FOOTHOLD_REPLAY_FAMILY_IDS) * 2 * 4


class FootholdReplayBuffer:
    def __init__(self, capacity: int, input_dim: int, device: torch.device | str) -> None:
        self.inputs = torch.zeros(capacity, input_dim, dtype=torch.float16, device=device)
        self.targets = torch.zeros(capacity, _TARGET_DIM, dtype=torch.float32, device=device)
        self.foot_ids = torch.zeros(capacity, dtype=torch.long, device=device)
        self.steps_to_contact = torch.zeros(capacity, dtype=torch.int16, device=device)
        self.bucket_ids = torch.zeros(capacity, dtype=torch.int8, device=device)
        self.capacity = capacity
        self.write_index = 0
        self.size = 0

    def append(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
        steps_to_contact: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> None:
        count = inputs.shape[0]
        if count == 0:
            return
        if targets.shape[-1] != _TARGET_DIM:
            raise ValueError(
                f"Expected foothold targets last dim {_TARGET_DIM}, got {targets.shape[-1]}."
            )
        if count > self.capacity:
            inputs = inputs[-self.capacity :]
            targets = targets[-self.capacity :]
            foot_ids = foot_ids[-self.capacity :]
            steps_to_contact = steps_to_contact[-self.capacity :]
            family_ids = family_ids[-self.capacity :]
            count = self.capacity

        horizon_bins = (
            (steps_to_contact >= 4).long()
            + (steps_to_contact >= 8).long()
            + (steps_to_contact >= 16).long()
        )
        family_bins = torch.full_like(family_ids, -1, dtype=torch.long)
        for family_bin, family_id in enumerate(FOOTHOLD_REPLAY_FAMILY_IDS):
            family_bins = torch.where(
                family_ids.long() == family_id, family_bin, family_bins
            )
        if torch.any(family_bins < 0):
            invalid = torch.unique(family_ids[family_bins < 0]).tolist()
            raise ValueError(f"Unsupported foothold terrain family ids: {invalid}.")
        bucket_ids = ((family_bins * 2 + foot_ids) * 4 + horizon_bins).to(torch.int8)

        first_count = min(count, self.capacity - self.write_index)
        first_slice = slice(self.write_index, self.write_index + first_count)
        self.inputs[first_slice] = inputs[:first_count].to(torch.float16)
        self.targets[first_slice] = targets[:first_count]
        self.foot_ids[first_slice] = foot_ids[:first_count]
        self.steps_to_contact[first_slice] = steps_to_contact[:first_count].to(torch.int16)
        self.bucket_ids[first_slice] = bucket_ids[:first_count]

        remaining = count - first_count
        if remaining > 0:
            self.inputs[:remaining] = inputs[first_count:].to(torch.float16)
            self.targets[:remaining] = targets[first_count:]
            self.foot_ids[:remaining] = foot_ids[first_count:]
            self.steps_to_contact[:remaining] = steps_to_contact[first_count:].to(torch.int16)
            self.bucket_ids[:remaining] = bucket_ids[first_count:]

        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def _balanced_indices(
        self, batch_size: int, candidate_multiplier: int
    ) -> torch.Tensor:
        """Approximately balance up/down, left/right, and contact horizon."""
        candidate_count = min(
            self.size, max(batch_size * max(int(candidate_multiplier), 1), batch_size)
        )
        candidates = torch.randint(
            self.size, (candidate_count,), device=self.inputs.device
        )
        candidate_buckets = self.bucket_ids[candidates].long()
        bucket_counts = torch.bincount(candidate_buckets, minlength=_NUM_BUCKETS)
        sample_weights = bucket_counts[candidate_buckets].float().reciprocal()
        selected = torch.multinomial(sample_weights, batch_size, replacement=True)
        return candidates[selected]

    def sample(
        self,
        batch_size: int,
        balanced: bool = False,
        candidate_multiplier: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, self.size)
        if balanced and batch_size > 0:
            indices = self._balanced_indices(batch_size, candidate_multiplier)
        else:
            indices = torch.randint(
                self.size, (batch_size,), device=self.inputs.device
            )
        return (
            self.inputs[indices].to(torch.float32),
            self.targets[indices],
            self.foot_ids[indices],
            self.steps_to_contact[indices].to(torch.long),
        )

    def clear(self) -> None:
        self.write_index = 0
        self.size = 0
