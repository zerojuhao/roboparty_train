"""Within-clip difficulty sampling from aligned tracking evaluation."""

import math

import torch


class MotionPhaseCurriculum:
    def __init__(self, lengths, *, bin_seconds=1.0, temperature=2.0, ema=0.8):
        self.lengths = torch.as_tensor(lengths).float()
        if self.lengths.ndim != 1 or not bool((self.lengths > 0).all()):
            raise ValueError("Motion lengths must be positive")
        if bin_seconds <= 0 or temperature <= 0 or not 0 <= ema < 1:
            raise ValueError("Invalid phase curriculum parameters")
        self.bin_seconds, self.temperature, self.ema = float(bin_seconds), float(temperature), float(ema)
        bins = torch.ceil(self.lengths / bin_seconds).long().clamp_min(1)
        self.valid = torch.arange(int(bins.max()), device=bins.device)[None] < bins[:, None]
        self.errors = torch.zeros_like(self.valid, dtype=torch.float32)
        self.seen = torch.zeros_like(self.valid)
        self.history = self.valid.float() / bins[:, None]
        self.updates = torch.zeros_like(bins)

    def update(self, motion_id, errors, dt):
        if not 0 <= motion_id < len(self.lengths) or dt <= 0:
            raise ValueError("Invalid motion ID or control period")
        errors = torch.as_tensor(errors, device=self.lengths.device, dtype=torch.float32).reshape(-1)
        if not torch.isfinite(errors).all() or bool((errors < 0).any()):
            raise ValueError("Phase errors must be finite and non-negative")
        for index in range(int(self.valid[motion_id].sum())):
            start = int(math.ceil(index * self.bin_seconds / dt - 1e-6))
            end = min(len(errors), int(math.ceil((index + 1) * self.bin_seconds / dt - 1e-6)))
            if start >= end:
                continue
            value = errors[start:end].mean()
            if self.seen[motion_id, index]:
                value = self.ema * self.errors[motion_id, index] + (1 - self.ema) * value
            self.errors[motion_id, index] = value
            self.seen[motion_id, index] = True
        current = self._difficulty()[motion_id]
        count = self.updates[motion_id].add_(1)
        self.history[motion_id].lerp_(current, 1.0 / float(count))

    def _difficulty(self):
        logits = self.temperature * self.errors.clamp(max=10.0)
        return logits.masked_fill(~self.valid, -torch.inf).softmax(-1)

    def probabilities(self):
        uniform = self.valid.float() / self.valid.sum(-1, keepdim=True)
        return 0.5 * uniform + 0.4 * self._difficulty() + 0.1 * self.history

    def sample_time(self, motion_ids, truncate_time=None):
        ids = torch.as_tensor(motion_ids, device=self.lengths.device, dtype=torch.long)
        limit = (self.lengths[ids] - float(truncate_time or 0.0)).clamp_min(0.0)
        starts = torch.arange(self.valid.shape[1], device=ids.device) * self.bin_seconds
        available = starts[None] < limit[:, None]
        available[:, 0] = True
        weights = self.probabilities()[ids] * available
        bins = torch.multinomial(weights, 1).squeeze(-1)
        start = bins * self.bin_seconds
        width = (limit - start).clamp(min=0.0, max=self.bin_seconds)
        return start + torch.rand_like(width) * width

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in ("errors", "seen", "history", "updates")}

    def load_state_dict(self, state):
        for name, value in state.items():
            target = getattr(self, name)
            if target.shape != value.shape:
                raise ValueError("Phase curriculum motion layout changed")
            target.copy_(value)
