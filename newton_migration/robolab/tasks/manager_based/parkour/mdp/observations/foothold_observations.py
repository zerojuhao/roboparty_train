# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Foothold predictor observation helpers."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def resolve_foothold_predictor_obs(obs: TensorDict) -> TensorDict:
    """Use critic terms as the privileged teacher input."""
    group = obs["critic"] if "critic" in obs else obs
    if isinstance(group, TensorDict):
        return group
    return obs


def build_foothold_predictor_state(privileged_obs: TensorDict) -> torch.Tensor:
    """Flatten critic terms into the foothold-teacher input.

    Teacher input is this vector plus the current clipped action.
    """
    parts = list(privileged_obs.values())
    if not parts:
        raise ValueError("No foothold predictor input terms found in critic observations.")
    return torch.cat(parts, dim=-1)
