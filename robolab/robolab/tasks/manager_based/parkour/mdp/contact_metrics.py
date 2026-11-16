# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-force metrics shared by parkour foothold labeling."""

from __future__ import annotations

import torch


def vertical_contact_mask(
    forces_w: torch.Tensor,
    vertical_force_ratio: float,
    vertical_force_min: float,
) -> torch.Tensor:
    """Whether contact is strong enough and dominated by vertical force."""
    force_xy = torch.linalg.norm(forces_w[..., :2], dim=-1)
    force_z = torch.abs(forces_w[..., 2])
    return (force_z >= vertical_force_min) & (
        force_z > vertical_force_ratio * force_xy
    )
