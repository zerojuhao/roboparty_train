# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Map parkour sub-terrains to coarse family ids for AMP disc terrain gate."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Keep in sync with AMPDiscriminatorCfg.num_terrain_families default.
PERLIN_ROUGH_FAMILY_ID = 0
DISCRETE_FAMILY_ID = 1
STAIRS_DOWN_FAMILY_ID = 2
STAIRS_UP_FAMILY_ID = 3
SLOPE_FAMILY_ID = 4
STAIRS_FAMILY_IDS = (STAIRS_DOWN_FAMILY_ID, STAIRS_UP_FAMILY_ID)
# Foothold train / predict / reward cover discrete footholds and stairs.
FOOTHOLD_GUIDANCE_FAMILY_IDS = (DISCRETE_FAMILY_ID, *STAIRS_FAMILY_IDS)
NUM_TERRAIN_FAMILIES = 5


def asymmetric_pitch_roll_penalty(
    projected_gravity: torch.Tensor,
    forward_tolerance: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Penalize tilt while allowing a forward-lean deadzone.

    Positive projected gravity on body x is forward lean. Forward tilt is
    penalized only beyond ``forward_tolerance`` (radians); backward tilt keeps
    the existing four-times penalty, and roll is unchanged.
    """
    tolerance = torch.as_tensor(
        forward_tolerance,
        dtype=projected_gravity.dtype,
        device=projected_gravity.device,
    )
    if torch.any((tolerance < 0.0) | (tolerance >= torch.pi / 2)):
        raise ValueError("forward_tolerance must be in [0, pi/2).")
    forward = torch.clamp(projected_gravity[:, 0], min=0.0)
    backward = torch.clamp(-projected_gravity[:, 0], min=0.0)
    forward_excess = torch.clamp(forward - torch.sin(tolerance), min=0.0)
    roll = projected_gravity[:, 1]
    return (
        torch.square(forward_excess)
        + torch.square(roll)
        + 16.0 * torch.square(backward)
    )


def terrain_foot_point_weights(
    local_x: torch.Tensor,
    family_ids: torch.Tensor,
    stairs_weight_min: float = 0.0,
    stairs_weight_max: float = 1.0,
) -> torch.Tensor:
    """Terrain-dependent sole weights along foot-local x (heel → toe).

    Up-stairs: toe-heavy; down-stairs: heel-heavy; discrete: mid-foot-heavy.
    Other terrain families use uniform weights.
    ``local_x`` is ``(P,)``; ``family_ids`` is ``(N,)``; returns ``(N, P)``.
    """
    x_min = local_x.min()
    x_max = local_x.max()
    x_frac = ((local_x - x_min) / (x_max - x_min + 1e-8)).clamp(0.0, 1.0)
    weight_span = stairs_weight_max - stairs_weight_min
    w_toe_heavy = stairs_weight_min + weight_span * x_frac
    w_heel_heavy = stairs_weight_max - weight_span * x_frac
    # Peak at mid-foot; heel and toe sit at stairs_weight_min.
    w_mid_heavy = stairs_weight_min + weight_span * (
        1.0 - 2.0 * torch.abs(x_frac - 0.5)
    )
    env_w = torch.ones(
        family_ids.shape[0], local_x.shape[0], device=local_x.device, dtype=local_x.dtype
    )
    mask_discrete = family_ids == DISCRETE_FAMILY_ID
    mask_up = family_ids == STAIRS_UP_FAMILY_ID
    mask_down = family_ids == STAIRS_DOWN_FAMILY_ID
    if mask_discrete.any():
        env_w[mask_discrete] = w_mid_heavy
    if mask_up.any():
        env_w[mask_up] = w_toe_heavy
    if mask_down.any():
        env_w[mask_down] = w_heel_heavy
    return env_w


def soft_absolute_clearance_unsupported(
    foot_z: torch.Tensor,
    terrain_z: torch.Tensor,
    height_offset: float = 0.03,
    height_tolerance: float = 0.03,
    transition_width: float = 0.005,
    point_weights: torch.Tensor | None = None,
    miss_unsupported: float = 1.0,
    unsupported_only: bool = False,
) -> torch.Tensor:
    """Weighted unsupported fraction from absolute foot-to-terrain clearance.

    ``clearance = foot_z - terrain_z - height_offset`` (sole bottom above terrain).
    Soft support uses ``sigmoid((height_tolerance - clearance) / width)``.
    Invalid rays contribute ``miss_unsupported``. Result is roughly in ``[0, 1]``.

    Default averages every sole point (supported points dilute the rate).
    ``unsupported_only`` zeros supported points in the numerator
    (``u_i <= 0.5``, i.e. clearance inside the tolerance band) but still
    divides by the full weight sum, so more hanging area yields a larger value
    and ``point_weights`` only amplify hanging points.
    """
    clearance = foot_z - terrain_z - height_offset
    width = max(float(transition_width), 1.0e-6)
    support = torch.sigmoid((height_tolerance - clearance) / width)
    valid = torch.isfinite(terrain_z)
    unsupported = torch.where(
        valid, 1.0 - support, torch.full_like(support, miss_unsupported)
    )
    if unsupported_only:
        boundary_eps = 8.0 * torch.finfo(clearance.dtype).eps
        unsupported_mask = (~valid) | (
            clearance > float(height_tolerance) + boundary_eps
        )
        unsupported = torch.where(
            unsupported_mask, unsupported, torch.zeros_like(unsupported)
        )
    if point_weights is None:
        point_weights = torch.ones(
            unsupported.shape[-1], device=unsupported.device, dtype=unsupported.dtype
        )
    while point_weights.ndim < unsupported.ndim:
        point_weights = point_weights.unsqueeze(-2)
    denom = point_weights.sum(dim=-1).clamp_min(torch.finfo(unsupported.dtype).eps)
    return (unsupported * point_weights).sum(dim=-1) / denom


def conservative_support_deficiency(
    foot_z: torch.Tensor,
    terrain_z: torch.Tensor,
    height_offset: float = 0.03,
    height_tolerance: float = 0.03,
    transition_width: float = 0.005,
    point_weights: torch.Tensor | None = None,
    miss_unsupported: float = 1.0,
    unsupported_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return conservative, area, and terrain-weighted support deficiencies.

    Terrain weights may increase the penalty for support in an unfavorable sole
    region, but may never hide unsupported sole area.
    """
    kwargs = {
        "height_offset": height_offset,
        "height_tolerance": height_tolerance,
        "transition_width": transition_width,
        "miss_unsupported": miss_unsupported,
        "unsupported_only": unsupported_only,
    }
    area = soft_absolute_clearance_unsupported(
        foot_z, terrain_z, point_weights=None, **kwargs
    )
    if point_weights is None:
        return area, area, area
    terrain = soft_absolute_clearance_unsupported(
        foot_z, terrain_z, point_weights=point_weights, **kwargs
    )
    return torch.maximum(area, terrain), area, terrain


def highest_plane_support_deficiency(
    terrain_z: torch.Tensor,
    height_tolerance: float = 0.03,
    transition_width: float = 0.005,
    point_weights: torch.Tensor | None = None,
    miss_unsupported: float = 1.0,
    unsupported_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure sole support relative to the highest finite terrain sample."""
    valid = torch.isfinite(terrain_z)
    has_plane = valid.any(dim=-1)
    support_plane = torch.where(
        valid, terrain_z, torch.full_like(terrain_z, -torch.inf)
    ).max(dim=-1, keepdim=True).values
    support_plane = torch.where(has_plane.unsqueeze(-1), support_plane, 0.0)
    deficiencies = conservative_support_deficiency(
        support_plane,
        terrain_z,
        height_offset=0.0,
        height_tolerance=height_tolerance,
        transition_width=transition_width,
        point_weights=point_weights,
        miss_unsupported=miss_unsupported,
        unsupported_only=unsupported_only,
    )
    return tuple(
        torch.where(has_plane, value, torch.ones_like(value)) for value in deficiencies
    )


def _family_id_from_name(name: str) -> int:
    lower = name.lower()
    # Match inverted stairs before plain "stairs" (names like pyramid_stairs_inv).
    if "perlin_rough" in lower:
        return PERLIN_ROUGH_FAMILY_ID
    if "stairs_gap" in lower:
        return STAIRS_UP_FAMILY_ID
    if "platforms" in lower:
        return DISCRETE_FAMILY_ID
    if "gap" in lower:
        return DISCRETE_FAMILY_ID
    if "stairs" in lower and "inv" in lower:
        return STAIRS_UP_FAMILY_ID
    if "stairs" in lower:
        return STAIRS_DOWN_FAMILY_ID
    if "slope" in lower:
        return SLOPE_FAMILY_ID
    # Unknown sub-terrains share the perlin_rough bucket.
    return PERLIN_ROUGH_FAMILY_ID


def get_terrain_family_ids(env: ManagerBasedEnv) -> torch.Tensor:
    """Return per-env terrain family ids, shape ``[num_envs]`` (long).

    Falls back to zeros (perlin_rough) when terrain is not a generator with
    subterrain indices (e.g. plane AMP tasks).
    """
    device = env.device
    num_envs = env.num_envs
    zeros = torch.zeros(num_envs, dtype=torch.long, device=device)

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return zeros
    if getattr(terrain.cfg, "terrain_type", None) != "generator":
        return zeros

    gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    terrain_gen = getattr(terrain, "terrain_generator", None)
    if gen_cfg is None or terrain_gen is None or not hasattr(terrain_gen, "get_subterrain_indices"):
        return zeros

    sub_names = list(gen_cfg.sub_terrains.keys())
    sub_to_family = torch.tensor(
        [_family_id_from_name(n) for n in sub_names], dtype=torch.long, device=device
    )
    sub_idx = terrain_gen.get_subterrain_indices(
        terrain.terrain_levels, terrain.terrain_types, device=device
    )
    return sub_to_family[sub_idx]


def terrain_type_indicator(env: ManagerBasedEnv, terrain_type: str) -> torch.Tensor:
    """Return a float indicator for one exact generated sub-terrain."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        raise ValueError("Exact terrain routing requires generated terrain.")

    sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    if terrain_type not in sub_terrain_names:
        raise ValueError(
            f"Terrain type {terrain_type!r} is not configured. "
            f"Available types: {sub_terrain_names}."
        )
    sub_terrain_indices = terrain.terrain_generator.get_subterrain_indices(
        terrain.terrain_levels, terrain.terrain_types, device=env.device
    )
    target_index = sub_terrain_names.index(terrain_type)
    return (sub_terrain_indices == target_index).to(torch.float32).unsqueeze(-1)


def terrain_type_route(
    env: ManagerBasedEnv, terrain_routes: dict[str, int]
) -> torch.Tensor:
    """Return one integer-valued AMP route for each generated sub-terrain."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        raise ValueError("Exact terrain routing requires generated terrain.")

    sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    unknown = set(terrain_routes) - set(sub_terrain_names)
    if unknown:
        raise ValueError(
            f"Terrain routes contain unconfigured types: {sorted(unknown)}. "
            f"Available types: {sub_terrain_names}."
        )
    for terrain_type, route_value in terrain_routes.items():
        if not isinstance(route_value, int) or route_value <= 0:
            raise ValueError(
                f"AMP route for {terrain_type!r} must be a positive integer."
            )

    sub_terrain_indices = terrain.terrain_generator.get_subterrain_indices(
        terrain.terrain_levels, terrain.terrain_types, device=env.device
    )
    routes = torch.zeros_like(sub_terrain_indices, dtype=torch.float32)
    for terrain_type, route_value in terrain_routes.items():
        target_index = sub_terrain_names.index(terrain_type)
        routes[sub_terrain_indices == target_index] = float(route_value)
    return routes.unsqueeze(-1)
