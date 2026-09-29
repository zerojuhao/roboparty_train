"""Contact and terrain support evaluation for imagined footholds."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import raycast_mesh

from ..terrain_family import (
    highest_plane_support_deficiency,
    terrain_foot_point_weights,
)

# Root pose for labels / support: pos(3) + quat(4) + left/right foot z in body frame (2).
_BASE_FRAME_DIM = 9


@configclass
class FootholdSupportCfg:
    """Sole scan geometry for imagined guidance and contact-support logs.

    The max-plane kernel sits the sole on the highest finite terrain under the
    shoe. Hanging is ``height_tolerance`` below that plane.
    ``feet_at_plane`` keeps its own ankle-relative ``height_offset``.
    """

    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    contact_sensor_name: str = "contact_forces"
    height_scanner_names: list[str] = ["left_height_scanner", "right_height_scanner"]
    height_tolerance: float = 0.03
    support_transition_width: float = 0.005
    touchdown_support_ratio_min: float = 0.6
    """Minimum supported sole fraction accepted as a touchdown label."""
    touchdown_vertical_force_ratio: float = 1.0
    """Require vertical force to exceed this multiple of horizontal force."""
    touchdown_vertical_force_min: float = 5.0
    """Minimum vertical contact force accepted as touchdown, in newtons."""
    ray_start_height: float = 2.0
    ray_max_distance: float = 10.0


def normalize_foothold_support_cfg(
    cfg: FootholdSupportCfg | Mapping,
) -> FootholdSupportCfg:
    if isinstance(cfg, FootholdSupportCfg):
        return cfg
    if isinstance(cfg, Mapping):
        return FootholdSupportCfg(**cfg)
    raise TypeError(f"Unsupported foothold support config type: {type(cfg)!r}")


def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pose_ok(candidate_xy_w: torch.Tensor, foot_yaw: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(candidate_xy_w).all(dim=-1) & torch.isfinite(foot_yaw)


class FootholdSupportEvaluator:
    """Evaluate sole support at imagined and real contacts."""

    def __init__(
        self,
        cfg: FootholdSupportCfg,
        sole_offsets: list[torch.Tensor],
        terrain_mesh,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.sole_offsets = sole_offsets
        self.terrain_mesh = terrain_mesh
        self.device = device

    def _sole_offsets_w(
        self, foot_ids: torch.Tensor, foot_yaws: torch.Tensor
    ) -> torch.Tensor:
        num_pairs = foot_ids.shape[0]
        num_sole_points = self.sole_offsets[0].shape[0]
        offsets = torch.empty(num_pairs, num_sole_points, 2, device=self.device)
        for foot_id in range(2):
            foot_mask = foot_ids == foot_id
            if foot_mask.any():
                offsets[foot_mask] = self.sole_offsets[foot_id]

        cos_foot = torch.cos(foot_yaws).unsqueeze(-1)
        sin_foot = torch.sin(foot_yaws).unsqueeze(-1)
        offset_x_w = cos_foot * offsets[..., 0] - sin_foot * offsets[..., 1]
        offset_y_w = sin_foot * offsets[..., 0] + cos_foot * offsets[..., 1]
        return torch.stack((offset_x_w, offset_y_w), dim=-1)

    def _sole_terrain_point_weights(
        self,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor,
        stairs_weight_min: float,
        stairs_weight_max: float,
    ) -> torch.Tensor:
        num_pairs = foot_ids.shape[0]
        num_points = self.sole_offsets[0].shape[0]
        weights = torch.ones(num_pairs, num_points, device=self.device)
        for foot_id in range(2):
            foot_mask = foot_ids == foot_id
            if not foot_mask.any():
                continue
            weights[foot_mask] = terrain_foot_point_weights(
                self.sole_offsets[foot_id][:, 0],
                family_ids[foot_mask],
                stairs_weight_min,
                stairs_weight_max,
            )
        return weights

    def _geometry_from_xy_b(
        self,
        xy_b: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaw: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raycast sole heights at predicted body-frame XY.

        ``foot_yaw`` is world yaw aligned with ``xy_b``.
        ``base_frames`` is ``[N, 9]``: root pos, root quat, per-foot body z.
        """
        frames = base_frames[env_ids]
        if frames.shape[-1] != _BASE_FRAME_DIM:
            raise ValueError(
                f"Expected base frame last dim {_BASE_FRAME_DIM}, got {frames.shape[-1]}."
            )
        root_pos = frames[:, :3]
        root_quat = frames[:, 3:7]
        foot_z_b = torch.where(foot_ids == 0, frames[:, 7], frames[:, 8])
        delta_w = quat_apply(
            root_quat, torch.stack((xy_b[:, 0], xy_b[:, 1], foot_z_b), dim=-1)
        )
        xy_w = root_pos[:, :2] + delta_w[:, :2]

        offsets_w = self._sole_offsets_w(foot_ids, foot_yaw)
        num_pairs = xy_w.shape[0]
        num_sole_points = offsets_w.shape[1]
        ray_starts = torch.zeros(num_pairs, num_sole_points, 3, device=self.device)
        ray_starts[..., :2] = xy_w[:, None, :] + offsets_w
        ray_starts[..., 2] = frames[:, 2, None] + self.cfg.ray_start_height
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0
        ray_hits, _, _, _ = raycast_mesh(
            ray_starts,
            ray_directions,
            mesh=self.terrain_mesh,
            max_dist=self.cfg.ray_max_distance,
        )
        return xy_w, foot_yaw, ray_hits[..., 2]

    def _contact_terrain_heights(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> torch.Tensor:
        offsets_w = self._sole_offsets_w(foot_ids, foot_yaws)
        ray_starts = torch.zeros(
            foot_pos_w.shape[0], offsets_w.shape[1], 3, device=self.device
        )
        ray_starts[..., :2] = foot_pos_w[:, None, :2] + offsets_w
        ray_starts[..., 2] = foot_pos_w[:, None, 2] + self.cfg.ray_start_height
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0
        ray_hits, _, _, _ = raycast_mesh(
            ray_starts,
            ray_directions,
            mesh=self.terrain_mesh,
            max_dist=self.cfg.ray_max_distance,
        )
        return ray_hits[..., 2]

    def _unsupported_components_from_heights(
        self,
        heights: torch.Tensor,
        point_w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conservative, area, and terrain deficiencies at the highest support plane."""
        deficiencies = highest_plane_support_deficiency(
            heights,
            height_tolerance=self.cfg.height_tolerance,
            transition_width=self.cfg.support_transition_width,
            point_weights=point_w,
            unsupported_only=True,
        )
        return tuple(
            torch.nan_to_num(value, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
            for value in deficiencies
        )

    def _unsupported_from_heights(
        self,
        heights: torch.Tensor,
        point_w: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._unsupported_components_from_heights(heights, point_w)[0]

    def contact_support_components(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.1,
        stairs_weight_max: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return conservative, area, and terrain-weighted support ratios."""
        point_w = None
        if enable_terrain_foot_weights and family_ids is not None:
            point_w = self._sole_terrain_point_weights(
                foot_ids, family_ids, stairs_weight_min, stairs_weight_max
            )
        deficiencies = self._unsupported_components_from_heights(
            self._contact_terrain_heights(foot_pos_w, foot_yaws, foot_ids), point_w
        )
        return tuple(1.0 - _finite_or_zero(value) for value in deficiencies)

    def contact_support_ratio(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.1,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        """Supported sole fraction at a real contact, same kernel as imagined guidance."""
        return self.contact_support_components(
            foot_pos_w,
            foot_yaws,
            foot_ids,
            family_ids,
            enable_terrain_foot_weights,
            stairs_weight_min,
            stairs_weight_max,
        )[0]

    def _support_deficiency_for_xy(
        self,
        xy_b: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaw: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        xy_w, foot_yaw, heights = self._geometry_from_xy_b(
            xy_b, base_frames, foot_yaw, env_ids, foot_ids
        )
        point_w = None
        if enable_terrain_foot_weights and family_ids is not None:
            point_w = self._sole_terrain_point_weights(
                foot_ids,
                family_ids[env_ids],
                stairs_weight_min,
                stairs_weight_max,
            )
        deficiency = self._unsupported_from_heights(heights, point_w)
        deficiency = torch.where(
            _pose_ok(xy_w, foot_yaw), deficiency, torch.ones_like(deficiency)
        )
        return _finite_or_zero(deficiency)

    def expected_support_deficiency(
        self,
        candidate_xy_b: torch.Tensor,
        candidate_weights: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
        eval_chunk_size: int = 64,
        reduction: str = "mean",
        candidate_yaws_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unsupported fraction over XY quadrature samples.

        ``candidate_xy_b`` is ``[P, G, 2]`` and ``candidate_weights`` is ``[P, G]``.
        ``foot_yaws`` is predicted mean world yaw ``[num_envs, 2]``.
        ``candidate_yaws_w`` optionally supplies yaw quadrature ``[P, G]``.
        ``reduction`` is ``max`` (worst sample) or ``mean`` (weighted average).
        """
        num_pairs, num_candidates = candidate_weights.shape
        if num_pairs == 0:
            return torch.zeros(0, device=self.device)
        if candidate_xy_b.shape != (num_pairs, num_candidates, 2):
            raise ValueError(
                "candidate_xy_b must have shape [P, G, 2] matching candidate_weights."
            )
        if candidate_yaws_w is not None and candidate_yaws_w.shape != (
            num_pairs,
            num_candidates,
        ):
            raise ValueError("candidate_yaws_w must have shape [P, G].")
        reduce_max = str(reduction).lower() == "max"
        weights = torch.nan_to_num(
            candidate_weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        expected = torch.zeros(num_pairs, device=self.device)
        chunk_size = max(int(eval_chunk_size), 1)
        for start in range(0, num_pairs, chunk_size):
            end = min(start + chunk_size, num_pairs)
            pair_count = end - start
            chunk_env_ids = env_ids[start:end]
            chunk_foot_ids = foot_ids[start:end]
            flat_xy = candidate_xy_b[start:end].reshape(-1, 2)
            flat_env_ids = chunk_env_ids[:, None].expand(-1, num_candidates).reshape(-1)
            flat_foot_ids = chunk_foot_ids[:, None].expand(-1, num_candidates).reshape(-1)
            if candidate_yaws_w is None:
                chunk_yaws = foot_yaws[chunk_env_ids, chunk_foot_ids]
                flat_yaws = chunk_yaws[:, None].expand(-1, num_candidates).reshape(-1)
            else:
                flat_yaws = candidate_yaws_w[start:end].reshape(-1)
            flat_deficiency = self._support_deficiency_for_xy(
                flat_xy,
                base_frames,
                flat_yaws,
                flat_env_ids,
                flat_foot_ids,
                family_ids=family_ids,
                enable_terrain_foot_weights=enable_terrain_foot_weights,
                stairs_weight_min=stairs_weight_min,
                stairs_weight_max=stairs_weight_max,
            ).view(pair_count, num_candidates)
            if reduce_max:
                expected[start:end] = flat_deficiency.max(dim=-1).values
            else:
                expected[start:end] = (weights[start:end] * flat_deficiency).sum(dim=-1)
        return _finite_or_zero(expected).clamp(0.0, 1.0)
