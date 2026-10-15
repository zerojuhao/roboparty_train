# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SSR estimation: velocity / foothold / foot-height auxiliaries.

Heads read ``concat(flattened proprio history, depth_latent)`` directly.
Actor features are
``[proprio_feat, stopgrad(v_hat), stopgrad(f_hat), stopgrad(h_hat), depth_latent]``
when foothold and foot-height actor channels are enabled. ``f_hat`` is 6D
``[lx, ly, lyaw, rx, ry, ryaw]``. ``h_hat`` is a compressed under-foot height code
trained by reconstructing the privileged height maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.mlp import MLP

FOOTHOLD_TEACHER_DIM = 6


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def mirror_foothold_teacher_xy(xy: torch.Tensor) -> torch.Tensor:
    """Left-right mirror of ``[lx, ly, lyaw, rx, ry, ryaw]``."""
    if xy.shape[-1] != FOOTHOLD_TEACHER_DIM:
        raise ValueError(
            f"Expected foothold teacher last dim {FOOTHOLD_TEACHER_DIM}, got {xy.shape[-1]}."
        )
    reshaped = xy.view(-1, 2, 3)
    swapped = reshaped.clone()
    swapped[:, 0] = reshaped[:, 1]
    swapped[:, 1] = reshaped[:, 0]
    swapped[..., 1] = -swapped[..., 1]
    swapped[..., 2] = -swapped[..., 2]
    return swapped.reshape(xy.shape)


@dataclass
class SSREstimationDims:
    """Resolved observation / latent sizes for estimation construction."""

    history_length: int
    proprio_term_dims: list[int]
    proprio_last_dim: int
    depth_latent_dim: int
    foot_height_dim: int
    velocity_dim: int = 3
    foothold_dim: int = FOOTHOLD_TEACHER_DIM
    foot_height_actor_dim: int = 16

    @property
    def proprio_flat_dim(self) -> int:
        return self.proprio_last_dim * self.history_length

    @property
    def fused_dim(self) -> int:
        return self.proprio_flat_dim + self.depth_latent_dim


@dataclass
class VelocityEstimationDims:
    """Resolved observation / latent sizes for velocity-only estimation."""

    history_length: int
    proprio_term_dims: list[int]
    proprio_last_dim: int
    depth_latent_dim: int
    velocity_dim: int = 3

    @property
    def proprio_flat_dim(self) -> int:
        return self.proprio_last_dim * self.history_length

    @property
    def fused_dim(self) -> int:
        return self.proprio_flat_dim + self.depth_latent_dim

    @property
    def actor_feature_dim(self) -> int:
        return self.proprio_flat_dim + self.velocity_dim + self.depth_latent_dim


class VelocityEstimationModule(MLP):
    """Estimate privileged base velocity from policy proprioception and depth."""

    def __init__(
        self,
        dims: VelocityEstimationDims,
        estimator_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        activation: str = "elu",
    ) -> None:
        super().__init__(dims.fused_dim, dims.velocity_dim, estimator_hidden_dims, activation)
        self.dims = dims
        self._clear_aux_cache()

    def _clear_aux_cache(self) -> None:
        self.last_velocity_pred: torch.Tensor | None = None
        self.last_velocity_target: torch.Tensor | None = None

    def clear_cache(self) -> None:
        self._clear_aux_cache()

    def forward(
        self,
        proprio_terms: list[torch.Tensor],
        depth_latent: torch.Tensor,
        critic_group: TensorDict | None = None,
        store_aux_targets: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_history = flatten_proprio_history(proprio_terms, self.dims.history_length)
        fused = torch.cat((proprio_history, depth_latent), dim=-1)
        velocity_pred = super().forward(fused)
        self.last_velocity_pred = velocity_pred
        if store_aux_targets and critic_group is not None:
            self.last_velocity_target = last_history_frame(
                critic_group["base_lin_vel"],
                self.dims.history_length,
                self.dims.velocity_dim,
            ).detach()
        else:
            self.last_velocity_target = None

        features = torch.cat(
            (proprio_history, velocity_pred.detach(), depth_latent), dim=-1
        )
        return features, velocity_pred, fused

    def aux_loss_and_metrics(
        self, velocity_coef: float
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.last_velocity_pred is None or self.last_velocity_target is None:
            return None, {}
        error = self.last_velocity_pred - self.last_velocity_target
        metrics = {
            "Estimation/velocity_rmse": torch.sqrt(torch.mean(error.detach().square()))
        }
        if velocity_coef == 0.0:
            return None, metrics
        return velocity_coef * torch.mean(error.square()), metrics


class SSREstimationModule(nn.Module):
    """Estimate SSR auxiliary heads from direct proprio/depth fusion."""

    def __init__(
        self,
        dims: SSREstimationDims,
        estimator_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        activation: str = "elu",
        enable_actor_foothold: bool = True,
        enable_actor_foot_height: bool = True,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.enable_actor_foothold = enable_actor_foothold
        self.enable_actor_foot_height = enable_actor_foot_height
        self.foothold_teacher_ready = True
        self._foothold_actor_mask: torch.Tensor | None = None
        fused_dim = dims.fused_dim

        self.velocity_estimator = MLP(
            fused_dim, dims.velocity_dim, estimator_hidden_dims, activation
        )
        self.foothold_estimator: MLP | None = None
        if enable_actor_foothold:
            self.foothold_estimator = MLP(
                fused_dim, dims.foothold_dim, estimator_hidden_dims, activation
            )
        self.foot_height_encoder: MLP | None = None
        if enable_actor_foot_height:
            self.foot_height_encoder = MLP(
                fused_dim, dims.foot_height_actor_dim, [128], activation
            )
            self.foot_height_decoder = MLP(
                dims.foot_height_actor_dim,
                2 * dims.foot_height_dim,
                decoder_hidden_dims,
                activation,
            )
        else:
            self.foot_height_decoder = MLP(
                fused_dim,
                2 * dims.foot_height_dim,
                decoder_hidden_dims,
                activation,
            )
        self._clear_aux_cache()

    def _clear_aux_cache(self) -> None:
        self.last_velocity_pred: torch.Tensor | None = None
        self.last_velocity_target: torch.Tensor | None = None
        self.last_foothold_pred: torch.Tensor | None = None
        self.last_foothold_target: torch.Tensor | None = None
        self.last_foot_height_pred: torch.Tensor | None = None
        self.last_foot_height_target: torch.Tensor | None = None

    def set_foothold_actor_mask(self, mask: torch.Tensor | None) -> None:
        self._foothold_actor_mask = mask

    def clear_cache(self) -> None:
        self._clear_aux_cache()

    @property
    def actor_feature_dim(self) -> int:
        dim = self.dims.proprio_flat_dim + self.dims.velocity_dim + self.dims.depth_latent_dim
        if self.enable_actor_foothold:
            dim += self.dims.foothold_dim
        if self.enable_actor_foot_height:
            dim += self.dims.foot_height_actor_dim
        return dim

    def forward(
        self,
        proprio_terms: list[torch.Tensor],
        depth_latent: torch.Tensor,
        critic_group: TensorDict | None = None,
        store_aux_targets: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_history = flatten_proprio_history(proprio_terms, self.dims.history_length)
        fused = torch.cat((proprio_history, depth_latent), dim=-1)

        velocity_pred = self.velocity_estimator(fused)
        foothold_pred = (
            self.foothold_estimator(fused) if self.foothold_estimator is not None else None
        )
        if self.foot_height_encoder is not None:
            foot_height_feat = self.foot_height_encoder(fused)
            foot_height_pred = self.foot_height_decoder(foot_height_feat)
        else:
            foot_height_feat = None
            foot_height_pred = self.foot_height_decoder(fused)

        foothold_active = foothold_pred is not None and self.foothold_teacher_ready
        self.last_velocity_pred = velocity_pred
        self.last_foothold_pred = foothold_pred if foothold_active else None
        self.last_foot_height_pred = foot_height_pred
        self._store_aux_targets(critic_group if store_aux_targets else None)

        proprio_feat = proprio_history
        parts = [proprio_feat, velocity_pred.detach()]
        if foothold_pred is not None:
            if foothold_active:
                actor_f = foothold_pred.detach().clone().view(-1, 2, 3)
                actor_f[..., 2] = wrap_angle(actor_f[..., 2])
                actor_f = actor_f.reshape(foothold_pred.shape)
                if self._foothold_actor_mask is not None:
                    mask = self._foothold_actor_mask.to(device=actor_f.device)
                    if mask.shape[0] == actor_f.shape[0]:
                        actor_f = actor_f * mask.to(dtype=actor_f.dtype).view(-1, 1)
                parts.append(actor_f)
            else:
                parts.append(torch.zeros_like(foothold_pred))
        if foot_height_feat is not None:
            parts.append(foot_height_feat.detach())
        parts.append(depth_latent)
        return torch.cat(parts, dim=-1), velocity_pred, fused

    def _store_aux_targets(self, critic_group: TensorDict | None) -> None:
        if critic_group is None:
            self.last_velocity_target = None
            self.last_foot_height_target = None
            self.last_foothold_target = None
            return
        self.last_velocity_target = last_history_frame(
            critic_group["base_lin_vel"],
            self.dims.history_length,
            self.dims.velocity_dim,
        ).detach()
        left = last_history_frame(
            critic_group["left_foot_height_map"],
            self.dims.history_length,
            self.dims.foot_height_dim,
        )
        right = last_history_frame(
            critic_group["right_foot_height_map"],
            self.dims.history_length,
            self.dims.foot_height_dim,
        )
        self.last_foot_height_target = torch.cat((left, right), dim=-1).detach()
        self.last_foothold_target = None

    def set_foothold_teacher_target(self, teacher: torch.Tensor | None) -> None:
        if (
            teacher is not None
            and self.last_foothold_pred is not None
            and self.foothold_teacher_ready
        ):
            self.last_foothold_target = teacher.detach()
        else:
            self.last_foothold_target = None

    def aux_loss(
        self,
        velocity_coef: float,
        foot_height_coef: float,
        foothold_coef: float = 0.0,
    ) -> torch.Tensor | None:
        total, _ = self.aux_loss_and_metrics(
            velocity_coef=velocity_coef,
            foot_height_coef=foot_height_coef,
            foothold_coef=foothold_coef,
        )
        return total

    def aux_loss_and_metrics(
        self,
        velocity_coef: float,
        foot_height_coef: float,
        foothold_coef: float = 0.0,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        weighted: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {}

        def _add(
            name: str,
            pred: torch.Tensor | None,
            target: torch.Tensor | None,
            coef: float,
        ) -> None:
            if pred is None or target is None:
                return
            err = pred - target
            metrics[f"Estimation/{name}_rmse"] = torch.sqrt(torch.mean(err.detach().square()))
            if coef != 0.0:
                weighted.append(coef * torch.mean(err.square()))

        def _add_foothold(coef: float) -> None:
            pred = self.last_foothold_pred
            target = self.last_foothold_target
            if pred is None or target is None:
                return
            valid = torch.isfinite(target).all(dim=-1)
            if not valid.any():
                return
            pred = pred[valid]
            target = target[valid]
            if pred.shape[-1] != FOOTHOLD_TEACHER_DIM or target.shape[-1] != FOOTHOLD_TEACHER_DIM:
                raise ValueError(
                    "Foothold aux expects 6D XY+yaw; "
                    f"got pred {tuple(pred.shape)} target {tuple(target.shape)}."
                )
            pred_pose = pred.view(-1, 2, 3)
            tgt_pose = target.view(-1, 2, 3)
            xy_err = pred_pose[..., :2] - tgt_pose[..., :2]
            yaw_err = wrap_angle(pred_pose[..., 2] - tgt_pose[..., 2])
            metrics["Estimation/foothold_xy_rmse"] = torch.sqrt(
                torch.mean(xy_err.detach().square())
            )
            metrics["Estimation/foothold_yaw_rmse"] = torch.sqrt(
                torch.mean(yaw_err.detach().square())
            )
            if coef != 0.0:
                weighted.append(coef * (torch.mean(xy_err.square()) + torch.mean(yaw_err.square())))

        _add("velocity", self.last_velocity_pred, self.last_velocity_target, velocity_coef)
        _add_foothold(foothold_coef if self.foothold_teacher_ready else 0.0)
        _add("foot_height", self.last_foot_height_pred, self.last_foot_height_target, foot_height_coef)

        if not weighted:
            return None, metrics
        return torch.stack(weighted).sum(), metrics


def resolve_estimation_dims(
    policy_group: TensorDict,
    critic_group: TensorDict,
    encoder_obs_names: list[str],
    depth_latent_dim: int,
    history_length: int,
    foot_height_actor_dim: int = 16,
) -> SSREstimationDims:
    proprio_term_dims: list[int] = []
    for name, value in policy_group.items():
        if name in encoder_obs_names:
            continue
        if value.ndim != 2:
            raise ValueError(
                f"SSR estimation expects flat proprio terms; got '{name}' with shape {tuple(value.shape)}."
            )
        if value.shape[-1] % history_length != 0:
            raise ValueError(
                f"Proprio term '{name}' dim {value.shape[-1]} is not divisible by "
                f"history_length={history_length}."
            )
        proprio_term_dims.append(value.shape[-1] // history_length)

    if not proprio_term_dims:
        raise ValueError("SSR estimation requires at least one proprioceptive policy term.")

    left = critic_group["left_foot_height_map"]
    right = critic_group["right_foot_height_map"]
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(
            "Left and right foot height maps must have matching dimensions; "
            f"got {left.shape[-1]} and {right.shape[-1]}."
        )
    # Foot height maps are configured as single-frame critic terms. Do not infer
    # history from divisibility by the proprio history length: a 24x5 scan has
    # 120 points and would otherwise be misread as 8 frames of 15 points.
    foot_height_dim = left.shape[-1]

    return SSREstimationDims(
        history_length=history_length,
        proprio_term_dims=proprio_term_dims,
        proprio_last_dim=sum(proprio_term_dims),
        depth_latent_dim=depth_latent_dim,
        foot_height_dim=foot_height_dim,
        foothold_dim=FOOTHOLD_TEACHER_DIM,
        foot_height_actor_dim=foot_height_actor_dim,
    )


def resolve_velocity_estimation_dims(
    policy_group: TensorDict,
    encoder_obs_names: list[str],
    depth_latent_dim: int,
    history_length: int,
) -> VelocityEstimationDims:
    """Resolve velocity-estimator dimensions from the policy observation group."""
    proprio_term_dims: list[int] = []
    for name, value in policy_group.items():
        if name in encoder_obs_names:
            continue
        if value.ndim != 2:
            raise ValueError(
                f"Velocity estimation expects flat proprio terms; got '{name}' "
                f"with shape {tuple(value.shape)}."
            )
        if value.shape[-1] % history_length != 0:
            raise ValueError(
                f"Proprio term '{name}' dim {value.shape[-1]} is not divisible by "
                f"history_length={history_length}."
            )
        proprio_term_dims.append(value.shape[-1] // history_length)
    if not proprio_term_dims:
        raise ValueError(
            "Velocity estimation requires at least one proprioceptive policy term."
        )
    return VelocityEstimationDims(
        history_length=history_length,
        proprio_term_dims=proprio_term_dims,
        proprio_last_dim=sum(proprio_term_dims),
        depth_latent_dim=depth_latent_dim,
    )


def flatten_proprio_history(
    proprio_terms: list[torch.Tensor], history_length: int
) -> torch.Tensor:
    full_history: list[torch.Tensor] = []
    for term in proprio_terms:
        term_dim = term.shape[-1] // history_length
        hist = term.view(term.shape[0], history_length, term_dim)
        full_history.append(hist.reshape(term.shape[0], -1))
    return torch.cat(full_history, dim=-1)


def last_history_frame(
    flat_history: torch.Tensor, history_length: int, frame_dim: int
) -> torch.Tensor:
    last_dim = flat_history.shape[-1]
    if last_dim == frame_dim:
        return flat_history
    expected = history_length * frame_dim
    if last_dim != expected:
        raise ValueError(
            f"Expected flat history dim {expected} or single-frame dim {frame_dim}, "
            f"got {last_dim}."
        )
    return flat_history.view(flat_history.shape[0], history_length, frame_dim)[:, -1, :]
