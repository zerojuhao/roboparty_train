# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour image encoders and mixture-of-experts heads built on the standard CNN model.

Optional SSR estimation (velocity / foothold / foot-height) folds into EncoderMoEModel
behind enable_ssr_estimation — the overnight foothold teacher is env-side; this flag
adds the actor-side estimation heads that training aux-loss still requires.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.cnn_model import CNNModel, _OnnxCNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.moe import MoeLayer
from rsl_rl.modules.ssr_estimation import (
    SSREstimationModule,
    VelocityEstimationModule,
    resolve_estimation_dims,
    resolve_velocity_estimation_dims,
)
from rsl_rl.utils import resolve_nn_activation


class Conv2dHeadModel(nn.Module):
    """The existing Parkour encoder architecture, with a 5.x model interface."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        hidden_sizes: list[int],
        output_size: int,
        paddings: list[int] | None = None,
        nonlinearity: str = "ReLU",
        use_maxpool: bool = False,
        last_activation: str | None = None,
        input_dtype: str | None = None,
        input_range: tuple[float, float] = (0.0, 255.0),
    ) -> None:
        super().__init__()
        self.input_dtype = input_dtype
        self.input_range = tuple(input_range)
        if input_dtype not in (None, "uint8", "float32"):
            raise ValueError("Encoder input_dtype must be uint8 or float32.")
        if len(input_range) != 2 or input_range[0] >= input_range[1]:
            raise ValueError("Encoder input_range must be increasing.")
        if input_dtype == "uint8" and self.input_range != (0.0, 255.0):
            raise ValueError("uint8 depth inputs must declare input_range=(0, 255).")
        paddings = [0] * len(channels) if paddings is None else paddings
        if not len(channels) == len(kernel_sizes) == len(strides) == len(paddings):
            raise ValueError("Encoder channels, kernels, strides and paddings must have the same length.")
        layers = []
        in_channels = [image_shape[0], *channels[:-1]]
        for in_ch, out_ch, kernel, stride, padding in zip(in_channels, channels, kernel_sizes, strides, paddings):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel, 1 if use_maxpool else stride, padding),
                    resolve_nn_activation(nonlinearity),
                ]
            )
            if use_maxpool and stride > 1:
                layers.append(nn.MaxPool2d(stride))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            conv_dim = self.conv(torch.zeros(1, *image_shape)).numel()
        self.head = MLP(conv_dim, output_size, hidden_sizes, nonlinearity, last_activation=last_activation)
        self.out_features = output_size
        self.output_dim = output_size
        self.output_channels = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dtype == "uint8" and x.dtype != torch.uint8:
            raise TypeError("Depth encoder expects uint8 input in [0, 255].")
        if self.input_dtype == "float32":
            if x.dtype != torch.float32:
                raise TypeError("Depth encoder expects float32 input.")
            x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        return self.head(self.conv(x).flatten(start_dim=1))


class EncoderModel(CNNModel):
    """Use the original Parkour encoder with the standard observation and model interfaces."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_obs_groups: tuple[str, ...] | list[str] = ("depth_image",),
        encoder_cfg: dict | None = None,
    ) -> None:
        if encoder_cfg is None or "output_size" not in encoder_cfg:
            raise ValueError("EncoderModel requires encoder_cfg.output_size.")
        cnns = {
            group: Conv2dHeadModel(tuple(obs[group].shape[1:]), **encoder_cfg)
            for group in obs_groups[obs_set]
            if group in encoder_obs_groups
        }
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            cnns=cnns,
        )


def _flatten_obs_group(value: torch.Tensor | TensorDict) -> torch.Tensor:
    """Flatten a possibly unconcatenated ObsGroup TensorDict into a 2D tensor."""
    if isinstance(value, TensorDict):
        parts = []
        for k, v in value.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 2:
                    parts.append(v)
                elif v.ndim > 2:
                    parts.append(v.reshape(v.shape[0], -1))
        if not parts:
            raise ValueError("Nested observation group has no tensor terms.")
        return torch.cat(parts, dim=-1)
    return value


def _obs_for_parent(obs: TensorDict, group_names: list[str]) -> TensorDict:
    """Clone obs with nested 1D groups flattened so CNNModel/EncoderMoE can size itself."""
    out = obs.copy()
    for name in group_names:
        if name not in out:
            continue
        value = out[name]
        if isinstance(value, TensorDict):
            sample = next(iter(value.values()))
            if isinstance(sample, torch.Tensor) and sample.ndim == 2:
                out[name] = _flatten_obs_group(value)
    return out


class EncoderMoEModel(EncoderModel):
    """Parkour MoE encoder; optional SSR estimation for foothold-conditioned training."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_moe_experts: int = 8,
        moe_gate_hidden_dims: list[int] | None = None,
        enable_ssr_estimation: bool = False,
        enable_velocity_estimation: bool = False,
        proprio_history_length: int = 8,
        estimation_estimator_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        estimation_decoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        enable_actor_foothold: bool = True,
        enable_actor_foot_height: bool = True,
        estimation_foot_height_actor_dim: int = 16,
        aux_velocity_coef: float = 1.0,
        aux_foot_height_coef: float = 1.0,
        aux_foothold_coef: float = 2.0,
        **kwargs,
    ) -> None:
        if enable_ssr_estimation and enable_velocity_estimation:
            raise ValueError("SSR already includes velocity estimation; enable only one estimation mode.")
        self.enable_ssr_estimation = enable_ssr_estimation
        self.enable_velocity_estimation = enable_velocity_estimation
        self.proprio_history_length = proprio_history_length
        self.enable_actor_foothold = enable_actor_foothold
        self.enable_actor_foot_height = enable_actor_foot_height
        self.estimation_foot_height_actor_dim = estimation_foot_height_actor_dim
        self.aux_velocity_coef = aux_velocity_coef
        self.aux_foot_height_coef = aux_foot_height_coef
        self.aux_foothold_coef = aux_foothold_coef
        self.ssr_estimation = None
        self.velocity_estimator = None
        self._ssr_actor_feature_dim: int | None = None
        self._ssr_dims = None

        if enable_ssr_estimation or enable_velocity_estimation:
            encoder_cfg = kwargs.get("encoder_cfg") or {}
            depth_latent_dim = int(encoder_cfg.get("output_size", 128))
            actor_groups = obs_groups.get("actor", obs_groups[obs_set])
            policy_name = next(g for g in actor_groups if "depth" not in g)
            critic_groups = obs_groups.get("critic", ["critic"])
            critic_name = next(g for g in critic_groups if "depth" not in g)
            policy_group = obs[policy_name]
            critic_group = obs[critic_name]
            if not isinstance(policy_group, TensorDict):
                # shortcut: named policy terms for SSR, upgrade when ObsGroup unconcatenated
                policy_group = TensorDict({"proprio": policy_group}, batch_size=policy_group.shape[:1])
            if not isinstance(critic_group, TensorDict):
                raise ValueError(
                    "SSR requires critic concatenate_terms=False so base_lin_vel / foot maps are named."
                )
            if enable_ssr_estimation:
                dims = resolve_estimation_dims(
                    policy_group,
                    critic_group,
                    encoder_obs_names=[],
                    depth_latent_dim=depth_latent_dim,
                    history_length=proprio_history_length,
                    foot_height_actor_dim=estimation_foot_height_actor_dim,
                )
            else:
                dims = resolve_velocity_estimation_dims(
                    policy_group, [], depth_latent_dim, proprio_history_length
                )
            self._ssr_dims = dims
            feat = dims.proprio_flat_dim + dims.velocity_dim + dims.depth_latent_dim
            if enable_ssr_estimation and enable_actor_foothold:
                feat += dims.foothold_dim
            if enable_ssr_estimation and enable_actor_foot_height:
                feat += dims.foot_height_actor_dim
            self._ssr_actor_feature_dim = feat

        # Parent cannot size nested TensorDict groups; flatten 1D groups.
        obs_parent = _obs_for_parent(obs, obs_groups[obs_set])
        super().__init__(
            obs_parent,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            **kwargs,
        )
        if self.distribution is not None and self.distribution.input_dim != self.distribution.output_dim:
            raise NotImplementedError("EncoderMoEModel requires a state-independent output distribution.")

        if enable_ssr_estimation:
            self.ssr_estimation = SSREstimationModule(
                self._ssr_dims,
                estimation_estimator_hidden_dims,
                estimation_decoder_hidden_dims,
                activation,
                enable_actor_foothold,
                enable_actor_foot_height,
            )
        elif enable_velocity_estimation:
            self.velocity_estimator = VelocityEstimationModule(
                self._ssr_dims, estimation_estimator_hidden_dims, activation
            )

        # MoE head: estimation path uses fused actor feature dim instead of CNN+proprio latent.
        moe_in = self._ssr_actor_feature_dim if self._ssr_actor_feature_dim is not None else self._get_latent_dim()
        output_dim = self.mlp[-1].out_features
        self.mlp = MoeLayer(
            moe_in,
            num_moe_experts,
            output_dim,
            activation=activation,
            expert_hidden_dims=list(hidden_dims),
            gate_hidden_dims=list(moe_gate_hidden_dims or []),
        )

    def _get_latent_dim(self) -> int:
        if self._ssr_actor_feature_dim is not None:
            return self._ssr_actor_feature_dim
        return super()._get_latent_dim()

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        flat_obs = _obs_for_parent(obs, list(self.obs_groups) + list(self.obs_groups_2d))
        estimator = self.ssr_estimation if self.ssr_estimation is not None else self.velocity_estimator
        if estimator is None:
            return EncoderModel.get_latent(self, flat_obs, masks=masks, hidden_state=hidden_state)

        if "critic" in obs and isinstance(obs["critic"], TensorDict):
            flat_obs["critic"] = obs["critic"]
        if "foothold_teacher" in obs:
            flat_obs["foothold_teacher"] = obs["foothold_teacher"]

        latent_cnn = torch.cat([self.cnns[g](flat_obs[g]) for g in self.obs_groups_2d], dim=-1)
        if self.obs_groups:
            proprio = torch.cat([flat_obs[g] for g in self.obs_groups], dim=-1)
            proprio = self.obs_normalizer(proprio)
        else:
            proprio = latent_cnn.new_zeros((latent_cnn.shape[0], 0))
        critic_group = obs.get("critic")
        if isinstance(critic_group, TensorDict):
            pass
        elif critic_group is not None:
            critic_group = None
        features, _, _ = estimator(
            [proprio],
            latent_cnn,
            critic_group=critic_group if isinstance(critic_group, TensorDict) else None,
            store_aux_targets=True,
        )
        if self.ssr_estimation is not None:
            self.ssr_estimation.set_foothold_teacher_target(obs.get("foothold_teacher"))
        return features

    def update_normalization(self, obs: TensorDict) -> None:
        flat_obs = _obs_for_parent(obs, list(self.obs_groups) + list(getattr(self, "obs_groups_2d", [])))
        return EncoderModel.update_normalization(self, flat_obs)

    def get_aux_loss(self) -> torch.Tensor | None:
        if self.velocity_estimator is not None:
            return self.velocity_estimator.aux_loss_and_metrics(self.aux_velocity_coef)[0]
        if self.ssr_estimation is None:
            return None
        return self.ssr_estimation.aux_loss(
            self.aux_velocity_coef, self.aux_foot_height_coef, self.aux_foothold_coef
        )

    def get_aux_metrics(self) -> dict[str, torch.Tensor]:
        if self.velocity_estimator is not None:
            return self.velocity_estimator.aux_loss_and_metrics(self.aux_velocity_coef)[1]
        if self.ssr_estimation is None:
            return {}
        _, metrics = self.ssr_estimation.aux_loss_and_metrics(
            self.aux_velocity_coef, self.aux_foot_height_coef, self.aux_foothold_coef
        )
        return metrics

    def set_foothold_teacher_ready(self, ready: bool) -> None:
        if self.ssr_estimation is not None:
            self.ssr_estimation.foothold_teacher_ready = ready

    def set_foothold_actor_mask(self, mask: torch.Tensor | None) -> None:
        if self.ssr_estimation is not None:
            self.ssr_estimation.set_foothold_actor_mask(mask)

    def export_mlp(self) -> nn.Module:
        if self.ssr_estimation is None and self.velocity_estimator is None:
            return copy.deepcopy(self.mlp)
        return _EstimationMoE(self)

    def as_jit(self) -> nn.Module:
        exported = super().as_jit()
        if self.ssr_estimation is not None or self.velocity_estimator is not None:
            exported.mlp = self.export_mlp()
        return exported

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        if self.ssr_estimation is None and self.velocity_estimator is None:
            return super().as_onnx(verbose)
        exported = _OnnxEstimationModel(self, verbose)
        exported.mlp = self.export_mlp()
        return exported


class _EstimationMoE(nn.Module):
    """Deployable estimates from normalized proprioception plus encoded depth."""

    def __init__(self, model: EncoderMoEModel):
        super().__init__()
        self.proprio_dim = model.obs_dim
        self.mlp = copy.deepcopy(model.mlp)
        ssr = model.ssr_estimation
        velocity = ssr.velocity_estimator if ssr is not None else model.velocity_estimator
        self.has_velocity = velocity is not None
        self.velocity = nn.Sequential(*copy.deepcopy(list(velocity))) if velocity is not None else nn.Identity()
        self.has_foothold = ssr is not None and ssr.foothold_estimator is not None
        self.foothold_ready = ssr is not None and ssr.foothold_teacher_ready
        self.foothold = copy.deepcopy(ssr.foothold_estimator) if self.has_foothold else nn.Identity()
        self.has_foot_height = ssr is not None and ssr.foot_height_encoder is not None
        self.foot_height = copy.deepcopy(ssr.foot_height_encoder) if self.has_foot_height else nn.Identity()

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        parts = [fused[..., : self.proprio_dim]]
        if self.has_velocity:
            parts.append(self.velocity(fused))
        if self.has_foothold:
            foothold = self.foothold(fused).reshape(-1, 2, 3)
            yaw = torch.atan2(torch.sin(foothold[..., 2:]), torch.cos(foothold[..., 2:]))
            foothold = torch.cat((foothold[..., :2], yaw), dim=-1).flatten(1)
            parts.append(foothold if self.foothold_ready else torch.zeros_like(foothold))
        if self.has_foot_height:
            parts.append(self.foot_height(fused))
        parts.append(fused[..., self.proprio_dim :])
        return self.mlp(torch.cat(parts, dim=-1))


class _OnnxEstimationModel(_OnnxCNNModel):
    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        inputs = super().get_dummy_inputs()
        return (inputs[0], *(value.to(torch.uint8) for value in inputs[1:]))
