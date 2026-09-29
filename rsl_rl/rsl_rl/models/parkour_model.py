"""Parkour depth encoder and mixture-of-experts models for RSL-RL 5."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.ssr_estimation import SSREstimationModule, resolve_estimation_dims
from rsl_rl.utils import resolve_nn_activation


class DepthEncoder(nn.Module):
    """The convolution and MLP head used by the demo Parkour policies."""

    def __init__(self, image_shape: tuple[int, int, int], cfg: dict):
        super().__init__()
        channels = cfg["channels"]
        activation = getattr(nn, cfg["nonlinearity"])
        layers = []
        input_channels = image_shape[0]
        for output_channels, kernel, stride, padding in zip(
            channels, cfg["kernel_sizes"], cfg["strides"], cfg["paddings"]
        ):
            conv_stride = 1 if cfg["use_maxpool"] else stride
            layers.extend((nn.Conv2d(input_channels, output_channels, kernel, conv_stride, padding), activation()))
            if cfg["use_maxpool"] and stride > 1:
                layers.append(nn.MaxPool2d(stride))
            input_channels = output_channels
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, *image_shape)).numel()
        head_layers = []
        input_dim = flat_dim
        for hidden_dim in cfg["hidden_sizes"]:
            head_layers.extend((nn.Linear(input_dim, hidden_dim), activation()))
            input_dim = hidden_dim
        head_layers.append(nn.Linear(input_dim, cfg["output_size"]))
        if cfg.get("last_activation"):
            head_layers.append(getattr(nn, cfg["last_activation"])())
        self.head = nn.Sequential(*head_layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(image).flatten(1))


class MoeLayer(nn.Module):
    """Demo-style soft mixture of expert MLP outputs."""

    def __init__(self, input_dim: int, output_dim: int, experts: int, hidden_dims: list[int], gate_dims: list[int], activation: str):
        super().__init__()

        def make_net(sizes: list[int]) -> nn.Sequential:
            layers = []
            for width_in, width_out in zip(sizes[:-2], sizes[1:-1]):
                layers.extend((nn.Linear(width_in, width_out), resolve_nn_activation(activation)))
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
            return nn.Sequential(*layers)

        self.gate = make_net([input_dim, *gate_dims, experts])
        self.experts = nn.ModuleList(
            make_net([input_dim, *hidden_dims, output_dim]) for _ in range(experts)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(values), dim=-1)
        outputs = torch.stack([expert(values) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", weights, outputs)


class ParkourModel(MLPModel):
    """Native actor or critic over Parkour's structured observation group."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict | None = None,
        num_moe_experts: int = 4,
        moe_gate_hidden_dims: list[int] = (),
        enable_velocity_estimation: bool = False,
        velocity_estimation_history_length: int = 8,
        velocity_estimation_estimator_hidden_dims: list[int] = (512, 256),
        aux_velocity_coef: float = 1.0,
        enable_ssr_estimation: bool = False,
        proprio_history_length: int = 8,
        estimation_estimator_hidden_dims: list[int] = (512, 256),
        estimation_decoder_hidden_dims: list[int] = (256, 128),
        enable_actor_foothold: bool = True,
        enable_actor_foot_height: bool = True,
        estimation_foot_height_actor_dim: int = 16,
        gate_foothold_estimation_on_predictor_ready: bool = True,
        aux_foot_height_coef: float = 1.0,
        aux_foothold_coef: float = 2.0,
    ):
        self.encoder_cfg = encoder_cfg
        self.enable_velocity_estimation = enable_velocity_estimation
        self.history_length = velocity_estimation_history_length
        self.aux_velocity_coef = aux_velocity_coef
        self.enable_ssr_estimation = enable_ssr_estimation
        self.enable_actor_foothold = enable_actor_foothold
        self.enable_actor_foot_height = enable_actor_foot_height
        self.estimation_foot_height_actor_dim = estimation_foot_height_actor_dim
        self.aux_foot_height_coef = aux_foot_height_coef
        self.aux_foothold_coef = aux_foothold_coef
        self.gate_foothold_estimation_on_predictor_ready = gate_foothold_estimation_on_predictor_ready
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        if self.image_shape is not None:
            self.depth_encoder = DepthEncoder(self.image_shape, encoder_cfg)
        else:
            self.depth_encoder = None
        if enable_ssr_estimation:
            dims = resolve_estimation_dims(
                obs[obs_groups["actor"][0]], obs[obs_groups["critic"][0]],
                [self.image_name], self.image_dim, proprio_history_length,
                foot_height_actor_dim=estimation_foot_height_actor_dim,
            )
            self.ssr_estimation = SSREstimationModule(
                dims, estimation_estimator_hidden_dims, estimation_decoder_hidden_dims,
                activation, enable_actor_foothold, enable_actor_foot_height,
            )
        else:
            self.ssr_estimation = None
        if enable_velocity_estimation and not enable_ssr_estimation:
            self.velocity_estimator = MLP(
                self.obs_dim + self.image_dim, 3, velocity_estimation_estimator_hidden_dims, activation
            )
        else:
            self.velocity_estimator = None
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MoeLayer(
            self._get_latent_dim(), mlp_output_dim, num_moe_experts,
            list(hidden_dims), list(moe_gate_hidden_dims), activation,
        )
        self._velocity_loss = None

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str):
        group_names = obs_groups[obs_set]
        if len(group_names) != 1:
            raise ValueError("ParkourModel expects one structured observation group")
        group = obs[group_names[0]]
        self.flat_names = [name for name, value in group.items() if value.ndim == 2]
        images = [name for name, value in group.items() if value.ndim == 4]
        if len(images) > 1:
            raise ValueError("ParkourModel supports one image observation")
        if images and self.encoder_cfg is None:
            raise ValueError("Depth encoder configuration is required for image observations")
        self.image_name = images[0] if images else None
        self.image_shape = tuple(group[self.image_name].shape[1:]) if images else None
        self.image_dim = self.encoder_cfg["output_size"] if images else 0
        flat_dim = sum(group[name].shape[-1] for name in self.flat_names)
        return group_names, flat_dim

    def _get_latent_dim(self) -> int:
        if self.enable_ssr_estimation:
            return (
                self.obs_dim + self.image_dim + 3
                + (6 if self.enable_actor_foothold else 0)
                + (self.estimation_foot_height_actor_dim if self.enable_actor_foot_height else 0)
            )
        return self.obs_dim + self.image_dim + (3 if self.enable_velocity_estimation else 0)

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        group = obs[self.obs_groups[0]]
        proprio = torch.cat([group[name] for name in self.flat_names], dim=-1)
        proprio = self.obs_normalizer(proprio)
        if self.depth_encoder is None:
            return proprio
        depth = self.depth_encoder(group[self.image_name])
        if self.ssr_estimation is not None:
            split_terms = list(torch.split(proprio, [group[name].shape[-1] for name in self.flat_names], dim=-1))
            features, _, _ = self.ssr_estimation(
                split_terms, depth, critic_group=obs.get("critic"), store_aux_targets=True
            )
            self.ssr_estimation.set_foothold_teacher_target(obs.get("foothold_teacher"))
            return features
        if self.velocity_estimator is None:
            return torch.cat((proprio, depth), dim=-1)
        velocity = self.velocity_estimator(torch.cat((proprio, depth), dim=-1))
        critic = obs.get("critic")
        target = critic.get("base_lin_vel") if critic is not None else None
        if target is not None:
            if target.shape[-1] != 3:
                target = target.reshape(target.shape[0], self.history_length, 3)[:, -1]
            self._velocity_loss = self.aux_velocity_coef * (velocity - target.detach()).square().mean()
        else:
            self._velocity_loss = None
        return torch.cat((proprio, velocity.detach(), depth), dim=-1)

    def get_aux_loss(self) -> torch.Tensor | None:
        if self.ssr_estimation is not None:
            return self.ssr_estimation.aux_loss(
                self.aux_velocity_coef, self.aux_foot_height_coef, self.aux_foothold_coef
            )
        return self._velocity_loss

    def set_foothold_teacher_ready(self, ready: bool) -> None:
        if self.ssr_estimation is not None:
            self.ssr_estimation.foothold_teacher_ready = ready

    def set_foothold_actor_mask(self, mask: torch.Tensor | None) -> None:
        if self.ssr_estimation is not None:
            self.ssr_estimation.set_foothold_actor_mask(mask)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            group = obs[self.obs_groups[0]]
            proprio = torch.cat([group[name] for name in self.flat_names], dim=-1)
            self.obs_normalizer.update(proprio)
