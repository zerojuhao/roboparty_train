"""Parkour depth encoder and mixture-of-experts models for RSL-RL 5."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.ssr_estimation import (
    SSREstimationModule,
    VelocityEstimationModule,
    resolve_estimation_dims,
    resolve_velocity_estimation_dims,
)
from rsl_rl.utils import resolve_nn_activation


class DepthEncoder(nn.Module):
    """The convolution and MLP head used by the demo Parkour policies."""

    def __init__(self, image_shape: tuple[int, int, int], cfg: dict):
        super().__init__()
        self.input_dtype = str(cfg.get("input_dtype", "uint8")).lower()
        self.input_range = tuple(float(value) for value in cfg.get("input_range", (0.0, 255.0)))
        if self.input_dtype not in {"uint8", "float32"}:
            raise ValueError("Depth encoder input_dtype must be 'uint8' or 'float32'")
        if len(self.input_range) != 2 or self.input_range[0] >= self.input_range[1]:
            raise ValueError(f"Depth encoder input_range must be increasing, got {self.input_range!r}")
        if self.input_dtype == "uint8" and self.input_range != (0.0, 255.0):
            raise ValueError("uint8 depth inputs must declare input_range=(0, 255)")
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
        if self.input_dtype == "uint8":
            if image.dtype != torch.uint8:
                raise TypeError(
                    "Depth encoder expects torch.uint8 input in [0, 255]; "
                    f"got {image.dtype}. Quantize normalized depth before calling the actor."
                )
            image = image.float().div(255.0)
        else:
            if image.dtype != torch.float32:
                raise TypeError(f"Depth encoder expects torch.float32 input, got {image.dtype}")
            low, high = self.input_range
            image = (image - low) / (high - low)
        return self.head(self.conv(image).flatten(1))


class NormalizedDepthEncoderOnnxExporter(nn.Module):
    """Export wrapper with the training uint8 quantization embedded in the graph."""

    def __init__(self, encoder: DepthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, depth_normalized: torch.Tensor) -> torch.Tensor:
        depth_uint8 = depth_normalized.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        return self.encoder(depth_uint8)


class ParkourActorOnnxExporter(nn.Module):
    """Deployable actor consuming raw proprioception and a precomputed depth latent."""

    def __init__(self, model: ParkourModel):
        super().__init__()
        self.obs_normalizer = model.obs_normalizer
        self.ssr_estimation = model.ssr_estimation
        self.velocity_estimator = model.velocity_estimator
        self.mlp = model.mlp
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.proprio_dims = list(model.flat_dims)
        self.proprio_dim = sum(self.proprio_dims)

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        proprio = self.obs_normalizer(actor_input[..., : self.proprio_dim])
        depth = actor_input[..., self.proprio_dim :]
        if self.ssr_estimation is not None:
            split_terms = list(torch.split(proprio, self.proprio_dims, dim=-1))
            features, _, _ = self.ssr_estimation(
                split_terms, depth, critic_group=None, store_aux_targets=False
            )
        elif self.velocity_estimator is not None:
            split_terms = list(torch.split(proprio, self.proprio_dims, dim=-1))
            features, _, _ = self.velocity_estimator(
                split_terms, depth, critic_group=None, store_aux_targets=False
            )
        else:
            features = torch.cat((proprio, depth), dim=-1)
        return self.deterministic_output(self.mlp(features))


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
        encoder_onnx_filename: str = "depth_encoder.onnx",
        actor_onnx_filename: str = "actor.onnx",
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
        self.encoder_onnx_filename = encoder_onnx_filename
        self.actor_onnx_filename = actor_onnx_filename
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
            dims = resolve_velocity_estimation_dims(
                obs[obs_groups["actor"][0]],
                [self.image_name],
                self.image_dim,
                velocity_estimation_history_length,
            )
            self.velocity_estimator = VelocityEstimationModule(
                dims, velocity_estimation_estimator_hidden_dims, activation
            )
        else:
            self.velocity_estimator = None
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MoeLayer(
            self._get_latent_dim(), mlp_output_dim, num_moe_experts,
            list(hidden_dims), list(moe_gate_hidden_dims), activation,
        )

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str):
        group_names = obs_groups[obs_set]
        if len(group_names) != 1:
            raise ValueError("ParkourModel expects one structured observation group")
        group = obs[group_names[0]]
        self.flat_names = [name for name, value in group.items() if value.ndim == 2]
        self.flat_dims = [group[name].shape[-1] for name in self.flat_names]
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
            split_terms = list(torch.split(proprio, self.flat_dims, dim=-1))
            features, _, _ = self.ssr_estimation(
                split_terms, depth, critic_group=obs.get("critic"), store_aux_targets=True
            )
            self.ssr_estimation.set_foothold_teacher_target(obs.get("foothold_teacher"))
            return features
        if self.velocity_estimator is None:
            return torch.cat((proprio, depth), dim=-1)
        split_terms = list(torch.split(proprio, self.flat_dims, dim=-1))
        features, _, _ = self.velocity_estimator(
            split_terms, depth, critic_group=obs.get("critic"), store_aux_targets=True
        )
        return features

    def get_aux_loss(self) -> torch.Tensor | None:
        if self.ssr_estimation is not None:
            return self.ssr_estimation.aux_loss(
                self.aux_velocity_coef, self.aux_foot_height_coef, self.aux_foothold_coef
            )
        if self.velocity_estimator is not None:
            loss, _ = self.velocity_estimator.aux_loss_and_metrics(self.aux_velocity_coef)
            return loss
        return None

    def get_aux_metrics(self) -> dict[str, torch.Tensor]:
        if self.ssr_estimation is not None:
            _, metrics = self.ssr_estimation.aux_loss_and_metrics(
                self.aux_velocity_coef, self.aux_foot_height_coef, self.aux_foothold_coef
            )
            return metrics
        if self.velocity_estimator is not None:
            _, metrics = self.velocity_estimator.aux_loss_and_metrics(self.aux_velocity_coef)
            return metrics
        return {}

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

    def export_as_onnx(self, obs: TensorDict, filedir: str, opset_version: int = 18) -> None:
        """Export a quantizing depth encoder and the downstream Parkour/SSR actor."""
        if self.depth_encoder is None or self.image_name is None:
            raise RuntimeError("Parkour ONNX export requires a depth observation")
        self.eval()
        os.makedirs(filedir, exist_ok=True)
        group = obs[self.obs_groups[0]]
        depth_uint8 = group[self.image_name]
        if depth_uint8.dtype != torch.uint8:
            raise TypeError(f"ONNX export expects uint8 training depth, got {depth_uint8.dtype}")
        depth_normalized = depth_uint8.float().div(255.0)
        encoder = NormalizedDepthEncoderOnnxExporter(self.depth_encoder).eval()
        with torch.no_grad():
            depth_latent = self.depth_encoder(depth_uint8)
            proprio = torch.cat([group[name] for name in self.flat_names], dim=-1)
            actor_input = torch.cat((proprio, depth_latent), dim=-1)
        actor = ParkourActorOnnxExporter(self).eval()
        encoder_path = os.path.join(filedir, self.encoder_onnx_filename)
        actor_path = os.path.join(filedir, self.actor_onnx_filename)
        torch.onnx.export(
            encoder,
            (depth_normalized,),
            encoder_path,
            opset_version=opset_version,
            input_names=["depth_normalized"],
            output_names=["depth_latent"],
            dynamic_axes={"depth_normalized": {0: "batch"}, "depth_latent": {0: "batch"}},
            external_data=False,
            do_constant_folding=True,
            dynamo=False,
        )
        torch.onnx.export(
            actor,
            (actor_input,),
            actor_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            external_data=False,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"Exported quantizing depth encoder to {encoder_path}")
        print(f"Exported Parkour actor to {actor_path}")
