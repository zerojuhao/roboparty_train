"""FB-CPR-Aux network modules, ported from UFO `humanoidverse/agents/nn_models.py`.

Plain torch modules (constructor args only, no config system). Input filters
are replaced by explicit `obs_keys` lists over dict observations.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def model_kwargs(cfg):
    """Read a model config before or after runner serialization."""
    return dict(cfg) if isinstance(cfg, dict) else cfg.to_dict()


def mlp(input_dim, output_dim, hidden_dims, activation, first_activation, layer_norm):
    """Build the FB MLP layout with optional normalization of its first hidden layer."""
    if any(width <= 0 for width in hidden_dims):
        raise ValueError(f"MLP hidden_dims must be positive, got {hidden_dims}")
    layers = []
    for index, width in enumerate(hidden_dims):
        layers.append(nn.Linear(input_dim, width))
        if index == 0 and layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(resolve_nn_activation(first_activation if index == 0 else activation))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


def parallel_orthogonal_(tensor, gain=1.0):
    if tensor.ndimension() == 2:
        return nn.init.orthogonal_(tensor, gain=gain)
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new_empty(n_parallel, rows, cols).normal_(0, 1)
    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()
        q, r = torch.linalg.qr(flat_tensor)
        q *= torch.diag(r, 0).sign()
        if rows < cols:
            q.t_()
        qs.append(q)
    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def soft_update_params(net, target_net, tau):
    net_params = tuple(x.data for x in net.parameters())
    target_net_params = tuple(x.data for x in target_net.parameters())
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


def linear_layer(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


class DenseParallel(nn.Module):
    def __init__(self, in_features, out_features, n_parallel, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.baddbmm(self.bias, input, self.weight)


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
            self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        return norm_input


class Norm(nn.Module):
    def forward(self, x):
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def residual_embedding(input_dim, hidden_dim, hidden_layers, activation, layer_norm, num_parallel=1):
    if hidden_dim < 2 or hidden_dim % 2 or hidden_layers < 2:
        raise ValueError("Residual models need an even hidden_dim >= 2 and embedding_layers >= 2")
    seq = [Block(input_dim, hidden_dim, activation, layer_norm, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, activation, layer_norm, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, activation, layer_norm, num_parallel)]
    return nn.Sequential(*seq)


class ResidualBlock(nn.Module):
    def __init__(self, dim, activation, layer_norm, num_parallel=1):
        super().__init__()
        layers = [layernorm(dim, num_parallel)] if layer_norm else []
        layers += [linear_layer(dim, dim, num_parallel), resolve_nn_activation(activation)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, layer_norm, num_parallel=1):
        super().__init__()
        seq = [layernorm(input_dim, num_parallel)] if layer_norm else []
        seq += [linear_layer(input_dim, output_dim, num_parallel)]
        if activation is not None:
            seq += [resolve_nn_activation(activation)]
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


class BackwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim, *, hidden_dims, activation, first_activation, layer_norm, norm):
        super().__init__()
        self.net = mlp(obs_dim, z_dim, hidden_dims, activation, first_activation, layer_norm)
        if norm:
            self.net.append(Norm())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ResidualForwardMap(nn.Module):
    """Forward map / critic head. Also used with output_dim=1 for critics."""

    def __init__(
        self, obs_dim, z_dim, action_dim, *, hidden_dim, hidden_layers, embedding_layers,
        activation, layer_norm, num_parallel, output_dim=None,
    ):
        super().__init__()
        if hidden_layers < 0 or num_parallel < 1:
            raise ValueError("hidden_layers must be non-negative and num_parallel must be positive")
        self.num_parallel = num_parallel
        self.embed_z = residual_embedding(
            obs_dim + z_dim, hidden_dim, embedding_layers, activation, layer_norm, num_parallel,
        )
        self.embed_sa = residual_embedding(
            obs_dim + action_dim, hidden_dim, embedding_layers, activation, layer_norm, num_parallel,
        )
        seq = [ResidualBlock(hidden_dim, activation, layer_norm, num_parallel) for _ in range(hidden_layers)]
        seq += [Block(hidden_dim, output_dim if output_dim else z_dim, None, layer_norm, num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs, z, action):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        out = self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))
        if self.num_parallel == 1:
            return out.unsqueeze(0)
        return out


class SharedCriticMap(nn.Module):
    """Two named scalar critics with a shared representation.

    Both critics observe the same input keys and train against the same batch,
    so the embedding towers and residual trunk are shared while each critic
    keeps its own output head. Forwards follow ``head_names`` order, defaulting
    to ``(disc, aux)``, with shape ``(num_parallel, batch, 1)`` each.
    """

    def __init__(
        self, obs_dim, z_dim, action_dim, *, hidden_dim, hidden_layers, embedding_layers,
        activation, layer_norm, num_parallel, head_names=("disc", "aux"),
    ):
        super().__init__()
        if hidden_layers < 0 or num_parallel < 1:
            raise ValueError("hidden_layers must be non-negative and num_parallel must be positive")
        self.num_parallel = num_parallel
        if len(head_names) != 2 or len(set(head_names)) != 2:
            raise ValueError("Shared critics require two distinct head names")
        self.head_names = tuple(head_names)
        self.embed_z = residual_embedding(
            obs_dim + z_dim, hidden_dim, embedding_layers, activation, layer_norm, num_parallel,
        )
        self.embed_sa = residual_embedding(
            obs_dim + action_dim, hidden_dim, embedding_layers, activation, layer_norm, num_parallel,
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation, layer_norm, num_parallel) for _ in range(hidden_layers)]
        )
        for name in self.head_names:
            self.add_module(f"head_{name}", Block(hidden_dim, 1, None, layer_norm, num_parallel))

    def forward(self, obs, z, action):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        features = self.trunk(torch.cat([sa_embedding, z_embedding], dim=-1))
        outputs = tuple(getattr(self, f"head_{name}")(features) for name in self.head_names)
        if self.num_parallel == 1:
            return tuple(value.unsqueeze(0) for value in outputs)
        return outputs


class ResidualActor(nn.Module):
    def __init__(
        self, obs_dim, z_dim, action_dim, *, hidden_dim, hidden_layers, embedding_layers, activation, layer_norm,
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, activation, layer_norm)
        self.embed_s = residual_embedding(obs_dim, hidden_dim, embedding_layers, activation, layer_norm)
        seq = [ResidualBlock(hidden_dim, activation, layer_norm) for _ in range(hidden_layers)]
        seq += [Block(hidden_dim, action_dim, None, layer_norm)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class _FixedAdaptiveAvgPool2d(nn.Module):
    """Static pooling used by the fixed-shape PUFO depth encoder."""

    def __init__(self, height: int, width: int, output_size: tuple[int, int]):
        super().__init__()
        self.height = height
        self.width = width
        rows, columns = output_size
        if min(height, width, rows, columns) < 1:
            raise ValueError("Pooling input and output sizes must be positive")
        self.row_ranges = [(i * height // rows, ((i + 1) * height + rows - 1) // rows) for i in range(rows)]
        self.column_ranges = [
            (i * width // columns, ((i + 1) * width + columns - 1) // columns) for i in range(columns)
        ]

    def forward(self, x):
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"Expected feature map {(self.height, self.width)}, got {tuple(x.shape[-2:])}")
        rows = torch.stack(
            tuple(
                x[..., start:end, :].mean(dim=-2)
                for start, end in self.row_ranges
            ),
            dim=-2,
        )
        return torch.stack(
            tuple(
                rows[..., start:end].mean(dim=-1)
                for start, end in self.column_ranges
            ),
            dim=-1,
        )


class DepthEncoder(nn.Module):
    """Encode temporal depth frames under an explicit dtype/range contract."""

    def __init__(
        self, input_shape, *, channels, kernel_sizes, strides, paddings, pool_shape,
        hidden_dims, activation, output_size, gru_layers,
        input_dtype="uint8", input_range=(0.0, 255.0),
    ):
        super().__init__()
        self.channels, self.height, self.width = input_shape
        self.output_size = output_size
        self.input_dtype = str(input_dtype).lower()
        if self.input_dtype not in {"uint8", "float32"}:
            raise ValueError("Depth encoder input_dtype must be 'uint8' or 'float32'.")
        if len(input_range) != 2 or not all(math.isfinite(float(x)) for x in input_range):
            raise ValueError(f"Depth encoder input_range must contain two finite values, got {input_range!r}.")
        self.input_range = (float(input_range[0]), float(input_range[1]))
        if not self.input_range[0] < self.input_range[1]:
            raise ValueError(f"Depth encoder input_range must be increasing, got {self.input_range!r}.")
        if self.input_dtype == "uint8" and self.input_range != (0.0, 255.0):
            raise ValueError("uint8 depth inputs must declare input_range=(0, 255).")
        if self.input_dtype == "float32" and self.input_range[0] < 0.0:
            raise ValueError("float32 depth input_range must have a non-negative lower bound.")
        if not channels or not (len(channels) == len(kernel_sizes) == len(strides) == len(paddings)):
            raise ValueError(
                "Depth CNN channels, kernel_sizes, strides and paddings must have matching nonzero lengths"
            )
        if min(*input_shape, *channels, *kernel_sizes, *strides, output_size, gru_layers) < 1 or min(paddings) < 0:
            raise ValueError("Depth encoder dimensions must be positive, with non-negative padding")
        feature_height, feature_width = self.height, self.width
        layers = []
        input_channels = 1
        for width, kernel, stride, padding in zip(channels, kernel_sizes, strides, paddings):
            layers += [nn.Conv2d(input_channels, width, kernel, stride=stride, padding=padding)]
            layers += [resolve_nn_activation(activation)]
            feature_height = (feature_height + 2 * padding - kernel) // stride + 1
            feature_width = (feature_width + 2 * padding - kernel) // stride + 1
            if min(feature_height, feature_width) < 1:
                raise ValueError("Depth CNN configuration collapses the spatial dimensions")
            input_channels = width
        layers += [_FixedAdaptiveAvgPool2d(feature_height, feature_width, pool_shape), nn.Flatten()]
        self.frame_cnn = nn.Sequential(*layers)
        projection = []
        input_dim = channels[-1] * math.prod(pool_shape)
        for width in hidden_dims:
            if width < 1:
                raise ValueError("Depth projection hidden_dims must be positive")
            projection += [nn.Linear(input_dim, width), resolve_nn_activation(activation)]
            input_dim = width
        self.frame_projection = nn.Sequential(*projection)
        self.gru = nn.GRU(input_dim, output_size, num_layers=gru_layers, batch_first=True)

    def forward(self, depth):
        expected = (self.channels, self.height, self.width)
        if depth.ndim != 4 or tuple(depth.shape[1:]) != expected:
            raise ValueError(
                f"Expected depth [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(depth.shape)}"
            )
        if self.input_dtype == "uint8":
            if depth.dtype != torch.uint8:
                raise TypeError(
                    "Depth encoder expects torch.uint8 input in [0, 255]; "
                    f"got {depth.dtype}. Quantize normalized depth before calling the actor."
                )
            depth = depth.float().div(255.0)
        else:
            if depth.dtype != torch.float32:
                raise TypeError(
                    "Depth encoder expects torch.float32 input; "
                    f"got {depth.dtype}. Declare a separate input contract for other dtypes."
                )
            low, high = self.input_range
            if not torch.jit.is_tracing():
                if not torch.isfinite(depth).all():
                    raise ValueError("Depth encoder received non-finite float32 depth values.")
                if bool((depth < low).any() or (depth > high).any()):
                    raise ValueError(
                        f"Depth encoder expected float32 depth in [{low}, {high}], "
                        f"got [{depth.min().item()}, {depth.max().item()}]."
                    )
            depth = (depth - low) / (high - low)
        batch = depth.shape[0]
        features = self.frame_projection(
            self.frame_cnn(depth.reshape(batch * self.channels, 1, self.height, self.width))
        )
        _, hidden = self.gru(features.reshape(batch, self.channels, -1))
        return hidden[-1]


class VisualResidualActor(nn.Module):
    """PUFO actor: proprioception, context and temporal depth."""

    def __init__(
        self,
        obs_dims,
        obs_keys,
        z_dim,
        action_dim,
        *,
        hidden_dim,
        hidden_layers,
        embedding_layers,
        activation,
        layer_norm,
        depth_shape,
        encoder_cfg,
        estimator_cfg,
        estimate_velocity,
        velocity_estimator_updates_encoder=True,
    ):
        super().__init__()
        self.obs_keys = tuple(obs_keys)
        self.depth_key = "depth_image"
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        self.depth_encoder = DepthEncoder(depth_shape, **encoder_cfg)
        depth_dim = self.depth_encoder.output_size
        state_dim = sum(obs_dims[key] for key in self.obs_keys)
        estimator_state_dim = sum(obs_dims[key] for key in ("state", "last_action", "history_actor"))
        self.estimate_velocity = bool(estimate_velocity)
        self.velocity_estimator_updates_encoder = bool(velocity_estimator_updates_encoder)
        if self.estimate_velocity:
            velocity_dim = obs_dims["velocity_target_yaw"]
            self.velocity_estimator = mlp(estimator_state_dim + depth_dim, velocity_dim, **estimator_cfg)
            state_dim += velocity_dim
        self.embed_z = residual_embedding(state_dim + z_dim, hidden_dim, embedding_layers, activation, layer_norm)
        self.embed_s = residual_embedding(state_dim, hidden_dim, embedding_layers, activation, layer_norm)
        self.fusion = Block(hidden_dim + depth_dim, hidden_dim, activation, layer_norm)
        self.policy = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation, layer_norm) for _ in range(hidden_layers)],
            Block(hidden_dim, action_dim, None, layer_norm),
        )

    def _mean_from_depth_latent(self, obs, z, depth_latent, velocity_depth_latent=None):
        state = torch.cat([obs[key] for key in self.obs_keys], dim=-1)
        velocity_estimate = None
        if self.estimate_velocity:
            estimator_state = torch.cat(
                [obs[key] for key in ("state", "last_action", "history_actor")], dim=-1
            )
            if velocity_depth_latent is None:
                velocity_depth_latent = depth_latent
            if not self.velocity_estimator_updates_encoder:
                velocity_depth_latent = velocity_depth_latent.detach()
            velocity_estimate = self.velocity_estimator(
                torch.cat([estimator_state, velocity_depth_latent], dim=-1)
            )
            state = torch.cat([state, velocity_estimate.detach()], dim=-1)
        z_embedding = self.embed_z(torch.cat([state, z], dim=-1))
        state_embedding = self.embed_s(state)
        fused = self.fusion(torch.cat([state_embedding, z_embedding, depth_latent], dim=-1))
        return torch.tanh(self.policy(fused)), velocity_estimate

    def forward_from_depth_latent(self, obs, z, depth_latent, std):
        mean, velocity_estimate = self._mean_from_depth_latent(obs, z, depth_latent)
        dist = TruncatedNormal(mean, torch.ones_like(mean) * std)
        if velocity_estimate is not None:
            dist.velocity_estimate = velocity_estimate
        return dist

    def forward_with_depth_gradient_diagnostics(self, obs, z, std):
        depth_latent = self.depth_encoder(obs[self.depth_key])
        actor_depth_latent = depth_latent.view_as(depth_latent)
        velocity_depth_latent = depth_latent.view_as(depth_latent)
        mean, velocity_estimate = self._mean_from_depth_latent(
            obs, z, actor_depth_latent, velocity_depth_latent
        )
        dist = TruncatedNormal(mean, torch.ones_like(mean) * std)
        if velocity_estimate is not None:
            dist.velocity_estimate = velocity_estimate
            dist.actor_depth_latent = actor_depth_latent
            dist.velocity_depth_latent = velocity_depth_latent
        return dist

    def forward(self, obs, z, std, depth_latent=None):
        if depth_latent is None:
            depth_latent = self.depth_encoder(obs[self.depth_key])
        dist = self.forward_from_depth_latent(obs, z, depth_latent, std)
        dist.depth_latent = depth_latent
        return dist

    def mean_from_depth_latent(self, obs, z, depth_latent):
        return self._mean_from_depth_latent(obs, z, depth_latent)[0]


class DepthEncoderOnnxExporter(nn.Module):
    """Small wrapper that gives the deployable depth graph one tensor input."""

    def __init__(self, encoder: DepthEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, depth):
        return self.encoder(depth)


class VisualActorOnnxExporter(nn.Module):
    """Deployable PUFO actor graph consuming a precomputed depth latent.

    The velocity estimator remains inside the actor graph.  Its prediction is
    detached before the policy trunk, matching training, but it is still part
    of the exported graph and therefore runs on the robot.
    """

    def __init__(self, actor: VisualResidualActor, normalizer: ObsNormalizer):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, state, last_action, history_actor, heading, velocity_command, z, depth_latent):
        obs = {
            "state": state,
            "last_action": last_action,
            "history_actor": history_actor,
            "heading": heading,
            "velocity_command": velocity_command,
        }
        obs = self.normalizer.normalize_dict(obs)
        return self.actor.mean_from_depth_latent(obs, z, depth_latent)


class BackwardContextOnnxExporter(nn.Module):
    """Export the backward map that produces the actor context ``z``."""

    def __init__(self, backward_map, normalizer, input_keys, *, norm_z=False):
        super().__init__()
        self.backward_map = backward_map
        self.normalizer = normalizer
        self.input_keys = tuple(input_keys)
        self.norm_z = bool(norm_z)

    def forward(self, *inputs):
        if len(inputs) != len(self.input_keys):
            raise ValueError(
                f"Expected {len(self.input_keys)} context inputs, got {len(inputs)}."
            )
        obs = self.normalizer.normalize_dict(dict(zip(self.input_keys, inputs)))
        context = self.backward_map(
            torch.cat([obs[key] for key in self.input_keys], dim=-1)
        )
        if self.norm_z:
            context = math.sqrt(context.shape[-1]) * F.normalize(context, dim=-1)
        return context


class Discriminator(nn.Module):
    def __init__(self, obs_dim, z_dim, *, hidden_dims, activation, first_activation, layer_norm):
        super().__init__()
        self.trunk = mlp(obs_dim + z_dim, 1, hidden_dims, activation, first_activation, layer_norm)

    def compute_logits(self, obs, z):
        return self.trunk(torch.cat([z, obs], dim=1))

    def forward(self, obs, z):
        return torch.sigmoid(self.compute_logits(obs, z))

    def compute_reward(self, obs, z, eps=1e-7):
        s = torch.clamp(self.forward(obs, z), eps, 1 - eps)
        return s.log() - (1 - s).log()


class ObsNormalizer(nn.Module):
    """Per-key BatchNorm running statistics, matching UFO's normalizer."""

    def __init__(self, obs_dims: dict[str, int], momentum=0.01, identity_keys=()):
        super().__init__()
        identity_keys = set(identity_keys)
        self.normalizers = nn.ModuleDict(
            {
                key: (
                    nn.Identity()
                    if key in identity_keys
                    else nn.BatchNorm1d(dim, affine=False, momentum=momentum)
                )
                for key, dim in obs_dims.items()
            }
        )

    def update_dict(self, obs: dict) -> None:
        with torch.no_grad():
            previous = self.training
            self.train(True)
            for k, normalizer in self.normalizers.items():
                if k in obs:
                    normalizer(obs[k])
            self.train(previous)

    def normalize_dict(self, obs: dict) -> dict:
        previous = self.training
        self.train(False)
        out = {k: self.normalizers[k](v) for k, v in obs.items() if k in self.normalizers}
        self.train(previous)
        return out


class EMA(nn.Module):
    """Bias-corrected exponential moving-average reward normalizer."""

    def __init__(self, tau=0.99, epsilon=1e-8, translate=False, scale=True):
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.translate = translate
        self.scale = scale
        self.register_buffer("mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("counter", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        with torch.no_grad():
            self.mean.mul_(self.tau).add_(x.mean(), alpha=1 - self.tau)
            self.mean_square.mul_(self.tau).add_(x.pow(2).mean(), alpha=1 - self.tau)
            self.counter.add_(1)
        norm = 1 - self.tau ** self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean.pow(2), min=self.epsilon)
        translate_mean = ema_mean if self.translate else 0
        scale_std = torch.sqrt(var) if self.scale else 1
        return (x - translate_mean) / scale_std
