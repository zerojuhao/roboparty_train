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


def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)


class ResidualBlock(nn.Module):
    def __init__(self, dim, num_parallel=1):
        super().__init__()
        self.mlp = nn.Sequential(layernorm(dim, num_parallel), linear_layer(dim, dim, num_parallel), nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, num_parallel=1):
        super().__init__()
        seq = [layernorm(input_dim, num_parallel), linear_layer(input_dim, output_dim, num_parallel)]
        if activation:
            seq += [nn.Mish()]
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


class BackwardMap(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim=256, hidden_layers=1, norm=True):
        super().__init__()
        seq = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim)]
        if norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ResidualForwardMap(nn.Module):
    """Forward map / critic head. Also used with output_dim=1 for critics."""

    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim=2048, hidden_layers=6, num_parallel=2, output_dim=None):
        super().__init__()
        self.num_parallel = num_parallel
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, hidden_layers, num_parallel)
        self.embed_sa = residual_embedding(obs_dim + action_dim, hidden_dim, hidden_layers, num_parallel)
        seq = [ResidualBlock(hidden_dim, num_parallel) for _ in range(hidden_layers)]
        seq += [Block(hidden_dim, output_dim if output_dim else z_dim, False, num_parallel)]
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
    """Discriminator critic and auxiliary critic with a shared representation.

    Both critics observe the same input keys and train against the same batch,
    so the embedding towers and residual trunk are shared while each critic
    keeps its own output head. Forwards return ``(disc, aux)`` with shape
    ``(num_parallel, batch, 1)`` each.
    """

    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim=1024, hidden_layers=6, num_parallel=2):
        super().__init__()
        self.num_parallel = num_parallel
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, hidden_layers, num_parallel)
        self.embed_sa = residual_embedding(obs_dim + action_dim, hidden_dim, hidden_layers, num_parallel)
        self.trunk = nn.Sequential(*[ResidualBlock(hidden_dim, num_parallel) for _ in range(hidden_layers)])
        self.head_disc = Block(hidden_dim, 1, False, num_parallel)
        self.head_aux = Block(hidden_dim, 1, False, num_parallel)

    def forward(self, obs, z, action):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        features = self.trunk(torch.cat([sa_embedding, z_embedding], dim=-1))
        disc = self.head_disc(features)
        aux = self.head_aux(features)
        if self.num_parallel == 1:
            return disc.unsqueeze(0), aux.unsqueeze(0)
        return disc, aux


class ResidualActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim=2048, hidden_layers=6, embedding_layers=2):
        super().__init__()
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = residual_embedding(obs_dim, hidden_dim, embedding_layers)
        seq = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)] + [Block(hidden_dim, action_dim, False)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


class _FixedAdaptiveAvgPool2x4(nn.Module):
    """Static pooling used by the fixed-shape PUFO depth encoder."""

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError(f"Expected feature map {(self.height, self.width)}, got {tuple(x.shape[-2:])}")
        rows = torch.stack(
            tuple(
                x[..., i * self.height // 2 : ((i + 1) * self.height + 1) // 2, :].mean(dim=-2)
                for i in range(2)
            ),
            dim=-2,
        )
        return torch.stack(
            tuple(
                rows[..., i * self.width // 4 : ((i + 1) * self.width + 3) // 4].mean(dim=-1)
                for i in range(4)
            ),
            dim=-1,
        )


class DepthEncoder(nn.Module):
    """Encode eight temporal depth frames into one deployable latent."""

    def __init__(self, channels=8, height=18, width=32, latent_dim=256):
        super().__init__()
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        feature_height = ((height + 1) // 2 + 1) // 2
        feature_width = ((width + 1) // 2 + 1) // 2
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            _FixedAdaptiveAvgPool2x4(feature_height, feature_width),
            nn.Flatten(),
        )
        self.frame_projection = nn.Sequential(nn.Linear(64 * 2 * 4, 256), nn.ReLU())
        self.gru = nn.GRU(256, latent_dim, batch_first=True)

    def forward(self, depth):
        expected = (self.channels, self.height, self.width)
        if depth.ndim != 4 or tuple(depth.shape[1:]) != expected:
            raise ValueError(
                f"Expected depth [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(depth.shape)}"
            )
        depth = depth.float()
        if depth.max().detach() > 1.0:
            depth = depth / 255.0
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
        hidden_dim=2048,
        hidden_layers=6,
        embedding_layers=2,
        depth_shape=(8, 18, 32),
        estimate_velocity=True,
    ):
        super().__init__()
        self.obs_keys = tuple(obs_keys)
        self.depth_key = "depth_image"
        self.depth_encoder = DepthEncoder(*depth_shape, latent_dim=256)
        state_dim = sum(obs_dims[key] for key in self.obs_keys)
        estimator_state_dim = sum(obs_dims[key] for key in ("state", "last_action", "history_actor"))
        self.estimate_velocity = bool(estimate_velocity)
        if self.estimate_velocity:
            self.velocity_estimator = nn.Sequential(
                nn.Linear(estimator_state_dim + 256, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 3),
            )
            state_dim += 3
        self.embed_z = residual_embedding(state_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = residual_embedding(state_dim, hidden_dim, embedding_layers)
        self.fusion = Block(hidden_dim + 256, hidden_dim, True)
        self.policy = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            Block(hidden_dim, action_dim, False),
        )

    def forward(self, obs, z, std):
        state = torch.cat([obs[key] for key in self.obs_keys], dim=-1)
        depth_latent = self.depth_encoder(obs[self.depth_key])
        velocity_estimate = None
        if self.estimate_velocity:
            estimator_state = torch.cat(
                [obs[key] for key in ("state", "last_action", "history_actor")], dim=-1
            )
            velocity_estimate = self.velocity_estimator(torch.cat([estimator_state, depth_latent], dim=-1))
            state = torch.cat([state, velocity_estimate.detach()], dim=-1)
        z_embedding = self.embed_z(torch.cat([state, z], dim=-1))
        state_embedding = self.embed_s(state)
        fused = self.fusion(torch.cat([state_embedding, z_embedding, depth_latent], dim=-1))
        mean = torch.tanh(self.policy(fused))
        dist = TruncatedNormal(mean, torch.ones_like(mean) * std)
        if velocity_estimate is not None:
            dist.velocity_estimate = velocity_estimate
        return dist



class Discriminator(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim=1024, hidden_layers=3):
        super().__init__()
        seq = [nn.Linear(obs_dim + z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def compute_logits(self, obs, z):
        return self.trunk(torch.cat([z, obs], dim=1))

    def forward(self, obs, z):
        return torch.sigmoid(self.compute_logits(obs, z))

    def compute_reward(self, obs, z, eps=1e-7):
        s = torch.clamp(self.forward(obs, z), eps, 1 - eps)
        return s.log() - (1 - s).log()


class ObsNormalizer(nn.Module):
    """Per-key BatchNorm running statistics, matching UFO's normalizer."""

    def __init__(self, obs_dims: dict[str, int], momentum=0.01):
        super().__init__()
        self.normalizers = nn.ModuleDict(
            {k: nn.BatchNorm1d(d, affine=False, momentum=momentum) for k, d in obs_dims.items()}
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
