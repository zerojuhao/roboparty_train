"""Attention map encoder for the direct AttnEnc locomotion task."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, MLP


class AttentionEncoder(nn.Module):
    def __init__(self, proprio_dim: int, embedding_dim: int, heads: int, map_size: tuple[int, int], resolution: float):
        super().__init__()
        length, width = map_size
        x = torch.linspace(0, (length - 1) * resolution, length)
        y = torch.linspace(0, (width - 1) * resolution, width)
        xx, yy = torch.meshgrid(x - x.mean(), y - y.mean(), indexing="xy")
        self.register_buffer("position", torch.stack((xx, yy), dim=0))
        self.length = length
        self.width = width
        self.embedding_dim = embedding_dim
        self.cnn = CNN(
            input_dim=(width, length), input_channels=1,
            output_channels=[16, 32, embedding_dim - 3], kernel_size=3,
            stride=1, norm="none", padding="replicate", activation="elu", flatten=False,
        )
        self.proprio_linear = MLP(proprio_dim, embedding_dim, [128], "elu")
        self.attention = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.key_norm = nn.LayerNorm(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, proprio: torch.Tensor, scan: torch.Tensor) -> torch.Tensor:
        heights = scan.reshape(-1, 1, self.width, self.length)
        image_features = self.cnn(heights)
        position = self.position.unsqueeze(0).expand(len(heights), -1, -1, -1)
        points = torch.cat((position, heights, image_features), dim=1)
        points = points.permute(0, 2, 3, 1).reshape(len(heights), -1, self.embedding_dim)
        query = self.query_norm(self.proprio_linear(proprio).unsqueeze(1))
        output, _ = self.attention(query, self.key_norm(points), self.key_norm(points))
        return self.output_norm(output).squeeze(1)


class AttentionModel(MLPModel):
    """Native actor or critic preserving the demo's attention input layout."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        embedding_dim: int = 32,
        head_num: int = 4,
        map_size: tuple[int, int] = (17, 11),
        map_resolution: float = 0.1,
        actor_history_length: int = 5,
        critic_history_length: int = 5,
        enable_critic_estimation: bool = True,
        estimation_slice: list[int] = (78, 79, 80),
        estimation_hidden_dims: list[int] = (256, 64),
        enable_obs_encoder: bool = True,
        obs_encoder_hidden_dims: list[int] = (256, 128),
        latent_dim: int = 32,
    ):
        self.obs_set_name = obs_set
        self.embedding_dim = embedding_dim
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.enable_critic_estimation = enable_critic_estimation
        self.enable_obs_encoder = enable_obs_encoder
        self.latent_dim = latent_dim if enable_obs_encoder else 0
        self.estimation_slice = list(estimation_slice)
        self.perception_group = "perception_a" if obs_set == "actor" else "perception_c"
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.obs_encoder = (
            MLP(self.obs_dim, latent_dim, obs_encoder_hidden_dims, activation)
            if enable_obs_encoder else None
        )
        if obs_set == "critic" and enable_obs_encoder:
            self.actor_obs_encoder = MLP(
                self.actor_dim, latent_dim, obs_encoder_hidden_dims, activation
            )
        else:
            self.actor_obs_encoder = None
        if obs_set == "actor" and enable_critic_estimation:
            estimator_input = latent_dim if enable_obs_encoder else self.obs_dim
            estimator_hidden = [2 * latent_dim] if enable_obs_encoder else estimation_hidden_dims
            self.estimator = MLP(estimator_input, len(estimation_slice), estimator_hidden, activation)
        else:
            self.estimator = None
        attention_input = self.actor_step_dim + self.latent_dim
        if enable_critic_estimation:
            attention_input += len(estimation_slice)
        self.encoder = AttentionEncoder(attention_input, embedding_dim, head_num, map_size, map_resolution)
        self._estimation_loss = None

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str):
        groups, dim = super()._get_obs_dim(obs, obs_groups, obs_set)
        self.actor_dim = obs[obs_groups["actor"][0]].shape[-1]
        self.actor_step_dim = self.actor_dim // self.actor_history_length
        self.critic_dim = obs[obs_groups["critic"][0]].shape[-1]
        self.critic_step_dim = self.critic_dim // self.critic_history_length
        return groups, dim

    def _get_latent_dim(self) -> int:
        if self.obs_set_name == "actor":
            if self.enable_critic_estimation or self.enable_obs_encoder:
                return self.actor_step_dim + self.latent_dim + (
                    len(self.estimation_slice) if self.enable_critic_estimation else 0
                ) + self.embedding_dim
            return self.obs_dim + self.embedding_dim
        if self.enable_obs_encoder:
            return self.critic_step_dim + self.latent_dim + self.embedding_dim
        return self.obs_dim + self.embedding_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        raw = super().get_latent(obs, masks, hidden_state)
        if self.obs_set_name == "actor":
            last = raw[:, -self.actor_step_dim:]
            latent = self.obs_encoder(raw) if self.obs_encoder is not None else None
            parts = [last]
            if latent is not None:
                parts.append(latent)
            if self.estimator is not None:
                prediction = self.estimator(latent if latent is not None else raw)
                critic_last = obs["critic"].reshape(-1, self.critic_history_length, self.critic_step_dim)[:, -1]
                target = critic_last[:, self.estimation_slice].detach()
                self._estimation_loss = (prediction - target).square().mean()
                parts.append(prediction.detach())
            enc_input = torch.cat(parts, dim=-1)
            embedding = self.encoder(enc_input, obs[self.perception_group])
            if self.enable_critic_estimation or self.enable_obs_encoder:
                return torch.cat((enc_input, embedding), dim=-1)
            return torch.cat((raw, embedding), dim=-1)

        last = raw.reshape(-1, self.critic_history_length, self.critic_step_dim)[:, -1]
        parts = [last[:, :self.actor_step_dim]]
        critic_latent = self.obs_encoder(raw) if self.obs_encoder is not None else None
        if self.actor_obs_encoder is not None:
            actor_history = raw.reshape(-1, self.critic_history_length, self.critic_step_dim)
            actor_history = actor_history[:, :, :self.actor_step_dim].reshape(-1, self.actor_dim)
            parts.append(self.actor_obs_encoder(actor_history))
        if self.enable_critic_estimation:
            parts.append(last[:, self.estimation_slice])
        enc_input = torch.cat(parts, dim=-1)
        embedding = self.encoder(enc_input, obs[self.perception_group])
        if critic_latent is not None:
            return torch.cat((last, critic_latent, embedding), dim=-1)
        return torch.cat((raw, embedding), dim=-1)

    def get_aux_loss(self) -> torch.Tensor | None:
        return self._estimation_loss
