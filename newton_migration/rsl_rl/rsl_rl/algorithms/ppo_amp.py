"""Adversarial motion priors on the native RSL-RL PPO training path."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections import deque
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import reduce_gradients_in_buckets, resolve_nn_activation


class AMPDiscriminator(nn.Module):
    def __init__(self, obs_dim: int, obs_steps: int, cfg: dict, loss_type: str):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_steps = obs_steps
        self.loss_type = loss_type
        self.style_reward_scale = cfg["style_reward_scale"]
        self.task_style_lerp = cfg["task_style_lerp"]
        self.normalizer = EmpiricalNormalization(obs_dim, until=1e8)
        activation = resolve_nn_activation(cfg["activation"])
        layers = []
        input_dim = obs_dim * obs_steps
        for hidden_dim in cfg["hidden_dims"]:
            layers.extend((nn.Linear(input_dim, hidden_dim), copy.deepcopy(activation)))
            input_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.linear = nn.Linear(input_dim, 1)

    def normalized(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalizer(obs.reshape(-1, self.obs_dim)).reshape(obs.shape[0], -1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.trunk(self.normalized(obs)))

    @torch.no_grad()
    def style_reward(self, obs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        score = self(obs).squeeze(-1)
        if self.loss_type == "LSGAN":
            reward = (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)
        elif self.loss_type == "GAN":
            reward = torch.nn.functional.softplus(score)
        else:
            reward = score
        return dt * self.style_reward_scale * reward, score

    def loss(self, policy_obs: torch.Tensor, demo_obs: torch.Tensor, penalty_scale: float) -> tuple:
        policy_score = self(policy_obs)
        demo_input = self.normalized(demo_obs).detach().requires_grad_(True)
        demo_score = self.linear(self.trunk(demo_input))
        if self.loss_type == "LSGAN":
            loss = 0.5 * ((policy_score + 1).square().mean() + (demo_score - 1).square().mean())
        elif self.loss_type == "GAN":
            loss = 0.5 * (
                torch.nn.functional.softplus(policy_score).mean()
                + torch.nn.functional.softplus(-demo_score).mean()
            )
        else:
            loss = policy_score.mean() - demo_score.mean()
        gradient = torch.autograd.grad(demo_score.sum(), demo_input, create_graph=True)[0]
        penalty = penalty_scale * gradient.norm(dim=-1).square().mean()
        return loss, penalty, policy_score.mean(), demo_score.mean()


class PPOAMP(PPO):
    """PPO with independent discriminator routes and style rewards."""

    def __init__(self, *args, amp_cfg: dict, obs_groups: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp_cfg = amp_cfg
        self.amp_obs_groups = obs_groups
        self.routes = amp_cfg["discriminators"]
        self.discriminators = nn.ModuleDict()
        self.disc_optimizers = {}
        self.disc_buffers = {}
        for name, route in self.routes.items():
            disc_cfg = {**amp_cfg["amp_discriminator"], **route.get("amp_discriminator", {})}
            disc = AMPDiscriminator(route["disc_obs_dim"], route["disc_obs_steps"], disc_cfg, amp_cfg["loss_type"])
            self.discriminators[name] = disc.to(self.device)
            self.disc_optimizers[name] = torch.optim.Adam(
                [
                    {"params": disc.trunk.parameters(), "weight_decay": amp_cfg["disc_trunk_weight_decay"]},
                    {"params": disc.linear.parameters(), "weight_decay": amp_cfg["disc_linear_weight_decay"]},
                ],
                lr=amp_cfg["disc_learning_rate"],
            )
            capacity = amp_cfg["disc_obs_buffer_size"]
            self.disc_buffers[name] = (deque(maxlen=capacity), deque(maxlen=capacity))
        if "general" not in self.discriminators:
            raise ValueError("AMP requires a general discriminator route")

    @staticmethod
    def _group_obs(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[name] for name in groups], dim=-1)

    def _collect_style_reward(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        route_values = None
        route_group = self.amp_cfg.get("route_obs_group")
        if route_group:
            route_values = obs[route_group].reshape(-1).long()
        style = torch.zeros(self.storage.num_envs, device=self.device)
        weights = torch.zeros_like(style)
        scores = torch.zeros_like(style)
        for name, route in self.routes.items():
            mask = (
                torch.ones_like(style, dtype=torch.bool)
                if route.get("all_envs", False) or route_values is None
                else route_values == route["route_value"]
            )
            if not mask.any():
                continue
            policy_set = route["policy_obs_set"]
            policy_obs = self._group_obs(obs, self.amp_obs_groups[policy_set])[mask]
            demo_obs = self._group_obs(obs, self.amp_obs_groups[route["demonstration_obs_set"]])[mask]
            reward, score = self.discriminators[name].style_reward(policy_obs, self.amp_cfg["step_dt"])
            weight = float(route.get("style_reward_weight", 1.0))
            style[mask] += weight * reward
            scores[mask] += weight * score
            weights[mask] += weight
            self.disc_buffers[name][0].append(policy_obs.detach())
            self.disc_buffers[name][1].append(demo_obs.detach())
        if (weights == 0).any():
            raise RuntimeError("AMP routes do not cover every environment")
        return style / weights, scores / weights

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        self.style_rewards, self.disc_score = self._collect_style_reward(obs)
        blend = self.discriminators["general"].task_style_lerp
        self.rewards_lerp = blend * rewards + (1.0 - blend) * self.style_rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    def update(self) -> dict[str, float]:
        losses = super().update()
        metrics = {"disc": 0.0, "disc_grad_penalty": 0.0, "disc_score": 0.0, "disc_demo_score": 0.0}
        updates = 0
        for name, disc in self.discriminators.items():
            policy_batches, demo_batches = self.disc_buffers[name]
            has_batches = torch.tensor(int(bool(policy_batches and demo_batches)), device=self.device)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(has_batches, op=torch.distributed.ReduceOp.MIN)
            if not has_batches.item():
                continue
            policy_data = torch.cat(tuple(policy_batches), dim=0)
            demo_data = torch.cat(tuple(demo_batches), dim=0)
            optimizer = self.disc_optimizers[name]
            batch_size = max(1, min(len(policy_data), len(demo_data)) // self.num_mini_batches)
            for _ in range(self.num_learning_epochs * self.num_mini_batches):
                policy_batch = policy_data[torch.randint(len(policy_data), (batch_size,), device=self.device)]
                demo_batch = demo_data[torch.randint(len(demo_data), (batch_size,), device=self.device)]
                loss, penalty, score, demo_score = disc.loss(
                    policy_batch, demo_batch, self.amp_cfg["grad_penalty_scale"]
                )
                optimizer.zero_grad()
                (loss + penalty).backward()
                if self.is_multi_gpu:
                    reduce_gradients_in_buckets(disc.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb)
                nn.utils.clip_grad_norm_(disc.parameters(), self.amp_cfg["disc_max_grad_norm"])
                optimizer.step()
                disc.normalizer.update(policy_batch.reshape(-1, disc.obs_dim))
                for key, value in zip(metrics, (loss, penalty, score, demo_score)):
                    metrics[key] += value.item()
                updates += 1
        if updates:
            losses.update({key: value / updates for key, value in metrics.items()})
        return losses

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        for disc in self.discriminators.values():
            for tensor in disc.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def save(self) -> dict:
        saved = super().save()
        saved["discriminators"] = self.discriminators.state_dict()
        saved["disc_optimizers"] = {name: opt.state_dict() for name, opt in self.disc_optimizers.items()}
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminators", True):
            self.discriminators.load_state_dict(loaded_dict["discriminators"], strict=strict)
        if load_cfg is None or load_cfg.get("disc_optimizers", True):
            for name, optimizer in self.disc_optimizers.items():
                optimizer.load_state_dict(loaded_dict["disc_optimizers"][name])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        amp_cfg = cfg["algorithm"]["amp_cfg"]
        groups = cfg["obs_groups"]
        routes = amp_cfg["discriminators"]
        if len(routes) > 1 and not amp_cfg.get("route_obs_group") and not all(
            route.get("all_envs", False) for route in routes.values()
        ):
            raise ValueError("Multiple AMP routes require a route observation group")
        for route in routes.values():
            policy_set = route["policy_obs_set"]
            demo_set = route["demonstration_obs_set"]
            policy_obs = PPOAMP._group_obs(obs, groups[policy_set])
            demo_obs = PPOAMP._group_obs(obs, groups[demo_set])
            if policy_obs.ndim != 3 or policy_obs.shape[1:] != demo_obs.shape[1:]:
                raise ValueError("AMP policy and demonstration histories must have equal 3D shapes")
            route["disc_obs_steps"] = policy_obs.shape[1]
            route["disc_obs_dim"] = policy_obs.shape[2]
        amp_cfg["step_dt"] = env.env.unwrapped.step_dt
        amp_cfg["discriminators"] = routes
        cfg["algorithm"]["obs_groups"] = groups
        return PPO.construct_algorithm(obs, env, cfg, device)
