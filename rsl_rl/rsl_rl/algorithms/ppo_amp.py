"""Adversarial motion priors on the native RSL-RL PPO training path."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from collections import deque
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models.amp_model import resolve_velocity_target_indices
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
        return loss, penalty, policy_score, demo_score


class PPOAMP(PPO):
    """PPO with independent discriminator routes and style rewards."""

    def __init__(self, *args, amp_cfg: dict, obs_groups: dict, resume_learning_rate: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if resume_learning_rate is not None and (not math.isfinite(resume_learning_rate) or resume_learning_rate <= 0):
            raise ValueError("resume_learning_rate must be finite and positive")
        self.resume_learning_rate = resume_learning_rate
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
        disc_num_updates = amp_cfg.get("disc_num_updates")
        if disc_num_updates is not None and disc_num_updates < 1:
            raise ValueError("disc_num_updates must be at least one when configured")
        self._rollout_distributions = {}

    @staticmethod
    def _group_obs(obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[name] for name in groups], dim=-1)

    def _new_distribution_stats(self) -> dict[str, torch.Tensor]:
        return {
            "sum": torch.zeros((), device=self.device),
            "sum_sq": torch.zeros((), device=self.device),
            "count": torch.zeros((), device=self.device),
            "min": torch.full((), torch.inf, device=self.device),
            "max": torch.full((), -torch.inf, device=self.device),
        }

    @staticmethod
    def _accumulate_distribution(stats: dict[str, torch.Tensor], values: torch.Tensor) -> None:
        values = values.detach().float()
        stats["sum"] += values.sum()
        stats["sum_sq"] += values.square().sum()
        stats["count"] += values.numel()
        stats["min"] = torch.minimum(stats["min"], values.min())
        stats["max"] = torch.maximum(stats["max"], values.max())

    def _record_rollout_distribution(self, name: str, values: torch.Tensor) -> None:
        stats = self._rollout_distributions.get(name)
        if stats is None:
            stats = self._new_distribution_stats()
            self._rollout_distributions[name] = stats
        self._accumulate_distribution(stats, values)

    def _reduce_distribution(self, stats: dict[str, torch.Tensor]) -> None:
        if not self.is_multi_gpu:
            return
        totals = torch.stack((stats["sum"], stats["sum_sq"], stats["count"]))
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        stats["sum"], stats["sum_sq"], stats["count"] = totals.unbind()
        torch.distributed.all_reduce(stats["min"], op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(stats["max"], op=torch.distributed.ReduceOp.MAX)

    @staticmethod
    def _distribution_mean(stats: dict[str, torch.Tensor]) -> float:
        count = stats["count"].clamp_min(1.0)
        return (stats["sum"] / count).item()

    def _consume_rollout_metrics(self) -> dict[str, float]:
        metrics = {}
        for name, stats in self._rollout_distributions.items():
            self._reduce_distribution(stats)
            prefix = "AMP/rollout/style_reward" if name == "combined" else f"AMP/{name}/rollout_style_reward"
            metrics[f"{prefix}_mean"] = self._distribution_mean(stats)
        self._rollout_distributions.clear()
        return metrics

    def _collect_style_reward(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        route_group = self.amp_cfg.get("route_obs_group")
        if len(self.routes) == 1 and route_group is None:
            name, route = next(iter(self.routes.items()))
            policy_obs = self._group_obs(obs, self.amp_obs_groups[route["policy_obs_set"]])
            demo_obs = self._group_obs(obs, self.amp_obs_groups[route["demonstration_obs_set"]])
            reward, score = self.discriminators[name].style_reward(
                policy_obs, self.amp_cfg["step_dt"]
            )
            self._record_rollout_distribution(name, reward)
            self.disc_buffers[name][0].append(policy_obs.detach())
            self.disc_buffers[name][1].append(demo_obs.detach())
            weight = float(route.get("style_reward_weight", 1.0))
            if weight == 0.0:
                raise RuntimeError("AMP routes do not cover every environment")
            if weight == 1.0:
                style, scores = reward, score
            else:
                weights = torch.full_like(reward, weight)
                style = weight * reward / weights
                scores = weight * score / weights
            if self.amp_cfg.get("turn_style_reward_multiplier", 1.0) != 1.0:
                style = self._scale_turn_style_reward(style, obs)
            self._record_rollout_distribution("combined", style)
            return style, scores

        route_values = None
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
            record_distribution = getattr(self, "_record_rollout_distribution", None)
            if record_distribution is not None:
                record_distribution(name, reward)
            weight = float(route.get("style_reward_weight", 1.0))
            style[mask] += weight * reward
            scores[mask] += weight * score
            weights[mask] += weight
            self.disc_buffers[name][0].append(policy_obs.detach())
            self.disc_buffers[name][1].append(demo_obs.detach())
        if (weights == 0).any():
            raise RuntimeError("AMP routes do not cover every environment")
        style = style / weights
        if self.amp_cfg.get("turn_style_reward_multiplier", 1.0) != 1.0:
            style = self._scale_turn_style_reward(style, obs)
        if record_distribution is not None:
            record_distribution("combined", style)
        return style, scores / weights

    def _scale_turn_style_reward(self, style: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        """Scale pure turns from the latest command, independently of discriminator inputs."""
        group = self.amp_cfg["turn_style_command_group"]
        start, stop = self.amp_cfg["turn_style_command_slice"]
        # Slice the command term before reshaping: flat histories are term-major.
        commands = obs[group][..., start:stop].reshape(obs[group].shape[0], -1, 3)[:, -1]
        turning = (commands[:, 2].abs() > 0.1) & (commands[:, :2].norm(dim=-1) < 0.1)
        multiplier = self.amp_cfg["turn_style_reward_multiplier"]
        return style * torch.where(turning, multiplier, 1.0)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        self.style_rewards, self.disc_score = self._collect_style_reward(obs)
        blend = self.discriminators["general"].task_style_lerp
        self.rewards_lerp = blend * rewards + (1.0 - blend) * self.style_rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses.update(self._consume_rollout_metrics())
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
            training_totals = torch.zeros(7, device=self.device)
            policy_logits = self._new_distribution_stats()
            demo_logits = self._new_distribution_stats()
            num_updates = self.amp_cfg.get("disc_num_updates")
            if num_updates is None:
                num_updates = self.num_learning_epochs * self.num_mini_batches
            for _ in range(num_updates):
                policy_batch = policy_data[torch.randint(len(policy_data), (batch_size,), device=self.device)]
                demo_batch = demo_data[torch.randint(len(demo_data), (batch_size,), device=self.device)]
                loss, penalty, policy_score, demo_score = disc.loss(
                    policy_batch, demo_batch, self.amp_cfg["grad_penalty_scale"]
                )
                optimizer.zero_grad()
                (loss + penalty).backward()
                if self.is_multi_gpu:
                    reduce_gradients_in_buckets(disc.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb)
                nn.utils.clip_grad_norm_(disc.parameters(), self.amp_cfg["disc_max_grad_norm"])
                optimizer.step()
                disc.normalizer.update(policy_batch.reshape(-1, disc.obs_dim))
                policy_score = policy_score.detach()
                demo_score = demo_score.detach()
                training_totals[0] += loss.detach()
                training_totals[1] += penalty.detach()
                training_totals[2] += 1
                training_totals[3] += (policy_score < 0.0).sum()
                training_totals[4] += policy_score.numel()
                training_totals[5] += (demo_score > 0.0).sum()
                training_totals[6] += demo_score.numel()
                self._accumulate_distribution(policy_logits, policy_score)
                self._accumulate_distribution(demo_logits, demo_score)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(training_totals, op=torch.distributed.ReduceOp.SUM)
            self._reduce_distribution(policy_logits)
            self._reduce_distribution(demo_logits)
            update_count = training_totals[2].clamp_min(1.0)
            policy_accuracy = training_totals[3] / training_totals[4].clamp_min(1.0)
            demo_accuracy = training_totals[5] / training_totals[6].clamp_min(1.0)
            policy_logit_mean = self._distribution_mean(policy_logits)
            demo_logit_mean = self._distribution_mean(demo_logits)
            route_metrics = {
                "accuracy": (0.5 * (policy_accuracy + demo_accuracy)).item(),
                "logit_gap": demo_logit_mean - policy_logit_mean,
            }
            losses.update({f"AMP/{name}/{key}": value for key, value in route_metrics.items()})
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
        if self.resume_learning_rate is not None:
            self.learning_rate = self.resume_learning_rate
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
            print(f"[INFO] AMP resumed learning rate: {self.learning_rate}, schedule: {self.schedule}")
        if load_cfg is None or load_cfg.get("discriminators", True):
            self.discriminators.load_state_dict(loaded_dict["discriminators"], strict=strict)
        if load_cfg is None or load_cfg.get("disc_optimizers", True):
            for name, optimizer in self.disc_optimizers.items():
                optimizer.load_state_dict(loaded_dict["disc_optimizers"][name])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        if (
            cfg["actor"].get("class_name") == "AMPModel"
            and cfg["actor"].get("enable_velocity_estimation", False)
        ):
            manager = env.unwrapped.observation_manager
            cfg["actor"]["velocity_target_indices"] = resolve_velocity_target_indices(
                manager.active_terms["critic"], manager.group_obs_term_dim["critic"]
            )
        amp_cfg = cfg["algorithm"]["amp_cfg"]
        groups = cfg["obs_groups"]
        routes = amp_cfg["discriminators"]
        if amp_cfg.get("turn_style_reward_multiplier", 1.0) != 1.0:
            group = amp_cfg["turn_style_command_group"]
            manager = env.unwrapped.observation_manager
            names = manager.active_terms[group]
            dims = manager.group_obs_term_dim[group]
            index = names.index("velocity_commands")
            command_width = dims[index][-1]
            if obs[group].ndim not in (2, 3) or command_width % 3 != 0:
                raise ValueError("Turn style scaling requires a vector command term with (vx, vy, yaw) frames")
            start = sum(dim[-1] for dim in dims[:index])
            amp_cfg["turn_style_command_slice"] = [start, start + command_width]
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
