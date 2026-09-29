"""FB-CPR-Aux off-policy agent, ported from UFO humanoidverse agents
(FBAgent -> FBcprAgent -> FBcprAuxAgent) to a plain rsl_rl-style algorithm.

Observation contract (dict of flat tensors):
    obs["state"]            part of every network input
    obs["privileged_state"] privileged input (b, disc, f, critic)
    obs["last_action"]      actor / f / critic
    obs["history_actor"]    actor / f / critic
"""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import autograd
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .fb_networks import (
    BackwardMap,
    Discriminator,
    EMA,
    ObsNormalizer,
    ResidualActor,
    VisualResidualActor,
    ResidualForwardMap,
    SharedCriticMap,
    soft_update_params,
    weight_init,
)
from ..storage.replay_buffer import ZBuffer

OBS_ORDER = ("state", "privileged_state", "last_action", "history_actor")
ACTOR_KEYS = ("state", "last_action", "history_actor")
PRIVILEGED_KEYS = ("state", "privileged_state")
FULL_KEYS = OBS_ORDER


def tree_to_device(x, device):
    if isinstance(x, dict):
        return {k: tree_to_device(v, device) for k, v in x.items()}
    return x.to(device)


class FbCprAuxAgent:
    _COMPILE_MODES: tuple[str, ...] = ("default", "max-autotune-no-cudagraphs")

    def __init__(self, obs_dims: Dict[str, int], action_dim: int, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.fb_target_tau = float(cfg.fb_target_tau)
        self.critic_target_tau = float(cfg.critic_target_tau)
        for name in ("fb_target_tau", "critic_target_tau"):
            tau = getattr(self, name)
            if not 0.0 <= tau <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {tau}.")
        precision = str(getattr(cfg, "matmul_precision", "highest")).lower()
        if precision not in ("highest", "high", "medium"):
            raise ValueError(
                f"matmul_precision must be 'highest', 'high' or 'medium', got {precision!r}."
            )
        torch.set_float32_matmul_precision(precision)
        self.use_bf16_amp = bool(getattr(cfg, "use_bf16_amp", False))
        self.use_fused_adam = (
            bool(getattr(cfg, "use_fused_adam", False)) and torch.device(device).type == "cuda"
        )
        self.action_dim = action_dim
        self.obs_dims = {k: obs_dims[k] for k in obs_dims if k != "depth_image"}
        self.actor_keys = tuple(getattr(cfg, "actor_obs_keys", ACTOR_KEYS))
        self.full_keys = tuple(getattr(cfg, "full_obs_keys", FULL_KEYS))
        self.privileged_keys = tuple(getattr(cfg, "privileged_obs_keys", PRIVILEGED_KEYS))
        self.visual_enabled = bool(getattr(cfg, "visual_enabled", False))
        self.depth_shape = tuple(getattr(cfg, "depth_shape", (8, 18, 32)))
        self.actor_obs_dim = sum(self.obs_dims[k] for k in self.actor_keys)
        self.privileged_obs_dim = sum(self.obs_dims[k] for k in self.privileged_keys)
        self.full_obs_dim = sum(self.obs_dims[k] for k in self.full_keys)
        z_dim = cfg.z_dim
        self.z_dim = z_dim
        self.actor_hidden_dim = int(getattr(cfg, "actor_hidden_dim", 1024))
        self.actor_hidden_layers = int(getattr(cfg, "actor_hidden_layers", 6))
        self.forward_hidden_dim = int(getattr(cfg, "forward_hidden_dim", 1024))
        self.forward_hidden_layers = int(getattr(cfg, "forward_hidden_layers", 6))
        self.critic_hidden_dim = int(getattr(cfg, "critic_hidden_dim", 1024))
        self.critic_hidden_layers = int(getattr(cfg, "critic_hidden_layers", 6))
        self.backward_hidden_dim = int(getattr(cfg, "backward_hidden_dim", 256))
        self.backward_hidden_layers = int(getattr(cfg, "backward_hidden_layers", 1))
        self.embedding_layers = int(getattr(cfg, "net_embedding_layers", 2))
        self.shared_value_net = bool(getattr(cfg, "shared_value_net", False))
        # Preserve the original pessimistic value estimate when a task config
        # does not explicitly choose an ensemble size.
        self.critic_members = int(getattr(cfg, "critic_ensemble_members", 2))

        if self.visual_enabled:
            self._actor = VisualResidualActor(
                self.obs_dims,
                self.actor_keys,
                z_dim,
                action_dim,
                hidden_dim=self.actor_hidden_dim,
                hidden_layers=self.actor_hidden_layers,
                embedding_layers=self.embedding_layers,
                depth_shape=self.depth_shape,
                estimate_velocity=bool(getattr(cfg, "estimated_velocity_enabled", True)),
            )
        else:
            self._actor = ResidualActor(
                self.actor_obs_dim, z_dim, action_dim,
                hidden_dim=self.actor_hidden_dim,
                hidden_layers=self.actor_hidden_layers,
                embedding_layers=self.embedding_layers,
            )
        self._backward_map = BackwardMap(
            self.privileged_obs_dim,
            z_dim,
            hidden_dim=self.backward_hidden_dim,
            hidden_layers=self.backward_hidden_layers,
        )
        self._forward_map = ResidualForwardMap(
            self.full_obs_dim, z_dim, action_dim,
            hidden_dim=self.forward_hidden_dim,
            hidden_layers=self.forward_hidden_layers,
            num_parallel=2,
        )
        if self.shared_value_net:
            self._critic_shared = SharedCriticMap(
                self.full_obs_dim, z_dim, action_dim,
                hidden_dim=self.critic_hidden_dim,
                hidden_layers=self.critic_hidden_layers,
                num_parallel=self.critic_members,
            )
        else:
            self._critic = ResidualForwardMap(
                self.full_obs_dim, z_dim, action_dim,
                hidden_dim=self.critic_hidden_dim,
                hidden_layers=self.critic_hidden_layers,
                num_parallel=self.critic_members,
                output_dim=1,
            )
            self._aux_critic = ResidualForwardMap(
                self.full_obs_dim, z_dim, action_dim,
                hidden_dim=self.critic_hidden_dim,
                hidden_layers=self.critic_hidden_layers,
                num_parallel=self.critic_members,
                output_dim=1,
            )
        discriminator_cfg = getattr(cfg, "discriminator", None)
        self.discriminator_loss = str(
            getattr(discriminator_cfg, "loss_type", getattr(cfg, "discriminator_loss", "bce"))
        ).lower()
        if self.discriminator_loss not in {"bce", "gan", "lsgan", "wgan"}:
            raise ValueError(
                "discriminator loss must be one of 'bce', 'gan', 'lsgan', or 'wgan', "
                f"got {self.discriminator_loss!r}."
            )
        discriminator_hidden_dims = getattr(
            discriminator_cfg,
            "hidden_dims",
            getattr(cfg, "discriminator_hidden_dims", (1024, 1024, 1024)),
        )
        self._discriminator = Discriminator(
            self.privileged_obs_dim,
            z_dim,
            hidden_dims=discriminator_hidden_dims,
        )

        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_backward_map = copy.deepcopy(self._backward_map)
        if self.shared_value_net:
            self._target_critic_shared = copy.deepcopy(self._critic_shared)
        else:
            self._target_critic = copy.deepcopy(self._critic)
            self._target_aux_critic = copy.deepcopy(self._aux_critic)

        self._obs_normalizer = ObsNormalizer(self.obs_dims, momentum=0.01)
        self._aux_reward_normalizer = EMA(tau=0.99, translate=False, scale=True)

        value_modules = (
            (self._critic_shared,) if self.shared_value_net else (self._critic, self._aux_critic)
        )
        target_value_modules = (
            (self._target_critic_shared,) if self.shared_value_net
            else (self._target_critic, self._target_aux_critic)
        )
        modules = [
            self._actor, self._backward_map, self._forward_map, *value_modules,
            self._discriminator, self._target_forward_map,
            self._target_backward_map, *target_value_modules,
            self._obs_normalizer,
            self._aux_reward_normalizer,
        ]
        for m in modules:
            m.to(device)
        for m in (self._actor, self._backward_map, self._forward_map, *value_modules, self._discriminator):
            m.apply(weight_init)
        self._target_forward_map.load_state_dict(self._forward_map.state_dict())
        self._target_backward_map.load_state_dict(self._backward_map.state_dict())
        if self.shared_value_net:
            self._target_critic_shared.load_state_dict(self._critic_shared.state_dict())
        else:
            self._target_critic.load_state_dict(self._critic.state_dict())
            self._target_aux_critic.load_state_dict(self._aux_critic.state_dict())

        adam_kwargs = {"fused": self.use_fused_adam}
        self.actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay, **adam_kwargs
        )
        self.forward_optimizer = torch.optim.Adam(
            self._forward_map.parameters(), lr=cfg.lr_f, weight_decay=cfg.weight_decay, **adam_kwargs
        )
        self.backward_optimizer = torch.optim.Adam(
            self._backward_map.parameters(), lr=cfg.lr_b, weight_decay=cfg.weight_decay, **adam_kwargs
        )
        if self.shared_value_net:
            # One optimizer owns the shared trunk and both heads. Their losses
            # are combined before a single backward and optimizer step.
            self.critic_optimizer = torch.optim.Adam(
                self._critic_shared.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay, **adam_kwargs
            )
        else:
            self.critic_optimizer = torch.optim.Adam(
                self._critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay, **adam_kwargs
            )
            self.aux_critic_optimizer = torch.optim.Adam(
                self._aux_critic.parameters(), lr=cfg.lr_aux_critic, weight_decay=cfg.weight_decay, **adam_kwargs
            )
        self.discriminator_optimizer = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=cfg.lr_discriminator,
            weight_decay=cfg.weight_decay_discriminator,
            **adam_kwargs,
        )

        bs = cfg.batch_size
        self.off_diag = 1 - torch.eye(bs, bs, device=device)
        self.off_diag_sum = self.off_diag.sum()
        self.z_buffer = ZBuffer(cfg.z_buffer_size, z_dim, device)
        self.env_idx_with_expert_rollout = None
        self.tracking_z = None
        # Parameter layouts are static across updates. Cache their bucket
        # partitioning so distributed gradient reduction does not rebuild it
        # six times per update.
        self._grad_bucket_cache = {}
        self.compile(getattr(cfg, "torch_compile_mode", None))

    def _autocast_context(self):
        """Use bf16 for update-network forwards while retaining fp32 master weights."""
        return torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.use_bf16_amp
        )

    def compile(self, mode: str | None = None) -> None:
        """Compile network forward methods without changing checkpoint keys."""
        if mode is None:
            return
        if mode not in self._COMPILE_MODES:
            raise ValueError(
                f"torch_compile_mode={mode!r} is not supported; use one of "
                f"{self._COMPILE_MODES} or None."
            )
        value_modules = (
            (self._critic_shared,) if self.shared_value_net
            else (self._critic, self._aux_critic)
        )
        target_value_modules = (
            (self._target_critic_shared,) if self.shared_value_net
            else (self._target_critic, self._target_aux_critic)
        )
        modules = (
            self._actor,
            self._backward_map,
            self._forward_map,
            *value_modules,
            self._target_backward_map,
            self._target_forward_map,
            *target_value_modules,
        )
        for module in modules:
            module.forward = torch.compile(module.forward, mode=mode)  # type: ignore[method-assign]
        # The discriminator stays eager because WGAN-GP requires double backward.

    # --------------------------------------------------------------- inputs

    def _parts(self, obs):
        parts = {k: obs[k] for k in self.obs_dims if k in obs}
        if self.visual_enabled and "depth_image" in obs:
            parts["depth_image"] = obs["depth_image"]
        return parts

    def _norm_parts(self, obs):
        with torch.no_grad():
            parts = self._obs_normalizer.normalize_dict(self._parts(obs))
        if self.visual_enabled and "depth_image" in obs:
            parts["depth_image"] = obs["depth_image"]
        return parts

    def _actor_input(self, parts):
        if self.visual_enabled:
            keys = [*self.actor_keys, "depth_image"]
            if "velocity_target_yaw" in parts:
                keys.append("velocity_target_yaw")
            return {key: parts[key] for key in keys}
        return torch.cat([parts[key] for key in self.actor_keys], dim=-1)

    def _full_input(self, parts):
        return torch.cat([parts[key] for key in self.full_keys], dim=-1)

    def _privileged_input(self, parts):
        return torch.cat([parts[key] for key in self.privileged_keys], dim=-1)

    def _actor_forward(self, parts, z):
        return self._actor(self._actor_input(parts), z, self.cfg.actor_std)

    def _actor_in(self, obs):
        return self._actor_input(self._norm_parts(obs))

    def _full_in(self, obs):
        return self._full_input(self._norm_parts(obs))

    def _b_in(self, obs):
        return self._privileged_input(self._norm_parts(obs))

    # --------------------------------------------------------------- acting

    @torch.no_grad()
    def act(self, obs, z, mean=False):
        dist = self._actor_forward(self._norm_parts(obs), z)
        if mean:
            return dist.mean
        return dist.sample()

    @torch.no_grad()
    def sample_z(self, n):
        z = torch.randn((n, self.z_dim), dtype=torch.float32, device=self.device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    @torch.no_grad()
    def _sample_tracking_z(self, replay_buffers, batch_dim, traj_length):
        batch = replay_buffers["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)
        next_obs = tree_to_device(batch["next"]["observation"], self.device)
        z = self._backward_map(self._b_in(next_obs))
        z = z.view(batch_dim, traj_length, z.shape[-1])
        for step in range(traj_length):
            end_idx = min(step + self.cfg.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self.project_z(z)

    @torch.no_grad()
    def maybe_update_rollout_context(self, z, step_count, replay_buffers=None):
        step_count = step_count.reshape(-1).to(self.device)
        if z is not None:
            mask = (step_count % self.cfg.update_z_every_step == 0).unsqueeze(-1)
            if self.cfg.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0])
            else:
                new_z = self.sample_z(z.shape[0])
            z = torch.where(mask, new_z, z)
        else:
            z = self.sample_z(step_count.shape[0])

        if self.cfg.rollout_expert_trajectories:
            if replay_buffers is None:
                raise ValueError("expert trajectory rollout requires replay buffers")
            length = self.cfg.rollout_expert_trajectories_length
            idxs = step_count % length
            refresh = self.tracking_z is None or torch.any(idxs == 0)
            if refresh:
                n_elem = int(self.cfg.rollout_expert_trajectories_percentage * step_count.shape[0])
                n_elem = max(1, n_elem)
                self.env_idx_with_expert_rollout = torch.randint(
                    0, step_count.shape[0], size=(n_elem,), device=self.device
                )
                self.tracking_z = self._sample_tracking_z(replay_buffers, n_elem, length)
            mod_time = idxs[self.env_idx_with_expert_rollout]
            rows = torch.arange(len(self.env_idx_with_expert_rollout), device=self.device)
            z[self.env_idx_with_expert_rollout] = self.tracking_z[rows, mod_time]
        return z

    # ------------------------------------------------------------- training

    @torch.no_grad()
    def encode_expert(self, next_obs):
        with self._autocast_context():
            b_expert = self._backward_map(next_obs)
        b_expert = b_expert.float()
        seq_length = self.cfg.seq_length
        z_expert = b_expert.view(-1, seq_length, b_expert.shape[-1]).mean(dim=1)
        z_expert = self.project_z(z_expert)
        return torch.repeat_interleave(z_expert, seq_length, dim=0)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal_input, expert_encodings):
        bs = self.cfg.batch_size
        z = self.sample_z(bs)
        p_goal = self.cfg.train_goal_ratio
        p_expert = self.cfg.expert_asm_ratio
        prob = torch.tensor([p_goal, p_expert, 1 - p_goal - p_expert], device=self.device)
        mix = torch.multinomial(prob, bs, replacement=True).reshape(-1, 1)
        perm = torch.randperm(bs, device=self.device)
        with self._autocast_context():
            goals = self._backward_map(train_goal_input[perm])
        goals = self.project_z(goals.float())
        z = torch.where(mix == 0, goals, z)
        perm = torch.randperm(expert_encodings.shape[0], device=self.device)[:bs]
        z = torch.where(mix == 1, expert_encodings[perm], z)
        return z

    def _avg_grad(self, params):
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return
        params = tuple(params)
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return

        param_key = tuple(id(param) for param in params if param.grad is not None)
        buckets = self._grad_bucket_cache.get(param_key)
        if buckets is None:
            bucket_bytes = max(1, int(getattr(self.cfg, "grad_bucket_size_mb", 64))) * 1024 * 1024
            bucket_list, bucket, size = [], [], 0
            for param in params:
                if param.grad is None:
                    continue
                grad_bytes = param.numel() * param.element_size()
                if bucket and size + grad_bytes > bucket_bytes:
                    bucket_list.append(tuple(bucket))
                    bucket, size = [], 0
                bucket.append(param)
                size += grad_bytes
            if bucket:
                bucket_list.append(tuple(bucket))
            buckets = tuple(bucket_list)
            self._grad_bucket_cache[param_key] = buckets

        pending = []
        for params_in_bucket in buckets:
            tensors = [param.grad for param in params_in_bucket]
            flat = _flatten_dense_tensors(tensors)
            work = torch.distributed.all_reduce(flat, async_op=True)
            pending.append((work, flat, tensors))

        scale = 1.0 / torch.distributed.get_world_size()
        for work, flat, tensors in pending:
            work.wait()
            flat.mul_(scale)
            torch._foreach_copy_(tensors, _unflatten_dense_tensors(flat, tensors))

    def update(self, replay_buffers) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        bs = cfg.batch_size
        expert_batch = replay_buffers["expert_slicer"].sample(bs)
        train_batch = replay_buffers["train"].sample(bs)

        train_obs = tree_to_device(train_batch["observation"], self.device)
        train_next_obs = tree_to_device(train_batch["next"]["observation"], self.device)
        train_action = train_batch["action"].to(self.device)
        train_z = train_batch["z"].to(self.device)
        discount = cfg.discount * ~train_batch["next"]["terminated"].to(self.device).squeeze(-1)
        expert_obs = tree_to_device(expert_batch["observation"], self.device)
        expert_next_obs = tree_to_device(expert_batch["next"]["observation"], self.device)

        with torch.no_grad():
            self._obs_normalizer.update_dict(self._parts(train_obs))
            self._obs_normalizer.update_dict(self._parts(train_next_obs))
            t_obs, t_next = self._norm_parts(train_obs), self._norm_parts(train_next_obs)
            e_obs, e_next = self._norm_parts(expert_obs), self._norm_parts(expert_next_obs)

        expert_b_obs = self._b_in_from_parts(e_obs)
        expert_b_next = self._b_in_from_parts(e_next)
        train_b_obs = self._b_in_from_parts(t_obs)
        goal = self._b_in_from_parts(t_next)
        expert_z = self.encode_expert(next_obs=expert_b_next)

        metrics = self.update_discriminator(
            expert_obs=expert_b_obs,
            expert_z=expert_z,
            train_obs=train_b_obs,
            train_z=train_z,
            grad_penalty=cfg.grad_penalty_discriminator,
        )

        z = self.sample_mixed_z(train_goal_input=goal, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)
        if cfg.relabel_ratio is not None:
            mask = torch.rand((bs, 1), device=self.device) <= cfg.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        f_obs, f_next = self._full_in_from_parts(t_obs), self._full_in_from_parts(t_next)
        next_actions = self._sample_target_actions(t_next, train_z, count=3)
        metrics.update(
            self.update_fb(
                obs=f_obs,
                action=train_action,
                discount=discount,
                next_obs=f_next,
                goal=goal,
                z=train_z,
                next_action=next_actions[0],
            )
        )
        aux_reward = torch.zeros((bs, 1), device=self.device)
        for name in cfg.aux_rewards:
            r = train_batch["aux_rewards"][name].to(self.device)
            if r.dim() == 2:
                r = r.squeeze(-1)
            scale = float(cfg.aux_rewards_scaling[name])
            aux_reward = aux_reward + (scale * r).unsqueeze(-1)
        aux_reward = self._aux_reward_normalizer(aux_reward)

        if self.shared_value_net:
            metrics.update(
                self.update_shared_critics(
                    obs=f_obs,
                    b_obs=train_b_obs,
                    action=train_action,
                    discount=discount,
                    aux_reward=aux_reward,
                    next_obs=f_next,
                    z=train_z,
                    next_action=next_actions[1],
                    next_aux_action=next_actions[2],
                )
            )
        else:
            metrics.update(
                self.update_critic(
                    obs=f_obs,
                    b_obs=train_b_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=f_next,
                    z=train_z,
                    next_action=next_actions[1],
                )
            )

            metrics.update(
                self.update_aux_critic(
                    obs=f_obs,
                    action=train_action,
                    discount=discount,
                    aux_reward=aux_reward,
                    next_obs=f_next,
                    z=train_z,
                    next_action=next_actions[2],
                )
            )
        metrics.update(self.update_actor(actor_obs=self._actor_in_from_parts(t_obs), full_obs=f_obs, z=train_z))

        soft_update_params(self._forward_map, self._target_forward_map, self.fb_target_tau)
        soft_update_params(self._backward_map, self._target_backward_map, self.fb_target_tau)
        if self.shared_value_net:
            soft_update_params(self._critic_shared, self._target_critic_shared, self.critic_target_tau)
        else:
            soft_update_params(self._critic, self._target_critic, self.critic_target_tau)
            soft_update_params(self._aux_critic, self._target_aux_critic, self.critic_target_tau)
        self._sync_floating_buffers()
        return metrics

    def _sync_floating_buffers(self):
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return
        world_size = torch.distributed.get_world_size()
        modules = (self._obs_normalizer, self._aux_reward_normalizer)
        with torch.no_grad():
            for module in modules:
                for buffer in module.buffers():
                    if buffer.is_floating_point():
                        torch.distributed.all_reduce(buffer)
                        buffer.div_(world_size)

    def _b_in_from_parts(self, p):
        return self._privileged_input(p)

    def _full_in_from_parts(self, p):
        return self._full_input(p)

    def _actor_in_from_parts(self, p):
        return self._actor_input(p)

    def _sample_target_actions(self, next_parts, z, count):
        with torch.no_grad():
            with self._autocast_context():
                distribution = self._actor_forward(next_parts, z)
                actions = tuple(
                    distribution.sample(clip=self.cfg.stddev_clip) for _ in range(count)
                )
            return tuple(action.float() for action in actions)

    # -------------------------------------------------------------- updates

    def update_discriminator(self, expert_obs, expert_z, train_obs, train_z, grad_penalty):
        with self._autocast_context():
            expert_logits = self._discriminator.compute_logits(obs=expert_obs, z=expert_z)
            unlabeled_logits = self._discriminator.compute_logits(obs=train_obs, z=train_z)
        expert_logits = expert_logits.float()
        unlabeled_logits = unlabeled_logits.float()
        if self.discriminator_loss == "bce":
            expert_loss = -F.logsigmoid(expert_logits)
            unlabeled_loss = F.softplus(unlabeled_logits)
            loss = torch.mean(expert_loss + unlabeled_loss)
        elif self.discriminator_loss == "gan":
            expert_loss = F.softplus(-expert_logits)
            unlabeled_loss = F.softplus(unlabeled_logits)
            loss = 0.5 * (expert_loss.mean() + unlabeled_loss.mean())
        elif self.discriminator_loss == "lsgan":
            expert_loss = 0.5 * (expert_logits - 1.0).square()
            unlabeled_loss = 0.5 * (unlabeled_logits + 1.0).square()
            loss = expert_loss.mean() + unlabeled_loss.mean()
        else:
            expert_loss = -expert_logits
            unlabeled_loss = unlabeled_logits
            loss = expert_loss.mean() + unlabeled_loss.mean()
        if grad_penalty is not None and grad_penalty > 0:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss = loss + grad_penalty * wgan_gp
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._avg_grad(self._discriminator.parameters())
        self.discriminator_optimizer.step()
        out = {
            "disc_loss": loss.detach(),
            "disc_expert_loss": expert_loss.detach().mean(),
            "disc_train_loss": unlabeled_loss.detach().mean(),
        }
        if grad_penalty is not None and grad_penalty > 0:
            out["disc_wgan_gp_loss"] = wgan_gp.detach()
        return out

    def gradient_penalty_wgan(self, real_obs, real_z, fake_obs, fake_z):
        bs = real_obs.shape[0]
        alpha = torch.rand(bs, 1, device=real_z.device)
        interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
        interpolated_z = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)
        with self._autocast_context():
            d_interpolates = self._discriminator.compute_logits(interpolated_obs, interpolated_z)
        d_interpolates = d_interpolates.float()
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=[interpolated_obs, interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        cat_gradients = torch.cat(gradients, dim=1)
        return ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

    def update_fb(self, obs, action, discount, next_obs, goal, z, next_action):
        with torch.no_grad():
            with self._autocast_context():
                target_fs = self._target_forward_map(next_obs, z, next_action)
                target_b = self._target_backward_map(goal)
            target_fs = target_fs.float()
            target_b = target_b.float()
            target_ms = torch.matmul(target_fs, target_b.transpose(0, 1))
            _, _, target_m = self._targets_uncertainty(target_ms, self.cfg.fb_pessimism_penalty)

        with self._autocast_context():
            fs = self._forward_map(obs, z, action)
            b = self._backward_map(goal)
        fs = fs.float()
        b = b.float()
        ms = torch.matmul(fs, b.transpose(0, 1))

        diff = ms - discount.unsqueeze(-1) * target_m
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        cov = torch.matmul(b, b.transpose(0, 1))
        orth_diag = -cov.diag().mean()
        orth_offdiag = 0.5 * (cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_offdiag + orth_diag
        fb_loss = fb_loss + self.cfg.ortho_coef * orth_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self._avg_grad([*self._forward_map.parameters(), *self._backward_map.parameters()])
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        return {
            "target_M": target_m.mean().detach(),
            "M1": ms[0].mean().detach(),
            "F1": fs[0].mean().detach(),
            "B": b.mean().detach(),
            "B_norm": torch.norm(b, dim=-1).mean().detach(),
            "z_norm": torch.norm(z, dim=-1).mean().detach(),
            "fb_loss": fb_loss.detach(),
            "fb_diag": fb_diag.detach(),
            "fb_offdiag": fb_offdiag.detach(),
            "orth_loss": orth_loss.detach(),
            "orth_loss_diag": orth_diag.detach(),
            "orth_loss_offdiag": orth_offdiag.detach(),
        }

    def update_critic(self, obs, b_obs, action, discount, next_obs, z, next_action):
        cfg = self.cfg
        num_parallel = self.critic_members
        with torch.no_grad():
            with self._autocast_context():
                reward_logits = self._discriminator.compute_logits(obs=b_obs, z=z)
            probability = torch.sigmoid(reward_logits.float()).clamp(1e-7, 1 - 1e-7)
            reward = probability.log() - (1 - probability).log()
            with self._autocast_context():
                next_qs = self._target_critic(next_obs, z, next_action)
            next_qs = next_qs.float()
            q_mean, q_unc, next_v = self._targets_uncertainty(next_qs, cfg.critic_pessimism_penalty)
            target_q = reward + discount.unsqueeze(-1) * next_v
            expanded = target_q.expand(num_parallel, -1, -1)
        with self._autocast_context():
            qs = self._critic(obs, z, action)
        qs = qs.float()
        critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._avg_grad(self._critic.parameters())
        self.critic_optimizer.step()
        return {
            "critic_loss": (critic_loss / num_parallel).detach(),
            "target_Q": target_q.mean().detach(),
            "Q1": qs.mean().detach(),
            "mean_next_Q": q_mean.mean().detach(),
            "unc_Q": q_unc.mean().detach(),
            "mean_disc_reward": reward.mean().detach(),
        }

    def update_aux_critic(self, obs, action, discount, aux_reward, next_obs, z, next_action):
        cfg = self.cfg
        num_parallel = self.critic_members
        with torch.no_grad():
            with self._autocast_context():
                next_qs = self._target_aux_critic(next_obs, z, next_action)
            next_qs = next_qs.float()
            q_mean, q_unc, next_v = self._targets_uncertainty(next_qs, cfg.aux_critic_pessimism_penalty)
            target_q = aux_reward + discount.unsqueeze(-1) * next_v
            expanded = target_q.expand(num_parallel, -1, -1)
        with self._autocast_context():
            qs = self._aux_critic(obs, z, action)
        qs = qs.float()
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded)
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self._avg_grad(self._aux_critic.parameters())
        self.aux_critic_optimizer.step()
        return {
            "aux_critic_loss": (aux_critic_loss / num_parallel).detach(),
            "target_auxQ": target_q.mean().detach(),
            "auxQ1": qs.mean().detach(),
            "mean_next_auxQ": q_mean.mean().detach(),
            "unc_auxQ": q_unc.mean().detach(),
            "mean_aux_reward": aux_reward.mean().detach(),
        }

    def update_shared_critics(
        self,
        obs,
        b_obs,
        action,
        discount,
        aux_reward,
        next_obs,
        z,
        next_action,
        next_aux_action,
    ):
        """Update both shared critic heads with one backward and optimizer step."""
        cfg = self.cfg
        num_parallel = self.critic_members
        with torch.no_grad():
            with self._autocast_context():
                reward_logits = self._discriminator.compute_logits(obs=b_obs, z=z)
            probability = torch.sigmoid(reward_logits.float()).clamp(1e-7, 1 - 1e-7)
            reward = probability.log() - (1 - probability).log()
            with self._autocast_context():
                next_qs = self._target_critic_shared(next_obs, z, next_action)[0]
                next_aux_qs = self._target_critic_shared(next_obs, z, next_aux_action)[1]
            next_qs = next_qs.float()
            next_aux_qs = next_aux_qs.float()
            q_mean, q_unc, next_v = self._targets_uncertainty(
                next_qs, cfg.critic_pessimism_penalty
            )
            aux_q_mean, aux_q_unc, next_aux_v = self._targets_uncertainty(
                next_aux_qs, cfg.aux_critic_pessimism_penalty
            )
            target_q = reward + discount.unsqueeze(-1) * next_v
            target_aux_q = aux_reward + discount.unsqueeze(-1) * next_aux_v

        with self._autocast_context():
            qs, aux_qs = self._critic_shared(obs, z, action)
        qs = qs.float()
        aux_qs = aux_qs.float()
        expanded_q = target_q.expand(num_parallel, -1, -1)
        expanded_aux_q = target_aux_q.expand(num_parallel, -1, -1)
        critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_q)
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(aux_qs, expanded_aux_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        (critic_loss + aux_critic_loss).backward()
        self._avg_grad(self._critic_shared.parameters())
        self.critic_optimizer.step()
        return {
            "critic_loss": (critic_loss / num_parallel).detach(),
            "target_Q": target_q.mean().detach(),
            "Q1": qs.mean().detach(),
            "mean_next_Q": q_mean.mean().detach(),
            "unc_Q": q_unc.mean().detach(),
            "mean_disc_reward": reward.mean().detach(),
            "aux_critic_loss": (aux_critic_loss / num_parallel).detach(),
            "target_auxQ": target_aux_q.mean().detach(),
            "auxQ1": aux_qs.mean().detach(),
            "mean_next_auxQ": aux_q_mean.mean().detach(),
            "unc_auxQ": aux_q_unc.mean().detach(),
            "mean_aux_reward": aux_reward.mean().detach(),
        }

    def update_actor(self, actor_obs, full_obs, z):
        cfg = self.cfg
        self.actor_optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            action_dist = self._actor_forward(actor_obs, z)
            action = action_dist.sample(clip=cfg.stddev_clip)
        action = action.float()

        if self.shared_value_net:
            frozen_modules = [self._critic_shared, self._forward_map]
        else:
            frozen_modules = [self._critic, self._aux_critic, self._forward_map]
        for name in ("_heading_critic", "_velocity_critic"):
            if hasattr(self, name):
                frozen_modules.append(getattr(self, name))
        frozen_params = [p for module in frozen_modules for p in module.parameters()]
        requires_grad = [p.requires_grad for p in frozen_params]
        for param in frozen_params:
            param.requires_grad_(False)
        try:
            with self._autocast_context():
                if self.shared_value_net:
                    qs_disc, qs_aux = self._critic_shared(full_obs, z, action)
                else:
                    qs_disc = self._critic(full_obs, z, action)
                    qs_aux = self._aux_critic(full_obs, z, action)
                fs = self._forward_map(full_obs, z, action)
            qs_disc = qs_disc.float()
            _, _, q_disc = self._targets_uncertainty(qs_disc, cfg.actor_pessimism_penalty)

            qs_aux = qs_aux.float()
            _, _, q_aux = self._targets_uncertainty(qs_aux, cfg.actor_pessimism_penalty)

            q_heading = torch.zeros_like(q_aux)
            if hasattr(self, "_heading_critic"):
                qs_heading = self._heading_critic(full_obs, z, action).float()
                _, _, q_heading = self._targets_uncertainty(qs_heading, cfg.actor_pessimism_penalty)

            q_velocity = torch.zeros_like(q_heading)
            if hasattr(self, "_velocity_critic"):
                qs_velocity = self._velocity_critic(full_obs, z, action).float()
                _, _, q_velocity = self._targets_uncertainty(qs_velocity, cfg.actor_pessimism_penalty)

            fs = fs.float()
            qs_fb = (fs * z).sum(-1)
            _, _, q_fb = self._targets_uncertainty(qs_fb, cfg.actor_pessimism_penalty)

            weight = q_fb.abs().mean().detach() if cfg.scale_reg else 1.0
            actor_loss = (
                -q_disc.mean() * cfg.reg_coeff * weight
                - q_aux.mean() * cfg.reg_coeff_aux * weight
                - q_fb.mean()
            )
            if hasattr(self, "_heading_critic"):
                actor_loss = actor_loss - q_heading.mean() * float(getattr(cfg, "reg_coeff_heading", 0.0)) * weight
            if hasattr(self, "_velocity_critic"):
                actor_loss = actor_loss - q_velocity.mean() * float(getattr(cfg, "reg_coeff_velocity", 0.0)) * weight
            estimator_loss = None
            if hasattr(action_dist, "velocity_estimate") and isinstance(actor_obs, dict) and "velocity_target_yaw" in actor_obs:
                estimator_loss = F.smooth_l1_loss(
                    action_dist.velocity_estimate.float(),
                    actor_obs["velocity_target_yaw"].detach().float(),
                )
                actor_loss = actor_loss + float(getattr(cfg, "velocity_estimator_loss_coef", 1.0)) * estimator_loss
            actor_loss.backward()
        finally:
            for param, enabled in zip(frozen_params, requires_grad):
                param.requires_grad_(enabled)
        self._avg_grad(self._actor.parameters())
        self.actor_optimizer.step()
        metrics = {
            "actor_loss": actor_loss.detach(),
            "Q_discriminator": q_disc.mean().detach(),
            "Q_aux": q_aux.mean().detach(),
            "Q_fb": q_fb.mean().detach(),
        }
        if hasattr(self, "_heading_critic"):
            metrics["Q_heading"] = q_heading.mean().detach()
        if hasattr(self, "_velocity_critic"):
            metrics["Q_velocity"] = q_velocity.mean().detach()
        if estimator_loss is not None:
            metrics["velocity_estimator_loss"] = estimator_loss.detach()
        return metrics

    @staticmethod
    def _targets_uncertainty(preds, pessimism_penalty):
        dim = 0
        preds_mean = preds.mean(dim=dim)
        if preds.shape[dim] == 1:
            # A single member carries no disagreement signal.
            preds_unc = torch.zeros_like(preds_mean)
            return preds_mean, preds_unc, preds_mean
        if preds.shape[dim] == 2:
            # For two ensemble members, the generic [N,N,...] pairwise
            # broadcast reduces exactly to abs(pred0 - pred1). This avoids a
            # large temporary tensor for the BxB FB target matrix.
            preds_unc = torch.abs(preds[0] - preds[1])
            return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
        preds_diffs = torch.abs(preds.unsqueeze(dim) - preds.unsqueeze(dim + 1))
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / num_parallel_scaling
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    # ------------------------------------------------------------------- io

    def state_dict(self):
        out = {
            "actor": self._actor.state_dict(),
            "backward_map": self._backward_map.state_dict(),
            "forward_map": self._forward_map.state_dict(),
            "discriminator": self._discriminator.state_dict(),
            "target_forward_map": self._target_forward_map.state_dict(),
            "target_backward_map": self._target_backward_map.state_dict(),
            "obs_normalizer": self._obs_normalizer.state_dict(),
            "aux_reward_normalizer": self._aux_reward_normalizer.state_dict(),
        }
        if self.shared_value_net:
            out["critic_shared"] = self._critic_shared.state_dict()
            out["target_critic_shared"] = self._target_critic_shared.state_dict()
        else:
            out["critic"] = self._critic.state_dict()
            out["aux_critic"] = self._aux_critic.state_dict()
            out["target_critic"] = self._target_critic.state_dict()
            out["target_aux_critic"] = self._target_aux_critic.state_dict()
        return out

    def load_state_dict(self, state):
        self._actor.load_state_dict(state["actor"])
        self._backward_map.load_state_dict(state["backward_map"])
        self._forward_map.load_state_dict(state["forward_map"])
        self._discriminator.load_state_dict(state["discriminator"])
        self._target_forward_map.load_state_dict(state.get("target_forward_map", state["forward_map"]))
        self._target_backward_map.load_state_dict(state.get("target_backward_map", state["backward_map"]))
        if self.shared_value_net:
            if "critic_shared" not in state:
                raise ValueError(
                    "checkpoint does not contain shared critic weights and cannot be "
                    "loaded into a shared_value_net model"
                )
            self._critic_shared.load_state_dict(state["critic_shared"])
            self._target_critic_shared.load_state_dict(
                state.get("target_critic_shared", state["critic_shared"])
            )
        else:
            self._critic.load_state_dict(state["critic"])
            self._aux_critic.load_state_dict(state["aux_critic"])
            self._target_critic.load_state_dict(state.get("target_critic", state["critic"]))
            self._target_aux_critic.load_state_dict(state.get("target_aux_critic", state["aux_critic"]))
        self._obs_normalizer.load_state_dict(state["obs_normalizer"])
        if "aux_reward_normalizer" in state:
            self._aux_reward_normalizer.load_state_dict(state["aux_reward_normalizer"])

    def optimizer_state_dict(self):
        out = {
            "actor": self.actor_optimizer.state_dict(),
            "forward": self.forward_optimizer.state_dict(),
            "backward": self.backward_optimizer.state_dict(),
            "critic": self.critic_optimizer.state_dict(),
            "discriminator": self.discriminator_optimizer.state_dict(),
        }
        if not self.shared_value_net:
            out["aux_critic"] = self.aux_critic_optimizer.state_dict()
        return out

    def load_optimizer_state_dict(self, state):
        self.actor_optimizer.load_state_dict(state["actor"])
        self.forward_optimizer.load_state_dict(state["forward"])
        self.backward_optimizer.load_state_dict(state["backward"])
        self.critic_optimizer.load_state_dict(state["critic"])
        if not self.shared_value_net:
            self.aux_critic_optimizer.load_state_dict(state["aux_critic"])
        self.discriminator_optimizer.load_state_dict(state["discriminator"])
