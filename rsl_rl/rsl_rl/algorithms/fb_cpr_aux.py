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
import json
import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch import autograd
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from .gradient_audit import eager_gradient_diagnostics, parameter_gradient_metrics

from .fb_networks import (
    BackwardMap,
    BackwardContextOnnxExporter,
    Discriminator,
    DepthEncoderOnnxExporter,
    EMA,
    ObsNormalizer,
    ResidualActor,
    VisualResidualActor,
    VisualActorOnnxExporter,
    ResidualForwardMap,
    SharedCriticMap,
    model_kwargs,
    soft_update_params,
    weight_init,
)
from ..storage.replay_buffer import ZBuffer
from ..utils.finite import check_finite_tree

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

    def __init__(self, obs_dims: Dict[str, int], action_dim: int, cfg, device="cuda", *, inference_only=False):
        self.inference_only = inference_only
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
        depth_encoder_cfg = model_kwargs(cfg.depth_encoder) if self.visual_enabled else None
        self.depth_shape = (
            tuple(depth_encoder_cfg.pop("input_shape"))
            if self.visual_enabled and "input_shape" in depth_encoder_cfg
            else None
        )
        if self.visual_enabled and self.depth_shape is None:
            raise ValueError("visual_enabled requires depth_encoder.input_shape")
        self.actor_obs_dim = sum(self.obs_dims[k] for k in self.actor_keys)
        self.privileged_obs_dim = sum(self.obs_dims[k] for k in self.privileged_keys)
        self.full_obs_dim = sum(self.obs_dims[k] for k in self.full_keys)
        z_dim = cfg.z_dim
        self.z_dim = z_dim
        self.shared_value_net = cfg.shared_value_net
        self.critic_members = cfg.critic_ensemble_members

        if self.visual_enabled:
            self._actor = VisualResidualActor(
                self.obs_dims,
                self.actor_keys,
                z_dim,
                action_dim,
                depth_shape=self.depth_shape,
                encoder_cfg=depth_encoder_cfg,
                estimator_cfg=model_kwargs(cfg.velocity_estimator),
                estimate_velocity=cfg.estimated_velocity_enabled,
                velocity_estimator_updates_encoder=getattr(cfg, "velocity_estimator_updates_encoder", True),
                **model_kwargs(cfg.actor),
            )
        else:
            self._actor = ResidualActor(
                self.actor_obs_dim, z_dim, action_dim,
                **model_kwargs(cfg.actor),
            )
        self._backward_map = BackwardMap(
            self.privileged_obs_dim,
            z_dim,
            norm=cfg.norm_z,
            **model_kwargs(cfg.backward_map),
        )
        if inference_only:
            context_keys = ("heading", "velocity_command", "velocity_target_yaw") if self.visual_enabled else ()
            self._obs_normalizer = ObsNormalizer(self.obs_dims, momentum=0.01, identity_keys=context_keys)
            for module in (self._actor, self._backward_map, self._obs_normalizer):
                module.to(device)
                module.requires_grad_(False)
            return
        self._forward_map = ResidualForwardMap(
            self.full_obs_dim, z_dim, action_dim,
            num_parallel=cfg.forward_ensemble_members,
            **model_kwargs(cfg.forward_map),
        )
        if self.shared_value_net:
            self._critic_shared = SharedCriticMap(
                self.full_obs_dim, z_dim, action_dim,
                num_parallel=self.critic_members,
                **model_kwargs(cfg.critic),
            )
        else:
            self._critic = ResidualForwardMap(
                self.full_obs_dim, z_dim, action_dim,
                num_parallel=self.critic_members,
                output_dim=1,
                **model_kwargs(cfg.critic),
            )
            self._aux_critic = ResidualForwardMap(
                self.full_obs_dim, z_dim, action_dim,
                num_parallel=self.critic_members,
                output_dim=1,
                **model_kwargs(cfg.aux_critic),
            )
        # The runner receives serialized config dictionaries from the training entrypoint.
        discriminator_cfg = model_kwargs(cfg.discriminator)
        self.discriminator_loss = discriminator_cfg.pop("loss_type").lower()
        self.discriminator_reward_type = discriminator_cfg.pop("reward_type").lower()
        if self.discriminator_loss not in {"bce", "gan", "lsgan", "wgan"}:
            raise ValueError(
                "discriminator loss must be one of 'bce', 'gan', 'lsgan', or 'wgan', "
                f"got {self.discriminator_loss!r}."
            )
        if self.discriminator_reward_type not in {"log_odds", "amp", "softplus", "raw"}:
            raise ValueError(f"Unsupported discriminator reward: {self.discriminator_reward_type!r}.")
        self._discriminator = Discriminator(
            self.privileged_obs_dim,
            z_dim,
            **discriminator_cfg,
        )

        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_backward_map = copy.deepcopy(self._backward_map)
        if self.shared_value_net:
            self._target_critic_shared = copy.deepcopy(self._critic_shared)
        else:
            self._target_critic = copy.deepcopy(self._critic)
            self._target_aux_critic = copy.deepcopy(self._aux_critic)

        context_keys = ("heading", "velocity_command", "velocity_target_yaw") if self.visual_enabled else ()
        self._obs_normalizer = ObsNormalizer(self.obs_dims, momentum=0.01, identity_keys=context_keys)
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
    def export_visual_onnx(self, path, *, encoder_filename="depth_encoder.onnx",
                           actor_filename="actor.onnx",
                           context_filename="backward_encoder.onnx",
                           metadata_filename="pufo_onnx.json",
                           opset_version=18):
        """Export depth, context/backward, and actor graphs with explicit contracts."""
        if not self.visual_enabled:
            raise RuntimeError("PUFO ONNX export requires visual_enabled=True.")
        expected_actor_keys = ("state", "last_action", "history_actor", "heading", "velocity_command")
        if tuple(self.actor_keys) != expected_actor_keys:
            raise ValueError(
                "PUFO ONNX export currently expects actor_obs_keys="
                f"{expected_actor_keys}, got {tuple(self.actor_keys)}."
            )
        os.makedirs(path, exist_ok=True)
        self._actor.eval()
        self._actor.depth_encoder.eval()
        self._obs_normalizer.eval()
        encoder = DepthEncoderOnnxExporter(self._actor.depth_encoder).eval()
        actor = VisualActorOnnxExporter(self._actor, self._obs_normalizer).eval()
        depth_dtype = torch.uint8 if self._actor.depth_encoder.input_dtype == "uint8" else torch.float32
        depth_input_name = f"depth_{self._actor.depth_encoder.input_dtype}"
        depth = torch.zeros((1, *self.depth_shape), dtype=depth_dtype, device=self.device)
        latent = encoder(depth)
        dummy = {
            "state": torch.zeros((1, self.obs_dims["state"]), device=self.device),
            "last_action": torch.zeros((1, self.obs_dims["last_action"]), device=self.device),
            "history_actor": torch.zeros((1, self.obs_dims["history_actor"]), device=self.device),
            "heading": torch.zeros((1, self.obs_dims["heading"]), device=self.device),
            "velocity_command": torch.zeros((1, self.obs_dims["velocity_command"]), device=self.device),
            "z": torch.zeros((1, self.z_dim), device=self.device),
            "depth_latent": latent,
        }
        encoder_path = os.path.join(path, encoder_filename)
        actor_path = os.path.join(path, actor_filename)
        context_path = os.path.join(path, context_filename)
        dynamic_axes = {"batch": {0: "batch"}}
        torch.onnx.export(
            encoder,
            (depth,),
            encoder_path,
            opset_version=opset_version,
            input_names=[depth_input_name],
            output_names=["depth_latent"],
            dynamic_axes={depth_input_name: {0: "batch"}, "depth_latent": {0: "batch"}},
            external_data=False,
            do_constant_folding=True,
            dynamo=False,
        )
        torch.onnx.export(
            actor,
            tuple(dummy.values()),
            actor_path,
            opset_version=opset_version,
            input_names=list(dummy),
            output_names=["actions"],
            dynamic_axes={name: dynamic_axes["batch"] for name in (*dummy, "actions")},
            external_data=False,
            do_constant_folding=True,
            dynamo=False,
        )
        # Keep context generation independent from the actor graph.  The
        # backward map consumes the same normalized privileged observations as
        # training and emits the z fed to the actor.
        context = BackwardContextOnnxExporter(
            copy.deepcopy(self._backward_map).to("cpu").eval(),
            copy.deepcopy(self._obs_normalizer).to("cpu").eval(),
            self.privileged_keys,
            # BackwardMap already contains Norm when cfg.norm_z is enabled;
            # preserve that exact graph instead of normalizing twice.
            norm_z=False,
        ).eval()
        context_inputs = tuple(
            torch.zeros((1, self.obs_dims[key]), dtype=torch.float32)
            for key in self.privileged_keys
        )
        torch.onnx.export(
            context,
            context_inputs,
            context_path,
            opset_version=opset_version,
            input_names=list(self.privileged_keys),
            output_names=["z"],
            dynamic_axes={key: {0: "batch"} for key in (*self.privileged_keys, "z")},
            external_data=False,
            do_constant_folding=True,
            dynamo=False,
        )
        metadata = {
            "format": "pufo_visual_onnx_v1",
            "depth_encoder": {
                "file": encoder_filename,
                "input_name": depth_input_name,
                "input_dtype": self._actor.depth_encoder.input_dtype,
                "input_range": list(self._actor.depth_encoder.input_range),
                "input_shape": [self.depth_shape[0], self.depth_shape[1], self.depth_shape[2]],
                "output_name": "depth_latent",
                "output_dim": self._actor.depth_encoder.output_size,
            },
            "actor": {
                "file": actor_filename,
                "inputs": {
                    key: int(self.obs_dims[key]) for key in expected_actor_keys
                } | {"z": int(self.z_dim), "depth_latent": int(self._actor.depth_encoder.output_size)},
                "output_name": "actions",
                "action_dim": int(self.action_dim),
                "action_rescale": float(getattr(self.cfg, "action_rescale", 1.0)),
                "action_range": [-1.0, 1.0],
                "velocity_estimator_inside_actor": bool(self._actor.estimate_velocity),
            },
            "context": {
                "exported": True,
                "file": context_filename,
                "inputs": {
                    key: int(self.obs_dims[key]) for key in self.privileged_keys
                },
                "input_order": list(self.privileged_keys),
                "output_name": "z",
                "z_dim": int(self.z_dim),
                "z_projected": bool(getattr(self.cfg, "norm_z", False)),
                "normalization_in_graph": True,
                "z_source": "privileged_obs_keys",
                "heading_source": "external",
                "velocity_command_source": "external",
                "training_generator": "PufoAgent.advance_rollout_context",
                "note": "Run this graph first and feed its z output to the actor graph. Context inputs must be available at deployment.",
            },
        }
        with open(os.path.join(path, metadata_filename), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        try:
            import onnx

            onnx.checker.check_model(onnx.load(encoder_path))
            onnx.checker.check_model(onnx.load(actor_path))
            onnx.checker.check_model(onnx.load(context_path))
        except ImportError:
            # ONNX is optional for training; the files are still usable by a
            # deployment environment that provides the runtime dependencies.
            pass
        return {
            "depth_encoder": encoder_path,
            "context": context_path,
            "actor": actor_path,
            "metadata": os.path.join(path, metadata_filename),
        }

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
    def sample_mixed_context_components(self, train_goal_input, expert_encodings):
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
        expert_indices = torch.randperm(expert_encodings.shape[0], device=self.device)[:bs]
        z = torch.where(mix == 1, expert_encodings[expert_indices], z)
        return z, mix, expert_indices

    @torch.no_grad()
    def sample_mixed_z(self, train_goal_input, expert_encodings):
        return self.sample_mixed_context_components(train_goal_input, expert_encodings)[0]

    def _prepare_train_observations(self, train_batch, obs, next_obs):
        return obs, next_obs

    def _relabel_train_observations(
        self, train_batch, expert_batch, obs, next_obs, sampled_z, mix_source, expert_indices, relabel_mask
    ):
        return obs, next_obs

    def _norm_relabelled_parts(self, obs, normalized):
        """Subclasses that relabel observations must normalize their changed fields here."""
        return normalized

    def _context_batch(self, train_batch, obs, next_obs, z, discount):
        return {
            "actor_obs": None,
            "next_actor_obs": next_obs,
            "obs": self._full_in_from_parts(obs),
            "next_obs": self._full_in_from_parts(next_obs),
            "z": z,
            "next_z": z,
            "discount": discount,
            "heading_valid": None,
            "velocity_valid": None,
        }

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

    def update(
        self, replay_buffers, *, depth_gradient_diagnostics: bool = False,
        reward_term_scales: dict[str, float] | None = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        bs = cfg.batch_size
        expert_batch = replay_buffers["expert_slicer"].sample(bs)
        train_batch = replay_buffers["train"].sample(bs)
        check_finite_tree({"expert_batch": expert_batch, "train_batch": train_batch}, "batches")

        train_obs = tree_to_device(train_batch["observation"], self.device)
        train_next_obs = tree_to_device(train_batch["next"]["observation"], self.device)
        train_action = train_batch["action"].to(self.device)
        train_z = train_batch["z"].to(self.device)
        rollout_z = train_z
        transition_terminated = train_batch["next"].get(
            "transition_terminated", train_batch["next"]["terminated"]
        )
        discount = cfg.discount * ~transition_terminated.to(self.device).squeeze(-1)
        expert_obs = tree_to_device(expert_batch["observation"], self.device)
        expert_next_obs = tree_to_device(expert_batch["next"]["observation"], self.device)

        train_obs, train_next_obs = self._prepare_train_observations(
            train_batch, train_obs, train_next_obs
        )
        with torch.no_grad():
            self._obs_normalizer.update_dict(self._parts(train_obs))
            self._obs_normalizer.update_dict(self._parts(train_next_obs))
            t_obs, t_next = self._norm_parts(train_obs), self._norm_parts(train_next_obs)
            e_obs, e_next = self._norm_parts(expert_obs), self._norm_parts(expert_next_obs)
        collection_obs, collection_next_obs = t_obs, t_next

        expert_b_obs = self._b_in_from_parts(e_obs)
        expert_b_next = self._b_in_from_parts(e_next)
        train_b_obs = self._b_in_from_parts(t_obs)
        goal = self._b_in_from_parts(t_next)
        expert_z = self.encode_expert(next_obs=expert_b_next)
        check_finite_tree(expert_z, "expert_z")

        metrics = self.update_discriminator(
            expert_obs=expert_b_obs,
            expert_z=expert_z,
            train_obs=train_b_obs,
            train_z=train_z,
            grad_penalty=cfg.grad_penalty_discriminator,
        )

        z, mix_source, expert_indices = self.sample_mixed_context_components(
            train_goal_input=goal, expert_encodings=expert_z
        )
        z = z.clone()
        check_finite_tree(z, "sampled_z")
        self.z_buffer.add(z)
        mask = (
            torch.zeros((bs, 1), dtype=torch.bool, device=self.device)
            if cfg.relabel_ratio is None
            else torch.rand((bs, 1), device=self.device) <= cfg.relabel_ratio
        )
        train_z = torch.where(mask, z, train_z)
        train_obs, train_next_obs = self._relabel_train_observations(
            train_batch,
            expert_batch,
            train_obs,
            train_next_obs,
            z,
            mix_source,
            expert_indices,
            mask,
        )
        t_obs = self._norm_relabelled_parts(train_obs, t_obs)
        t_next = self._norm_relabelled_parts(train_next_obs, t_next)

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
        if reward_term_scales:
            metrics.update(self._replay_reward_metrics(train_batch["aux_rewards"], reward_term_scales))
        aux_reward = train_batch["aux_rewards"]["aux_reward"].to(self.device)
        if aux_reward.dim() == 1:
            aux_reward = aux_reward.unsqueeze(-1)
        elif aux_reward.dim() != 2 or aux_reward.shape[-1] != 1:
            raise ValueError(f"Expected aux_reward with shape [batch] or [batch, 1], got {tuple(aux_reward.shape)}")
        aux_reward = self._aux_reward_normalizer(aux_reward)
        check_finite_tree(aux_reward, "normalized_aux_reward")

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
        context = self._context_batch(
            train_batch, collection_obs, collection_next_obs, rollout_z, discount
        )
        metrics.update(self._update_context_critics(train_batch=train_batch, action=train_action, **context))
        metrics.update(
            self.update_actor(
                actor_obs=t_obs,
                full_obs=f_obs,
                z=train_z,
                context_actor_obs=context["actor_obs"],
                context_full_obs=context["obs"],
                context_z=context["z"],
                heading_valid=context["heading_valid"],
                velocity_valid=context["velocity_valid"],
                depth_gradient_diagnostics=depth_gradient_diagnostics,
            )
        )

        soft_update_params(self._forward_map, self._target_forward_map, self.fb_target_tau)
        soft_update_params(self._backward_map, self._target_backward_map, self.fb_target_tau)
        if self.shared_value_net:
            soft_update_params(self._critic_shared, self._target_critic_shared, self.critic_target_tau)
        else:
            soft_update_params(self._critic, self._target_critic, self.critic_target_tau)
            soft_update_params(self._aux_critic, self._target_aux_critic, self.critic_target_tau)
        self._sync_floating_buffers()
        check_finite_tree(metrics, "update_metrics")
        return metrics

    def _replay_reward_metrics(self, rewards, scales):
        # Use the optimizer's actual replay batch, before reward normalization.
        means = torch.stack([rewards[name].detach().float().mean() for name in scales]).to(self.device)
        metrics = {}
        weighted = []
        for (name, scale), mean in zip(scales.items(), means.unbind()):
            metrics[f"Reward/raw/{name}"] = mean
            metrics[f"Reward/weighted/{name}"] = mean * scale
            weighted.append(mean * scale)
        metrics["Reward/weighted/total"] = torch.stack(weighted).sum()
        return metrics

    def _update_context_critics(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Update optional task-specific critics before updating the actor."""
        return {}

    def _sync_floating_buffers(self):
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return
        world_size = torch.distributed.get_world_size()
        modules = (self._obs_normalizer, self._aux_reward_normalizer)
        with torch.no_grad():
            groups = {}
            for module in modules:
                for buffer in module.buffers():
                    if buffer.is_floating_point():
                        groups.setdefault((buffer.device, buffer.dtype), []).append(buffer)
            # These statistics are small; one collective avoids a rank rendezvous
            # for every individual mean/variance. Keep integer counters local.
            for buffers in groups.values():
                flat = _flatten_dense_tensors(buffers)
                torch.distributed.all_reduce(flat)
                flat.div_(world_size)
                torch._foreach_copy_(buffers, _unflatten_dense_tensors(flat, buffers))

    def _b_in_from_parts(self, p):
        return self._privileged_input(p)

    def _full_in_from_parts(self, p):
        return self._full_input(p)

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

    def discriminator_reward_from_logits(self, logits):
        logits = logits.float()
        if self.discriminator_reward_type == "log_odds":
            probability = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
            return probability.log() - (1 - probability).log()
        if self.discriminator_reward_type == "amp":
            return (1.0 - 0.25 * (logits - 1.0).square()).clamp_min(0.0)
        if self.discriminator_reward_type == "softplus":
            return F.softplus(logits)
        return logits

    def update_critic(self, obs, b_obs, action, discount, next_obs, z, next_action):
        cfg = self.cfg
        num_parallel = self.critic_members
        with torch.no_grad():
            with self._autocast_context():
                reward_logits = self._discriminator.compute_logits(obs=b_obs, z=z)
            reward = self.discriminator_reward_from_logits(reward_logits)
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
            reward = self.discriminator_reward_from_logits(reward_logits)
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

    @eager_gradient_diagnostics
    def update_actor(
        self,
        actor_obs,
        full_obs,
        z,
        context_actor_obs=None,
        context_full_obs=None,
        context_z=None,
        heading_valid=None,
        velocity_valid=None,
        depth_gradient_diagnostics=False,
    ):
        cfg = self.cfg
        self.actor_optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            if depth_gradient_diagnostics and self.visual_enabled:
                action_dist = self._actor.forward_with_depth_gradient_diagnostics(
                    self._actor_input(actor_obs), z, cfg.actor_std
                )
            else:
                action_dist = self._actor_forward(actor_obs, z)
            depth_ablation_metrics = {}
            if depth_gradient_diagnostics and self.visual_enabled and isinstance(actor_obs, dict):
                with torch.no_grad():
                    reference_action = action_dist.mean.detach()
                    depth = actor_obs["depth_image"]
                    for name, ablated_depth in (
                        ("zero", torch.zeros_like(depth)),
                        ("temporal_reverse", torch.flip(depth, dims=(1,))),
                    ):
                        ablated_obs = dict(actor_obs)
                        ablated_obs["depth_image"] = ablated_depth
                        ablated = self._actor_forward(ablated_obs, z)
                        depth_ablation_metrics[f"depth_ablation/action_mae_{name}"] = (
                            ablated.mean.float() - reference_action.float()
                        ).abs().mean()
                        if hasattr(action_dist, "velocity_estimate"):
                            depth_ablation_metrics[f"depth_ablation/velocity_mae_{name}"] = (
                                ablated.velocity_estimate.float()
                                - action_dist.velocity_estimate.detach().float()
                            ).abs().mean()
            action = action_dist.sample(clip=cfg.stddev_clip)
        track_depth_gradients = (
            depth_gradient_diagnostics
            and hasattr(action_dist, "actor_depth_latent")
            and hasattr(action_dist, "velocity_depth_latent")
        )
        if track_depth_gradients:
            action_dist.actor_depth_latent.retain_grad()
            action_dist.velocity_depth_latent.retain_grad()
        action = action.float()

        if self.shared_value_net:
            frozen_modules = [self._critic_shared, self._forward_map]
        else:
            frozen_modules = [self._critic, self._aux_critic, self._forward_map]
        for name in ("_heading_critic", "_velocity_critic", "_context_critic_shared"):
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
            context_action = action
            if context_actor_obs is not None:
                with self._autocast_context():
                    # Relabeling changes commands, not images. Share the encoder graph
                    # only within this update; diagnostics retain their original branches.
                    reuse_depth = (
                        self.visual_enabled and not depth_gradient_diagnostics
                        and context_actor_obs["depth_image"] is actor_obs["depth_image"]
                    )
                    if reuse_depth:
                        context_dist = self._actor(
                            self._actor_input(context_actor_obs), context_z, cfg.actor_std,
                            depth_latent=action_dist.depth_latent,
                        )
                    else:
                        context_dist = self._actor_forward(context_actor_obs, context_z)
                    context_action = context_dist.sample(clip=cfg.stddev_clip)
                context_action = context_action.float()
            qs_heading = qs_velocity = None
            if hasattr(self, "_context_critic_shared"):
                qs_heading, qs_velocity = self._context_critic_shared(
                    context_full_obs if context_full_obs is not None else full_obs,
                    context_z if context_z is not None else z,
                    context_action,
                )
            elif hasattr(self, "_heading_critic"):
                qs_heading = self._heading_critic(
                    context_full_obs if context_full_obs is not None else full_obs,
                    context_z if context_z is not None else z,
                    context_action,
                )
            if qs_heading is not None:
                _, _, q_heading = self._targets_uncertainty(qs_heading.float(), cfg.actor_pessimism_penalty)
                if heading_valid is not None:
                    valid = heading_valid.to(device=q_heading.device, dtype=q_heading.dtype).reshape(-1)
                    q_heading = (q_heading.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0)

            q_velocity = torch.zeros_like(q_aux)
            if hasattr(self, "_velocity_critic"):
                qs_velocity = self._velocity_critic(
                    context_full_obs if context_full_obs is not None else full_obs,
                    context_z if context_z is not None else z,
                    context_action,
                )
            if qs_velocity is not None:
                _, _, q_velocity = self._targets_uncertainty(qs_velocity.float(), cfg.actor_pessimism_penalty)
                if velocity_valid is not None:
                    valid = velocity_valid.to(device=q_velocity.device, dtype=q_velocity.dtype).reshape(-1)
                    q_velocity = (q_velocity.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0)

            fs = fs.float()
            qs_fb = (fs * z).sum(-1)
            _, _, q_fb = self._targets_uncertainty(qs_fb, cfg.actor_pessimism_penalty)

            weight = q_fb.abs().mean().detach() if cfg.scale_reg else 1.0
            loss_terms = {
                "discriminator": -q_disc.mean() * cfg.reg_coeff * weight,
                "aux": -q_aux.mean() * cfg.reg_coeff_aux * weight,
                "fb": -q_fb.mean(),
            }
            actor_loss = sum(loss_terms.values())
            if qs_heading is not None:
                loss_terms['heading'] = -q_heading.mean() * float(getattr(cfg, "reg_coeff_heading", 0.0)) * weight
                actor_loss = actor_loss + loss_terms['heading']
            if qs_velocity is not None:
                loss_terms['velocity'] = -q_velocity.mean() * float(getattr(cfg, "reg_coeff_velocity", 0.0)) * weight
                actor_loss = actor_loss + loss_terms['velocity']
            estimator_loss = None
            estimator_metrics = {}
            if (
                hasattr(action_dist, "velocity_estimate")
                and isinstance(actor_obs, dict)
                and "velocity_target_yaw" in actor_obs
            ):
                prediction = action_dist.velocity_estimate.float()
                target = actor_obs["velocity_target_yaw"].detach().float()
                error = prediction - target
                estimator_loss = error.square().mean()
                loss_terms['estimator'] = float(getattr(cfg, "velocity_estimator_loss_coef", 1.0)) * estimator_loss
                actor_loss = actor_loss + loss_terms['estimator']
                with torch.no_grad():
                    absolute = error.abs()
                    estimator_metrics = {"velocity_estimator/loss": estimator_loss.detach()}
                    for axis, value in zip("xyz", absolute.unbind(dim=-1)):
                        estimator_metrics[f"velocity_estimator/mae_{axis}"] = value.mean()
                    for index, axis in enumerate("xyz"):
                        estimator_metrics[f"velocity_estimator/rmse_{axis}"] = error[:, index].square().mean().sqrt()
                        estimator_metrics[f"velocity_estimator/bias_{axis}"] = error[:, index].mean()
                    stopped = torch.linalg.vector_norm(target[:, :2], dim=-1) < 0.1
                    moving_estimate = torch.linalg.vector_norm(prediction[:, :2], dim=-1) > 0.2
                    estimator_metrics["velocity_estimator/false_motion_when_stopped"] = (
                        (stopped & moving_estimate).float().sum() / stopped.sum().clamp_min(1)
                    )
                    estimator_metrics["velocity_estimator/stopped_fraction"] = stopped.float().mean()
                    stopped_count = stopped.sum().clamp_min(1)
                    for index, axis in enumerate("xyz"):
                        estimator_metrics[f"velocity_estimator/stopped_mae_{axis}"] = (
                            absolute[:, index] * stopped
                        ).sum() / stopped_count
            parameter_metrics = {}
            if track_depth_gradients:
                # Include the separate context forward and the actual weighted losses.
                parameter_metrics = parameter_gradient_metrics(loss_terms, self._actor.depth_encoder.parameters())
                # autograd.grad can trigger retained intermediate gradients; keep the
                # legacy latent diagnostics limited to the real backward below.
                action_dist.actor_depth_latent.grad = None
                action_dist.velocity_depth_latent.grad = None
            actor_loss.backward()
        finally:
            for param, enabled in zip(frozen_params, requires_grad):
                param.requires_grad_(enabled)
        depth_gradient_metrics = {}
        if track_depth_gradients:
            actor_grad = action_dist.actor_depth_latent.grad.float().flatten()
            velocity_grad = action_dist.velocity_depth_latent.grad
            if velocity_grad is None:
                velocity_grad = torch.zeros_like(action_dist.velocity_depth_latent)
            velocity_grad = velocity_grad.float().flatten()
            actor_norm = torch.linalg.vector_norm(actor_grad)
            velocity_norm = torch.linalg.vector_norm(velocity_grad)
            norm_product = actor_norm * velocity_norm
            cosine = torch.where(
                norm_product > 0.0,
                torch.dot(actor_grad, velocity_grad) / norm_product,
                torch.zeros_like(norm_product),
            )
            depth_gradient_metrics = {
                "depth_latent_actor_grad_norm": actor_norm.detach(),
                "depth_latent_velocity_grad_norm": velocity_norm.detach(),
                "depth_latent_grad_cosine": cosine.detach(),
                "depth_latent_velocity_actor_grad_ratio": (
                    velocity_norm / actor_norm.clamp_min(torch.finfo(actor_norm.dtype).eps)
                ).detach(),
            }
        self._avg_grad(self._actor.parameters())
        self.actor_optimizer.step()
        metrics = {
            "actor_loss": actor_loss.detach(),
            "Q_discriminator": q_disc.mean().detach(),
            "Q_aux": q_aux.mean().detach(),
            "Q_fb": q_fb.mean().detach(),
        }
        if qs_heading is not None:
            metrics["Q_heading"] = q_heading.mean().detach()
        if qs_velocity is not None:
            metrics["Q_velocity"] = q_velocity.mean().detach()
        if estimator_loss is not None:
            metrics["velocity_estimator_loss"] = estimator_loss.detach()
            metrics.update(estimator_metrics)
        metrics.update(depth_ablation_metrics)
        metrics.update(depth_gradient_metrics)
        metrics.update(parameter_metrics)
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

    def tracking_state_dict(self):
        """Only modules consumed by tracking evaluation, with unchanged weights."""
        return {
            "actor": self._actor.state_dict(),
            "backward_map": self._backward_map.state_dict(),
            "obs_normalizer": self._obs_normalizer.state_dict(),
        }

    def state_dict(self):
        if self.inference_only:
            return self.tracking_state_dict()
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
        if self.inference_only:
            self._obs_normalizer.load_state_dict(state["obs_normalizer"])
            return
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
