"""MBFB H32 dynamics and F targets with mandatory FB sequence-value SVG.

Algorithm reference: Xuewang01/MBFB a25f2b05 (2026-10-29). RP1 observation,
action, history and external-push contracts remain those of the local simulator.
The sequence-conditioned terminal critic uses FB rewards, without pose supervision.
"""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace

from .mbfb_dynamics import MBFBDynamics, PhysicalReplay, rollout_error
from .world_model import UFOStateCodec, quat_apply, yaw_quat
from .world_model_mbfb_diagnostics import sampling_gradient_probe, validate_horizons
from .world_model_mbfb_sequence import FBSequenceValue, fit_sequence_value, sequence_directions, sequence_svg_loss


def all_ranks_true(value: torch.Tensor) -> bool:
    """Reduce a scalar condition so every rank makes the same update decision."""
    flag = torch.as_tensor(value).detach().int().clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag)


class MBFBWorldModel(nn.Module):
    """Coordinate real dynamics fitting, imagined FB targets and actor gradients."""

    version = 14

    def __init__(self, cfg, codec_kwargs, xi_dim, actor_obs_dim, z_dim, device, *, kinematics, dt, velocity_dim=2):
        super().__init__()
        self.cfg_dict = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
        self.cfg = cfg = SimpleNamespace(**self.cfg_dict)
        probe_interval = getattr(cfg, "gradient_probe_interval", 0)
        if type(probe_interval) is not int or probe_interval < 0:
            raise ValueError("world_model.gradient_probe_interval must be a nonnegative integer")
        if type(cfg.model_validation_gate) is not bool:
            raise ValueError("world_model.model_validation_gate must be a boolean")
        if any(type(step) is not int or step < 1 for step in cfg.validation_horizons):
            raise ValueError("world_model.validation_horizons must contain positive integers")
        for key in (
            "window_length",
            "batch_size",
            "update_interval",
            "imagination_interval",
            "imagination_batch_size",
            "horizon",
            "validation_interval",
            "validation_batch_size",
            "replay_steps",
            "forward_horizon",
            "scoring_chunk_size",
            "model_validation_patience",
            "value_fit_epochs",
        ):
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
                raise ValueError(f"world_model.{key} must be a positive integer")
        for key in ("model_warmup_updates", "value_warmup_updates"):
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 0:
                raise ValueError(f"world_model.{key} must be a nonnegative integer")
        for key in ("learning_rate", "actor_grad_ratio", "actor_max_grad_norm", "model_validation_max_ratio"):
            if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
                raise ValueError(f"world_model.{key} must be finite and positive")
        for key in ("actor_weight", "forward_weight", "tracking_coef"):
            if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
                raise ValueError(f"world_model.{key} must be finite and nonnegative")
        if (
            cfg.objective != "mbfb"
            or cfg.ensemble_size != 1
            or cfg.fit_batches != 1
            or cfg.svg_action_mode not in {"mean", "sample"}
            or not 0 <= cfg.actor_weight <= 1
            or cfg.replay_steps < cfg.window_length
            or dt <= 0
            or not math.isfinite(dt)
            or not 0 <= cfg.fb_covariance_ridge
            or not 0 < cfg.value_tau <= 1
        ):
            raise ValueError("Invalid MBFB dynamics, schedule or actor configuration")
        if kinematics is None:
            raise ValueError("MBFB requires verified FK for reconstructed observations")
        self.codec = UFOStateCodec(**codec_kwargs)
        self.kinematics = kinematics
        self.dynamics = MBFBDynamics(xi_dim, cfg.hidden_dims, dt,
                                    use_bf16_amp=getattr(cfg, "use_bf16_amp", True))
        self.xi_dim = xi_dim
        if (
            cfg.tracking_coef != 0
            or type(cfg.sequence_context_length) is not int
            or cfg.sequence_context_length < 1
            or not 0 <= cfg.value_lambda <= 1
        ):
            raise ValueError("FB sequence value requires valid context/returns and tracking_coef=0")
        self.sequence_value = FBSequenceValue(
            59 + xi_dim + actor_obs_dim + 4 + velocity_dim,
            z_dim,
            cfg.sequence_context_length,
            cfg.value_hidden_dims,
            2 if cfg.discriminator_reward else 1,
        )
        self.target_sequence_value = copy.deepcopy(self.sequence_value).requires_grad_(False)
        self.register_buffer("sequence_value_updates", torch.zeros((), dtype=torch.long))
        for name, initial in (("calls", 0), ("updates", 0), ("attempts", 0), ("imagination_updates", 0)):
            self.register_buffer(name, torch.tensor(initial, dtype=torch.long))
        self.register_buffer("model_validation_passes", torch.zeros((), dtype=torch.long))
        self.register_buffer("model_ready", torch.tensor(False))
        self.register_buffer("model_validation_ratio", torch.zeros(()))
        self._fit_succeeded = False
        self._gradient_probe_due = False
        self.register_buffer("validation_stats", torch.zeros(4))
        self.register_buffer("validated", torch.tensor(False))
        self.to(device)
        self.optimizer = torch.optim.Adam(
            self.dynamics.parameters(), lr=cfg.learning_rate, fused=torch.device(device).type == "cuda"
        )
        self.sequence_value_optimizer = torch.optim.Adam(
            self.sequence_value.parameters(), lr=cfg.learning_rate, fused=torch.device(device).type == "cuda"
        )
        self.replay = None
        self._svg = None
        self._value_batch = None
        self._svg_graph = None
        self._fit_graph = None
        self._value_graph = None
        self._forward_graph = None
        self._combine_gradients = combine_actor_gradients
        if getattr(cfg, "compile_gradient_mixing", False) and self.device.type == "cuda":
            self._combine_gradients = torch.compile(combine_actor_gradients, fullgraph=True)

    @property
    def device(self) -> torch.device:
        """Device shared by model parameters and runtime counters."""
        return self.calls.device

    def validate_checkpoint(self, state):
        for key, value in self.state_dict().items():
            saved = state.get(key)
            if saved is None or saved.shape != value.shape:
                if saved is None and (
                    key.startswith("target_sequence_value.")
                    or key in {"model_validation_passes", "model_ready", "model_validation_ratio"}
                ):
                    continue
                raise ValueError(f"MBFB checkpoint shape/key mismatch: {key}")
            if key.startswith("codec.") and not torch.equal(saved.cpu(), value.cpu()):
                raise ValueError(f"MBFB action/observation contract mismatch: {key}")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.validate_checkpoint(state_dict)
        # Older v14 checkpoints stored a quality gate; learned tensors are compatible.
        state_dict = state_dict.copy()
        state_dict.pop("gate_open_at", None)
        state_dict.pop("revalidate_from", None)
        # Old MBFB checkpoints have no target critic or validation-gate buffers.
        for key, value in self.sequence_value.state_dict().items():
            state_dict.setdefault(f"target_sequence_value.{key}", value.detach().clone())
        state_dict.setdefault("model_validation_passes", self.model_validation_passes)
        state_dict.setdefault("model_ready", self.model_ready)
        state_dict.setdefault("model_validation_ratio", self.model_validation_ratio)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.replay = None
        self._fit_succeeded = False
        self._gradient_probe_due = False
        self.validated.fill_(False)
        self._svg = None
        self._value_batch = None
        self._svg_graph = None
        self._fit_graph = None
        self._value_graph = None
        self._forward_graph = None
        return result

    @staticmethod
    def physical_state(physical, position, last_action):
        return torch.cat((position[:, :2], physical, last_action), -1)

    @torch.no_grad()
    def record(self, physical, position, next_position, last_action, action, dones):
        if self.replay is None:
            self.replay = PhysicalReplay(self.cfg.replay_steps, len(action), self.xi_dim, self.device)
        executed = self.codec.env_action(action)
        self.replay.add(
            state=self.physical_state(physical["state"], position, last_action),
            next_state=self.physical_state(physical["next_state"], next_position, executed),
            action=executed,
            context=physical["xi"],
            restart=dones.bool().reshape(-1, 1) | physical["disturbance_mask"].bool(),
        )

    def weight(self) -> float:
        """Enable actor imagination only after model and value warmup gates pass."""
        return float(
            self._fit_succeeded
            and bool(self.model_ready)
            and int(self.sequence_value_updates) >= self.cfg.value_warmup_updates
        )

    def model_ready_for_value(self) -> bool:
        """Allow detached value fitting before its terminal value reaches the actor."""
        return self._fit_succeeded and bool(self.model_ready)

    @torch.profiler.record_function("mbfb/dynamics_fit")
    def fit(self, replay, agent, *, expert_replay=None):
        self.calls.add_(1)
        self._fit_succeeded = False
        self._svg = None
        self._value_batch = None
        cfg = self.cfg
        metrics = {}
        if getattr(cfg, "diagnostics", True):
            zero = self.calls.new_zeros((), dtype=torch.float32)
            metrics.update({
                "wm/budget/imagination_fraction": zero,
                "wm/budget/trajectories_global_per_update": zero,
                "wm/budget/transitions_global_per_update": zero,
            })
        if self.replay is not None and self.replay.size >= cfg.window_length:
            call = int(self.calls)
            if call % cfg.update_interval == 0:
                batch = self.replay.sample(cfg.batch_size, cfg.window_length)
                self.dynamics.observe_increments(batch["state"][:, 0], batch["next_state"][:, 0])
                self.optimizer.zero_grad(set_to_none=True)
                if getattr(cfg, "cuda_graph_dynamics", False) and self.device.type == "cuda":
                    from .world_model_mbfb_graph import DynamicsGraph

                    if self._fit_graph is None or not self._fit_graph.matches(dict(batch=batch)):
                        self._fit_graph = None
                        self._fit_graph = DynamicsGraph(self, batch)
                    errors, loss, gradients = self._fit_graph.replay(dict(batch=batch))
                    for parameter, gradient in zip(self.dynamics.parameters(), gradients):
                        parameter.grad = gradient
                else:
                    errors = rollout_error(self.dynamics, batch)
                    loss = errors.mean()
                    loss.backward()
                agent._avg_grad(self.dynamics.parameters())
                norm = nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
                success = all_ranks_true(torch.isfinite(loss) & torch.isfinite(norm))
                self.attempts.add_(1)
                if success:
                    self.optimizer.step()
                    self.updates.add_(1)
                    self._fit_succeeded = True
                    if (
                        not cfg.model_validation_gate
                        and int(self.updates) >= cfg.model_warmup_updates
                    ):
                        self.model_ready.fill_(True)
                    metrics.update({
                        "wm/loss": loss.detach().clone(),
                        "wm/loss_first": errors[:, 0].mean().detach(),
                        "wm/loss_last": errors[:, -1].mean().detach(),
                    })
                self.optimizer.zero_grad(set_to_none=True)
                metrics["wm/skipped_updates"] = self.calls.new_full((), float(not success), dtype=torch.float32)
            if call % cfg.validation_interval == 0:
                metrics.update(self.validate())
        metrics.update({
            "mb/weight": self.calls.new_full((), self.weight(), dtype=torch.float32),
            "wm/updates": self.updates.float().clone(),
            "wm/attempts": self.attempts.float().clone(),
            "wm/fit_succeeded": self.calls.new_full((), float(self._fit_succeeded), dtype=torch.float32),
        })
        if bool(self.validated):
            for key, value in zip(("val_first", "val_last", "cv_first", "cv_last"), self.validation_stats):
                metrics[f"wm/{key}"] = value.clone()
            metrics["wm/val_ratio"] = self.validation_stats[1] / self.validation_stats[3].clamp_min(1e-12)
        metrics.update({
            "wm/model_ready": self.model_ready.float().clone(),
            "wm/model_validation_passes": self.model_validation_passes.float().clone(),
            "wm/model_validation_ratio": self.model_validation_ratio.clone(),
        })
        return metrics

    @torch.no_grad()
    def validate(self):
        batch = self.replay.sample(self.cfg.validation_batch_size, self.cfg.window_length, validation=True)
        model = rollout_error(self.dynamics, batch).mean(0)
        baseline = rollout_error(self.dynamics, batch, baseline=True).mean(0)
        stats = torch.stack((model[0], model[-1], baseline[0], baseline[-1]))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
            stats.div_(torch.distributed.get_world_size())
        self.validation_stats.copy_(stats)
        self.validated.fill_(True)
        ratio = stats[1] / stats[3].clamp_min(1e-12)
        valid = torch.isfinite(stats).all() & torch.isfinite(ratio) & (stats[3] > 1e-6)
        passed = valid & (ratio <= self.cfg.model_validation_max_ratio)
        self.model_validation_ratio.copy_(torch.where(valid, ratio, torch.zeros_like(ratio)))
        if not self.cfg.model_validation_gate:
            self.model_ready.copy_(self.updates >= self.cfg.model_warmup_updates)
        elif int(self.updates) >= self.cfg.model_warmup_updates and bool(passed):
            self.model_validation_passes.add_(1)
        elif self.cfg.model_validation_gate:
            self.model_validation_passes.zero_()
        if self.cfg.model_validation_gate:
            self.model_ready.copy_(self.model_validation_passes >= self.cfg.model_validation_patience)
        return validate_horizons(self) if getattr(self.cfg, "diagnostics", True) else {}

    def observe(self, state, previous, heading, velocity, valid, *, privileged=True):
        physical = state[:, 2:61]
        obs = self.codec.observation(
            physical,
            state[:, 61:],
            self.codec.advance_history(previous["history_actor"], previous),
            heading,
            velocity,
            valid,
            noisy=True,
        )
        if privileged:
            obs["privileged_state"] = self.kinematics(physical)
        return obs

    @staticmethod
    def commands(state, obs, heading_valid):
        frame = yaw_quat(state[:, 3:7])
        local_heading = torch.cat((1 - obs["heading"][:, :1], obs["heading"][:, 1:2]), -1)
        heading = quat_apply(frame, F.pad(local_heading, (0, 1)))[:, :2]
        local_velocity = obs["velocity_command"][:, :-1]
        dim = local_velocity.shape[-1]
        velocity = quat_apply(frame, F.pad(local_velocity, (0, 3 - dim)))[:, :dim]
        valid = torch.cat((heading_valid.reshape(-1, 1).float(), obs["velocity_command"][:, -1:]), -1)
        return heading, velocity, valid

    def action(self, agent, obs, z, *, svg=False, normalized=None):
        norm = agent._obs_normalizer.normalize_dict(obs) if normalized is None else normalized
        with agent._autocast_context():
            dist = agent._actor_forward(norm, z, imagination=True)
        return (
            dist.mean if svg and self.cfg.svg_action_mode == "mean" else dist.sample(clip=agent.cfg.stddev_clip)
        ).float()

    def post_push_state(self, physical):
        state = physical["next_state"].to(self.device)
        disturbance = physical["disturbance"].to(self.device)
        offset = quat_apply(state[:, 1:5], self.codec.root_com_pos.expand(len(state), -1))
        velocity = torch.cat(
            (disturbance[:, :3] - torch.linalg.cross(disturbance[:, 3:], offset), disturbance[:, 3:]), -1
        )
        velocity = torch.where(physical["disturbance_mask"].to(self.device).bool(), velocity, state[:, 5:11])
        return torch.cat((state[:, :5], velocity, state[:, 11:]), -1)

    @torch.no_grad()
    @torch.profiler.record_function("mbfb/forward_target")
    def forward_target(self, agent, batch, next_obs, z, discount, heading_valid):
        if self.weight() == 0 or self.cfg.forward_weight == 0:
            return None
        state = self.physical_state(
            self.post_push_state(batch["world_model"]),
            batch["world_model"]["next_position"].to(self.device),
            next_obs["last_action"],
        )
        context = batch["world_model"]["xi"].to(self.device)
        commands = self.commands(state, next_obs, heading_valid)
        inputs = dict(state=state, context=context, obs=next_obs, z=z, commands=commands)
        if getattr(self.cfg, "cuda_graph_forward", False) and self.device.type == "cuda":
            from .world_model_mbfb_graph import ForwardGraph

            if self._forward_graph is None or not self._forward_graph.matches(inputs):
                self._forward_graph = None
                self._forward_graph = ForwardGraph(self, agent, inputs)
            future, terminal = self._forward_graph.replay(inputs)
        else:
            future, terminal = self.forward_rollout(agent, **inputs)
        if not all_ranks_true(torch.isfinite(future).all() & torch.isfinite(terminal).all()):
            return None
        return {
            "future": future,
            "terminal": terminal,
            "alive": (discount > 0).detach(),
            "discount": discount * agent.cfg.discount ** (self.cfg.forward_horizon - 1),
            "weight": self.cfg.forward_weight * self.weight(),
            "mbfb": True,
        }

    @torch.no_grad()
    def forward_rollout(self, agent, state, context, obs, z, commands):
        future = torch.zeros_like(z)
        for step in range(1, self.cfg.forward_horizon):
            action = self.action(agent, obs, z)
            state = self.dynamics(state, self.codec.env_action(action), context)
            obs = self.observe(state, obs, *commands)
            with agent._autocast_context():
                embedding = agent._backward_map(agent._b_in(obs)).float()
            future += agent.cfg.discount**step * embedding
        action = self.action(agent, obs, z)
        with agent._autocast_context():
            terminal = agent._target_forward_map(agent._full_in(obs), z, action).float()
        return future, terminal

    @torch.no_grad()
    @torch.profiler.record_function("mbfb/prepare_actor")
    def prepare_actor(self, agent, batch, obs, z, goal, selected, expert, heading_valid):
        self._svg = None
        self._gradient_probe_due = False
        if not self.model_ready_for_value() or self.cfg.actor_weight == 0:
            return
        call = int(self.calls)
        if call % self.cfg.imagination_interval:
            return
        probe_interval = getattr(self.cfg, "gradient_probe_interval", 0)
        self._gradient_probe_due = probe_interval > 0 and call % probe_interval == 0
        n, horizon = min(self.cfg.imagination_batch_size, len(z)), self.cfg.horizon
        physical = batch["world_model"]
        state = self.physical_state(
            physical["state"][:n].to(self.device), physical["position"][:n].to(self.device), obs["last_action"][:n]
        )
        obs = {key: value[:n].detach() for key, value in obs.items()}
        commands = tuple(
            value[:, None].expand(-1, horizon + 1, -1).clone() for value in self.commands(state, obs, heading_valid[:n])
        )
        latent_steps = horizon + self.cfg.sequence_context_length
        latents = z[:n, None].expand(-1, latent_steps, -1).clone()
        rows = selected[:n].reshape(-1).nonzero().flatten() if self.cfg.tracking_z else z.new_empty(0, dtype=torch.long)
        if rows.numel():
            window = agent.cfg.seq_length
            length, count = latent_steps + window - 1, len(rows)
            sample = expert.sample(count * length, seq_length=length)
            observations = {key: value.to(self.device) for key, value in sample["next"]["observation"].items()}
            with agent._autocast_context():
                embedded = agent._backward_map(agent._b_in(observations)).float().reshape(count, length, -1)
            tracking_z = agent.project_z(embedded.unfold(1, window, 1).mean(-1))
            latents[rows] = tracking_z
            from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
                align_world_vector,
                heading_observation,
                velocity_observation,
            )

            def field(key):
                return sample[key].to(self.device).reshape(count, length, -1)[:, : horizon + 1]

            heading = field("heading_forward_xy")
            velocity_key = "reference_velocity_world" if commands[1].shape[-1] == 3 else "reference_velocity_world_xy"
            forward = state.new_tensor([1.0, 0.0, 0.0]).expand(count, -1)
            anchor = quat_apply(yaw_quat(state[rows, 3:7]), forward)[:, None, :2]
            commands[0][rows] = align_world_vector(heading, heading[:, :1], anchor)
            commands[1][rows] = align_world_vector(field(velocity_key), heading[:, :1], anchor)
            commands[2][rows, :, :1] = 1
            commands[2][rows, :, 1:] = field("reference_velocity_valid").to(commands[2].dtype)
            # The new z sequence and its commands start at the same expert transition.
            obs["heading"] = obs["heading"].clone()
            obs["velocity_command"] = obs["velocity_command"].clone()
            obs["heading"][rows] = heading_observation(anchor[:, 0], commands[0][rows, 0], commands[2][rows, 0, :1])
            obs["velocity_command"][rows] = velocity_observation(
                commands[1][rows, 0], anchor[:, 0], commands[2][rows, 0, 1:]
            )
        self._svg = dict(
            state=state,
            context=physical["xi"][:n].to(self.device),
            obs=obs,
            latents=latents,
            goal=goal.detach(),
            commands=commands,
        )
        if getattr(self.cfg, "cuda_graph_imagination", False) and self.device.type == "cuda":
            from .world_model_mbfb_graph import SVGGraph

            inputs = dict(self._svg, directions=latents[:, :horizon])
            if self._svg_graph is None or not self._svg_graph.matches(inputs):
                # Capture before the real actor loss keeps default-stream grad nodes alive.
                inputs["directions"] = sequence_directions(
                    agent, goal, latents, horizon, ridge=self.cfg.fb_covariance_ridge
                )
                parameters = tuple(p for p in agent._actor.parameters() if p.requires_grad)
                self._svg_graph = None
                with torch.enable_grad():
                    self._svg_graph = SVGGraph(self, agent, parameters, inputs)

    def svg_loss(self, agent, **svg):
        return sequence_svg_loss(self, agent, **svg)

    def actor_gradients(self, agent, loss_terms, *, native_gradients=None):
        svg, self._svg = self._svg, None
        parameters = tuple(p for p in agent._actor.parameters() if p.requires_grad)
        if svg is None:
            with torch.profiler.record_function("fbcpr/actor_native_backward"):
                if native_gradients is None:
                    sum(loss_terms.values()).backward()
                else:
                    for parameter, gradient in zip(parameters, native_gradients[0]):
                        parameter.grad = gradient
            return {}

        def gradients(loss, retain_graph=False):
            if not loss.requires_grad:
                return tuple(torch.zeros_like(p) for p in parameters)
            values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
            return tuple(torch.zeros_like(p) if value is None else value for p, value in zip(parameters, values))

        with torch.profiler.record_function("fbcpr/actor_native_backward"):
            main, aux = (
                native_gradients
                if native_gradients is not None
                else native_actor_gradients(loss_terms, parameters, split=True)
            )
        if self._svg_graph is not None and getattr(self.cfg, "cuda_graph_imagination", False):
            # Keep the covariance solve/error check outside capture; B changes every update.
            svg["directions"] = sequence_directions(
                agent, svg["goal"], svg["latents"], self.cfg.horizon, ridge=self.cfg.fb_covariance_ridge
            )
            with torch.profiler.record_function("mbfb/svg_graph"):
                model_loss, metrics, model, self._value_batch = self._svg_graph.replay(svg)
            # Per-update diagnostics must not become persistent entries in graph outputs.
            metrics = dict(metrics)
        else:
            with torch.profiler.record_function("mbfb/svg_forward"):
                model_loss, metrics = self.svg_loss(agent, **svg)
            with torch.profiler.record_function("mbfb/svg_backward"):
                model = gradients(model_loss)
        if self._gradient_probe_due:
            with torch.profiler.record_function("mbfb/sampling_gradient_probe"):
                metrics.update(sampling_gradient_probe(self, agent, svg))
        if getattr(self.cfg, "diagnostics", True):
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            count = len(svg["state"])
            budget = model_loss.new_tensor([
                1,
                count * world_size,
                count * world_size * self.cfg.horizon,
                count,
                self.cfg.horizon,
            ])
            metrics.update(
                dict(
                    zip(
                        (
                            "wm/budget/imagination_fraction",
                            "wm/budget/trajectories_global_per_update",
                            "wm/budget/transitions_global_per_update",
                            "wm/budget/trajectories_per_imagination_local",
                            "wm/budget/horizon",
                        ),
                        budget.unbind(),
                    )
                )
            )
        with torch.profiler.record_function("mbfb/value_fit"):
            metrics.update(fit_sequence_value(self, agent))
        with torch.profiler.record_function("mbfb/gradient_mix"):
            metrics.update(
                mix_actor_gradients(
                    parameters,
                    main,
                    aux,
                    model,
                    self.cfg.actor_weight * self.weight(),
                    self.cfg.actor_grad_ratio,
                    combine=self._combine_gradients,
                )
            )
        self.imagination_updates.add_(1)
        metrics["wm/imagination_updates"] = self.imagination_updates.float().clone()
        metrics["wm/actor_loss"] = torch.nan_to_num(model_loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        return {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in metrics.items()}


def native_actor_gradients(loss_terms, parameters, *, split):
    """Use separate objectives only when imagination reweights the main gradient."""
    if split:
        main = sum(value for name, value in loss_terms.items() if name not in ("aux", "estimator"))
        auxiliary = loss_terms["aux"] + loss_terms.get("estimator", 0)
        losses = (main, auxiliary)
    else:
        losses = (sum(loss_terms.values()),)
    result = []
    for index, loss in enumerate(losses):
        values = (
            torch.autograd.grad(loss, parameters, retain_graph=index < len(losses) - 1, allow_unused=True)
            if loss.requires_grad
            else (None,) * len(parameters)
        )
        if split:
            values = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values))
        result.append(values)
    return tuple(result)


def mix_actor_gradients(parameters, main, aux, model, coefficient, max_ratio, *, combine=None):
    """Assign the combined gradients while leaving the optimizer step to the agent."""
    gradients, metrics = (combine or combine_actor_gradients)(main, aux, model, coefficient, max_ratio)
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient
    return metrics


def combine_actor_gradients(main, aux, model, coefficient, max_ratio):
    """Blend bounded model gradients into the main objective, preserving auxiliary gradients."""
    baseline = tuple(a + b for a, b in zip(main, aux))

    def norm(values):
        return torch.linalg.vector_norm(torch.stack([value.norm() for value in values]))

    base_norm, model_norm = norm(baseline), norm(model)
    finite = torch.isfinite(model_norm)
    scale = (max_ratio * base_norm / model_norm.clamp_min(1e-12)).clamp(max=1.0)
    gradients = tuple(
        (1 - coefficient) * a + b + coefficient * torch.where(finite, scale * m, torch.zeros_like(m))
        for a, b, m in zip(main, aux, model)
    )
    return gradients, {
        "mb/actor_base_grad_norm": base_norm,
        "mb/actor_model_coef": base_norm.new_tensor(coefficient),
        "mb/actor_grad_scale": torch.where(finite, scale, 0.0),
        "mb/actor_model_clipped": (finite & (scale < 1)).float(),
        "mb/actor_model_base_ratio": torch.where(
            finite, coefficient * scale * model_norm / base_norm.clamp_min(1e-12), 0.0
        ),
        "mb/model_base_cosine": torch.nan_to_num(
            sum((a * m).sum() for a, m in zip(baseline, model)) / (base_norm * model_norm).clamp_min(1e-12)
        ),
        "mb/actor_model_grad_norm": torch.where(finite, model_norm, 0.0),
        "mb/actor_model_weighted_grad_norm": torch.where(finite, coefficient * scale * model_norm, 0.0),
        "mb/svg_nonfinite": (~finite).float(),
    }
