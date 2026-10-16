"""Off-policy FB-CPR-Aux runner, following the codebase runner conventions
(constructor/learn/save/load interface mirrors OnPolicyRunner)."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from datetime import timedelta
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

from ..algorithms.fb_cpr_aux import FbCprAuxAgent
from ..algorithms.pufo import PufoAgent
from ..storage.replay_buffer import ReplayBuffer, TrajectoryDictBuffer
from ..storage.replay_checkpoint import load_replay, save_replay
from ..utils.finite import check_finite_tree, nonfinite_flag, raise_nonfinite_details
from .tracking_evaluation import (
    PersistentTrackingEvaluator, isolated_evaluator_device, merge_tracking_shards, motion_shard,
)


DEFAULT_RSL_RL_LOG_ROOT = "/data/zym/roboparty_train/logs/rsl_rl"


def default_rsl_rl_log_root() -> str:
    """Return the data-disk root used for UFO/PUFO run artifacts."""
    return os.path.abspath(os.environ.get("ROBO_TRAIN_LOG_ROOT", DEFAULT_RSL_RL_LOG_ROOT))


def _obs_term_scale(term, default: float = 1.0) -> float:
    """Float ObsTerm.scale; None/absent means no scaling (Isaac Lab default)."""
    scale = getattr(term, "scale", None)
    if scale is None:
        return default
    if hasattr(scale, "detach"):
        return float(scale.detach().float().reshape(-1)[0].item())
    if isinstance(scale, (tuple, list)):
        return float(scale[0])
    return float(scale)


HISTORY_KEYS = ("actions", "base_ang_vel", "joint_pos", "joint_vel", "projected_gravity")
UFO_POLICY_SLICES = {
    "base_ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_pos": slice(6, 30),
    "joint_vel": slice(30, 54),
    "actions": slice(54, 78),
}
UFO_HISTORY_LENGTHS = dict.fromkeys(HISTORY_KEYS, 4)


def ufo_obs_dims(
    critic_dim: int,
    history_lengths: dict[str, int] | None = None,
    context_enabled: bool = False,
) -> dict[str, int]:
    joint_dim = UFO_POLICY_SLICES["joint_pos"].stop - UFO_POLICY_SLICES["joint_pos"].start
    history_lengths = history_lengths or UFO_HISTORY_LENGTHS
    history_dim = sum(
        (part.stop - part.start) * history_lengths[name]
        for name, part in UFO_POLICY_SLICES.items()
    )
    dims = {
        "state": 2 * joint_dim + 6,
        "privileged_state": critic_dim,
        "last_action": joint_dim,
        "history_actor": history_dim,
        "velocity_target_yaw": 3,
    }
    if context_enabled:
        dims.update({"heading": 2, "velocity_command": 3})
    return dims


def make_ufo_history(num_envs: int, device, history_lengths=None) -> "HistoryHandler":
    dims = {name: part.stop - part.start for name, part in UFO_POLICY_SLICES.items()}
    return HistoryHandler(num_envs, history_lengths or UFO_HISTORY_LENGTHS, dims, device)


def pufo_obs_dims(
    critic_dim: int, terrain_priv_dim: int, history_lengths: dict[str, int] | None = None
) -> dict[str, int]:
    dims = ufo_obs_dims(critic_dim, history_lengths)
    dims.update({
        "heading": 2,
        "velocity_command": 3,
        "velocity_target_yaw": 3,
        "terrain_priv": terrain_priv_dim,
    })
    return dims


def ufo_current_obs_values(obs_buf: dict) -> dict[str, torch.Tensor]:
    policy = obs_buf["policy"]
    return {name: policy[:, part] for name, part in UFO_POLICY_SLICES.items()}


def ufo_obs_dict(obs_buf: dict, history: "HistoryHandler", with_history: bool = True) -> dict:
    values = ufo_current_obs_values(obs_buf)
    obs = {
        "state": torch.cat(
            [values["joint_pos"], values["joint_vel"], values["projected_gravity"], values["base_ang_vel"]],
            dim=-1,
        ),
        "last_action": values["actions"],
        "privileged_state": obs_buf["critic"],
        "velocity_target_yaw": obs_buf["velocity_target_yaw"],
    }
    if with_history:
        obs["history_actor"] = history.flat()
    return obs



def pufo_obs_dict(obs_buf: dict, history: "HistoryHandler", with_history: bool = True) -> dict:
    """Build the independent H200 PUFO observation contract."""
    obs = ufo_obs_dict(obs_buf, history, with_history)
    depth = obs_buf["visual"]
    if depth.dtype == torch.uint8:
        obs["depth_image"] = depth
    else:
        if not depth.is_floating_point():
            raise TypeError(f"PUFO depth observations must be float32 [0, 1] or uint8 [0, 255], got {depth.dtype}.")
        if depth.dtype != torch.float32:
            depth = depth.float()
        if not torch.isfinite(depth).all():
            raise ValueError("PUFO depth observations contain non-finite values before quantization.")
        if bool((depth < 0.0).any() or (depth > 1.0).any()):
            raise ValueError(
                f"PUFO depth observations must be normalized to [0, 1] before quantization, "
                f"got [{depth.min().item()}, {depth.max().item()}]."
            )
        obs["depth_image"] = depth.mul(255.0).round().to(torch.uint8)
    obs["heading"] = obs_buf["heading"]
    obs["velocity_command"] = obs_buf["velocity_command"]
    obs["velocity_target_yaw"] = obs_buf["velocity_target_yaw"]
    obs["terrain_priv"] = obs_buf["terrain_priv"]
    return obs


class HistoryHandler:
    def __init__(self, num_envs, lengths: dict[str, int], dims: dict[str, int], device):
        self.history = {k: torch.zeros(num_envs, n, dims[k], device=device) for k, n in lengths.items()}

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        for k in self.history:
            self.history[k][env_ids] = 0.0

    def add(self, data: dict[str, torch.Tensor]):
        for k in self.history:
            v = data[k]
            self.history[k] = torch.cat([self.history[k][:, 1:], v.unsqueeze(1)], dim=1)

    def flat(self) -> torch.Tensor:
        return torch.cat([self.history[k].flatten(1) for k in HISTORY_KEYS], dim=-1)


def _joint_state_observation(state, num_dofs):
    """Extract all relative joint positions, excluding velocities and other state."""
    if num_dofs < 1 or state.shape[-1] < num_dofs:
        raise ValueError("Evaluation state does not contain the robot's complete joint positions")
    return state[..., :num_dofs]


def _distance_proximity(next_obs, tracking_target, bound=2.0, margin=2, prefix=""):
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds = dist <= bound
    out_bounds = dist > bound + margin
    prox = in_bounds.float() + ((bound + margin - dist) / margin) * (~in_bounds).float() * (~out_bounds).float()
    return {f"{prefix}proximity": prox.mean().item(), f"{prefix}distance": dist.mean().item()}


def _distance_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def _emd(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Earth Mover's Distance between two equal-size trajectory point sets (UFO solver).

    Uniform weights over equal point counts on both sides reduce optimal
    transport to a linear assignment problem; Jonker-Volgenant (scipy)
    returns the exact optimum and is immune to the degenerate cost matrices
    early policies produce (ot.emd2's network simplex can cycle on those).
    Clip-windowed eval keeps both sets the same length by construction.
    """
    from scipy.optimize import linear_sum_assignment

    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    cost_matrix = _distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    rows, cols = linear_sum_assignment(cost_matrix)
    transport_cost = cost_matrix[rows, cols].sum() / cost_matrix.shape[0]
    return {f"{prefix}emd": float(transport_cost)}


def _joint_pos_metrics(joint_pos, target_joint_pos, dt=0.02):
    if dt <= 0:
        raise ValueError("Metric timestep must be positive")
    stats = {}
    stats["joint_angle_l2_rad"] = torch.norm(joint_pos - target_joint_pos, dim=-1).mean().item()
    vel_gt = torch.diff(target_joint_pos, dim=0) / dt
    vel_pred = torch.diff(joint_pos, dim=0) / dt
    stats["joint_velocity_l2_rad_s"] = torch.norm(vel_pred - vel_gt, dim=-1).mean().item()
    accel_gt = torch.diff(target_joint_pos, n=2, dim=0) / dt**2
    accel_pred = torch.diff(joint_pos, n=2, dim=0) / dt**2
    stats["joint_acceleration_l2_rad_s2"] = torch.norm(accel_pred - accel_gt, dim=-1).mean().item()
    stats.update(_distance_proximity(joint_pos, target_joint_pos, prefix=""))
    return stats


class FbCprRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", *, inference_only=False):
        self.env = env
        if isinstance(train_cfg, dict):
            train_cfg = SimpleNamespace(**train_cfg)
        self.cfg = train_cfg
        self.action_rescale = float(self.cfg.action_rescale)
        self.eval_interval_steps = int(getattr(self.cfg, "eval_interval_steps", 0) or 0)
        self.log_interval_steps = int(getattr(self.cfg, "log_interval_steps", 384_000))
        if self.log_interval_steps <= 0:
            raise ValueError("log_interval_steps must be positive.")
        self._tracking_evaluator = None
        self.checkpoint_interval_steps = int(getattr(self.cfg, "checkpoint_interval_steps", 0) or 0)
        self.checkpoint_replay = bool(getattr(self.cfg, "checkpoint_replay", True))
        self.replay_checkpoint_keep = int(getattr(self.cfg, "replay_checkpoint_keep", 1))
        self.resume_replay_warmup_steps = int(getattr(self.cfg, "resume_replay_warmup_steps", 10_240))
        if self.replay_checkpoint_keep < 1 or self.resume_replay_warmup_steps < 1:
            raise ValueError("Replay retention and resume warmup must be positive.")
        self._replay_warmup_until = 0
        self._resumed = False
        if self.eval_interval_steps < 0:
            raise ValueError("eval_interval_steps must be non-negative.")
        if self.checkpoint_interval_steps <= 0:
            raise ValueError("checkpoint_interval_steps must be positive.")
        if int(getattr(self.cfg, "eval_num_envs", 64)) <= 0:
            raise ValueError("eval_num_envs must be positive.")
        self.eval_reference_chunk = int(getattr(self.cfg, "eval_reference_chunk", 128))
        if self.eval_reference_chunk <= 0:
            raise ValueError("eval_reference_chunk must be positive.")
        eval_timeout = float(getattr(self.cfg, "eval_timeout_s", 1800))
        if not math.isfinite(eval_timeout) or eval_timeout <= 0:
            raise ValueError("eval_timeout_s must be finite and positive.")
        self.log_dir = log_dir or os.path.join(
            default_rsl_rl_log_root(), train_cfg.experiment_name
        )
        self.device = device
        self._configure_multi_gpu()
        self.is_main_process = self.gpu_global_rank == 0
        if self.is_distributed:
            shared_log_dir = [self.log_dir if self.is_main_process else None]
            torch.distributed.broadcast_object_list(shared_log_dir, src=0)
            self.log_dir = shared_log_dir[0]
        if self.is_main_process and not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if self.is_distributed:
            torch.distributed.barrier()
        if self.is_main_process:
            logger_type = str(getattr(self.cfg, "logger", "tensorboard")).lower()
            if logger_type == "wandb":
                # Scalar logging to W&B (plus a local tensorboard copy). Model
                # checkpoints stay on disk and are never uploaded.
                from rsl_rl.utils.wandb_log_writer import WandbLogWriter

                self.writer = WandbLogWriter(
                    log_dir=self.log_dir,
                    project_name=str(getattr(self.cfg, "wandb_project", "rp1.3_ufo")),
                )
            else:
                self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tb"))
        else:
            self.writer = None

        mgr = getattr(env, "unwrapped", env)
        self._mgr_env = mgr
        self.num_envs = mgr.num_envs
        self.control_dt = mgr.step_dt
        self.max_episode_length_s = float(mgr.max_episode_length_s)
        if not math.isfinite(self.max_episode_length_s) or self.max_episode_length_s <= 0.0:
            raise ValueError("max_episode_length_s must be finite and positive.")

        self.visual_enabled = bool(getattr(train_cfg, "visual_enabled", False))
        self.context_enabled = self.visual_enabled or bool(
            getattr(train_cfg, "heading_critic_enabled", False)
            or getattr(train_cfg, "velocity_critic_enabled", False)
        )
        if self.visual_enabled:
            self._mgr_env._pufo_velocity_error_scale_mps = float(
                getattr(train_cfg, "velocity_error_scale_mps", 0.5)
            )
        history_lengths = UFO_HISTORY_LENGTHS
        critic_dim = int(sum(mgr.observation_manager.group_obs_dim["critic"]))
        terrain_priv_dim = int(sum(mgr.observation_manager.group_obs_dim.get("terrain_priv", ())))
        obs_dims = (
            pufo_obs_dims(critic_dim, terrain_priv_dim, history_lengths)
            if self.visual_enabled
            else ufo_obs_dims(critic_dim, context_enabled=self.context_enabled)
        )
        self.obs_dims = obs_dims
        agent_class = PufoAgent if self.context_enabled else FbCprAuxAgent
        self.agent = agent_class(
            obs_dims, mgr.action_manager.total_action_dim, train_cfg,
            device=self.device, inference_only=inference_only,
        )
        self._broadcast_agent_state()

        self.history = make_ufo_history(self.num_envs, device, history_lengths if self.visual_enabled else None)

        # Keep replay storage on host memory for both UFO and PUFO.  This
        # avoids reserving the full multi-million-transition buffer on GPU;
        # PUFO additionally compacts its uint8 depth history below.
        replay_device = "cpu"
        depth_offsets = ()
        if self.visual_enabled:
            depth_params = mgr.cfg.observations.visual.depth_image.params
            depth_offsets = tuple(
                i * int(depth_params["history_skip_frames"])
                for i in range(int(depth_params["num_output_frames"]) - 1, -1, -1)
            )
        self.replay = {"train": ReplayBuffer(
            capacity=self.cfg.buffer_size,
            num_envs=self.num_envs,
            device=replay_device,
            output_key_t=("observation", "action", "z", "aux_rewards"),
            output_key_tp1=(
                "observation",
                "terminated",
                "truncated",
                "transition_terminated",
                "transition_truncated",
            ),
            end_key="episode_end",
            transition_outcomes=True,
            compact_depth_history=self.visual_enabled,
            depth_history_offsets=depth_offsets,
        )}
        self._expert_built = False
        self._z = None
        self._context = None
        self.tot_iterations = 0
        self.env_iterations = 0
        reward_cfg = getattr(getattr(self._mgr_env, "cfg", None), "rewards", None)
        if reward_cfg is None:
            raise RuntimeError("FB-CPR requires the environment to define a rewards config.")
        # Read the configured RewTerm fields so reward names and weights remain
        # part of the environment task definition.
        self._reward_terms = {}
        for name in dir(reward_cfg):
            if name.startswith("_"):
                continue
            term = getattr(reward_cfg, name)
            if callable(getattr(term, "func", None)) and hasattr(term, "weight"):
                weight = float(term.weight)
                if weight != 0.0:
                    self._reward_terms[str(name)] = term
        if not self._reward_terms:
            raise RuntimeError("FB-CPR requires at least one nonzero RewTerm in the environment rewards config.")
        self._reward_term_names = tuple(self._reward_terms)
        self._reward_term_scales = {
            name: float(term.weight) for name, term in self._reward_terms.items()
        }
        # Episode-return buffer for the PPO-style `Train/mean_reward` curve.
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._reward_episode_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._rewbuffer: deque[float] = deque(maxlen=100)

    def _configure_multi_gpu(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.environ.get("RANK", str(local_rank)))
        self.is_distributed = world_size > 1
        self._eval_group = None
        if self.is_distributed:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl")
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(self.device)
            # Evaluation and replay disk I/O must not hold an NCCL collective open.
            self._eval_group = torch.distributed.new_group(
                backend="gloo",
                timeout=timedelta(seconds=max(3600, float(getattr(self.cfg, "eval_timeout_s", 1800)) + 120)),
            )

    def _broadcast_agent_state(self):
        if not self.is_distributed:
            return
        with torch.no_grad():
            for module_state in self.agent.state_dict().values():
                for tensor in module_state.values():
                    torch.distributed.broadcast(tensor, src=0)

    def _average_metrics(self, metrics):
        if not self.is_distributed:
            return metrics
        if not metrics:
            return metrics
        world_size = torch.distributed.get_world_size()
        # One collective per log batch, rather than one per reward/loss term.
        keys = sorted(metrics)
        values = torch.stack([metrics[key].reshape(()) for key in keys])
        torch.distributed.all_reduce(values)
        return dict(zip(keys, (values / world_size).unbind()))

    def _write_metrics(self, metrics, global_step):
        if self.writer is None:
            return
        if hasattr(self.writer, "log_metrics"):
            self.writer.log_metrics(metrics, global_step)
        else:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step)
            self.writer.flush()

    @staticmethod
    def _print_training_metrics(metrics, global_step, update, fps):
        losses = [f"{key}={value:.4f}" for key, value in metrics.items() if "loss" in key.lower()]
        lines = [
            f"[FbCpr] env_steps={global_step:,} update={update:,} fps={fps:,.0f} "
            f"reward={metrics['Reward/weighted/total']:.4f} " + " ".join(losses)
        ]
        for prefix in ("Reward/raw/", "Reward/weighted/"):
            terms = [f"{key[len(prefix):]}={value:.4f}" for key, value in metrics.items()
                     if key.startswith(prefix)]
            lines.append(f"  {prefix[:-1]}: " + " ".join(terms))
        print("\n".join(lines), flush=True)

    def _mean_update_metrics(self, updates):
        """Average the minibatch metrics from one agent-update phase."""
        keys = dict.fromkeys(key for metrics in updates for key in metrics)
        return {
            key: torch.stack([metrics[key] for metrics in updates if key in metrics]).mean()
            for key in keys
        }

    @staticmethod
    def _accumulate_update_metrics(sums, counts, metrics):
        for key, value in metrics.items():
            value = value.detach()
            if key in sums:
                sums[key].add_(value)
            else:
                sums[key] = value.clone()
            counts[key] = counts.get(key, 0) + 1

    # ------------------------------------------------------------- helpers

    def _build_expert_buffer(self):
        from robolab.tasks.manager_based.ufo.expert_motion_loader import load_expert_trajectories_from_motion_lib

        command = self._mgr_env.command_manager.get_term("motion")
        default_dof_pos = self._mgr_env.scene["robot"].data.default_joint_pos
        base_ang_vel_scale = _obs_term_scale(self._mgr_env.cfg.observations.policy.base_ang_vel)
        episodes, file_names = load_expert_trajectories_from_motion_lib(
            command.motion_lib, self.control_dt, default_dof_pos, device=self.device,
            base_ang_vel_scale=base_ang_vel_scale,
        )
        expert_keys = ("observation",)
        if self.context_enabled:
            expert_keys += (
                "heading_forward_xy",
                "reference_velocity_world_xy",
                "reference_velocity_valid",
            )
        buffer = TrajectoryDictBuffer(
            episodes=episodes,
            device=self.device,
            seq_length=self.cfg.seq_length,
            output_key_t=expert_keys,
            output_key_tp1=expert_keys,
        )
        buffer.file_names = file_names
        self.replay["expert_slicer"] = buffer
        self._expert_built = True

    def _obs_dict(self, obs_buf: dict, with_history=True):
        if self.visual_enabled:
            return pufo_obs_dict(obs_buf, self.history, with_history)
        return ufo_obs_dict(obs_buf, self.history, with_history)

    def _current_obs_values(self, obs_buf: dict) -> dict:
        return ufo_current_obs_values(obs_buf)

    @staticmethod
    def _time_major(tree):
        if isinstance(tree, dict):
            return {k: FbCprRunner._time_major(v) for k, v in tree.items()}
        return tree.unsqueeze(0)

    def _env_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(action * self.action_rescale, -self.action_rescale, self.action_rescale)

    def _evaluate_reward_terms(self) -> dict[str, torch.Tensor]:
        """Read pre-reset rewards cached by Isaac Lab (weighted, without dt)."""
        manager = self._mgr_env.reward_manager
        indices = {name: index for index, name in enumerate(manager.active_terms)}
        return {
            name: manager._step_reward[:, indices[name]].detach().clone() / self._reward_term_scales[name]
            for name in self._reward_term_names
        }

    def _root_heading_velocity(self):
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import heading_xy

        robot = self._mgr_env.scene["robot"]
        quat = robot.data.body_quat_w
        velocity = robot.data.body_link_lin_vel_w
        quat = quat if isinstance(quat, torch.Tensor) else quat.torch
        velocity = velocity if isinstance(velocity, torch.Tensor) else velocity.torch
        return heading_xy(quat[:, 0]), velocity[:, 0, :2]

    _check_finite_tree = staticmethod(check_finite_tree)

    # ---------------------------------------------------------------- learn

    def _physics_snapshot(self):
        robot = self._mgr_env.scene["robot"]
        command = self._mgr_env.command_manager.get_term("motion")
        snapshot = {"episode_length": self._mgr_env.episode_length_buf.detach().clone()}
        for name in ("root_state_w", "joint_pos", "joint_vel", "applied_torque", "joint_pos_target"):
            value = getattr(robot.data, name, None)
            if value is not None:
                value = value if torch.is_tensor(value) else value.torch
                snapshot[name] = value.detach().clone()
        for name in ("motion_ids", "motion_times", "reset_motion_ids", "reset_motion_times"):
            value = getattr(command, name, None)
            if value is not None:
                snapshot[name] = value.detach().clone()
        return snapshot

    def _check_finite_rollout(self, value, name, before=None, local_error=None):
        failed = nonfinite_flag(value, device=self.device)
        if local_error is not None:
            failed.fill_(1)
        if self.is_distributed:
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if not failed.item():
            return
        error = local_error
        try:
            raise_nonfinite_details(value, name)
        except FloatingPointError as exc:
            error = str(exc)
        if error is not None:
            rank = getattr(self, "gpu_global_rank", 0)
            print(f"[FbCpr] rank={rank} {error}", flush=True)
            if before is not None and self.log_dir:
                path = os.path.join(self.log_dir, f"nonfinite_rank{rank}_iter{self.env_iterations}.pt")
                try:
                    torch.save({"error": error, "before": before, "after": self._physics_snapshot(),
                                "observations": value,
                                "ground_failure": getattr(self._mgr_env, "_pufo_ground_failure", None)}, path)
                    print(f"[FbCpr] physics failure snapshot: {path}", flush=True)
                except Exception as exc:
                    print(f"[FbCpr] failed to save physics snapshot: {exc}", flush=True)
        raise FloatingPointError(error or f"Another rank detected non-finite state or a physics failure at {name}")

    def _step_training_environment(self, action, before):
        try:
            result = self.env.step(action)
        except Exception as exc:
            # Join the same next-observation check as successful ranks, so a
            # sensor/reset/physics failure cannot strand peers in a collective.
            self._check_finite_rollout({}, "environment_step", before,
                                      local_error=f"{type(exc).__name__}: {exc}")
            raise
        observations, reward, dones, infos = result
        observations = {name: value.to(self.device) for name, value in observations.items()}
        self._check_finite_rollout(observations, f"next_obs@env_iter={self.env_iterations}", before)
        if self.replay["train"].transition_outcomes:
            final_obs = infos.get("final_obs") if isinstance(infos, dict) else None
            error = None
            if bool(dones.any()) and final_obs is None:
                error = "Auto-reset lost final_obs; enable compute_final_obs on the training environment"
            elif final_obs is not None and "_final_obs" in infos:
                if bool((dones.bool() & ~infos["_final_obs"].bool()).any()):
                    error = "final_obs does not cover every reset environment"
            self._check_finite_rollout(final_obs or {}, "final_obs", before, local_error=error)
        return observations, reward, dones, infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        try:
            return self._learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.close_evaluator()

    def close_evaluator(self):
        evaluator = getattr(self, "_tracking_evaluator", None)
        if evaluator is not None:
            evaluator.close()
            self._tracking_evaluator = None

    def _finish_reward_episodes(self, reset_ids):
        if not self.is_main_process or len(reset_ids) == 0:
            return
        if not hasattr(self, "_reward_episode_valid"):
            self._reward_episode_valid = torch.ones_like(self._reward_buf, dtype=torch.bool)
        valid = self._reward_episode_valid[reset_ids]
        self._rewbuffer.extend(self._reward_buf[reset_ids[valid]].tolist())
        self._reward_buf[reset_ids] = 0.0
        self._reward_episode_valid[reset_ids] = True

    def _max_episode_length(self) -> int:
        value = getattr(self._mgr_env, "max_episode_length", None)
        if value is None:
            value = round(self.max_episode_length_s / self.control_dt)
        value = int(value)
        if value <= 0:
            raise ValueError("max_episode_length must be positive")
        return value

    def _randomize_initial_episode_phases(self, enabled: bool) -> torch.Tensor:
        episode_length = self._mgr_env.episode_length_buf
        if not hasattr(self, "_reward_episode_valid"):
            self._reward_episode_valid = torch.ones_like(episode_length, dtype=torch.bool)
        if not enabled:
            self._reward_episode_valid.fill_(True)
            return episode_length
        generator = torch.Generator(device=episode_length.device)
        seed = (
            int(getattr(self.cfg, "seed", 0))
            + 1_000_003 * int(getattr(self, "gpu_global_rank", 0))
            + 97 * int(self.env_iterations)
        )
        generator.manual_seed(seed)
        phases = torch.randint(
            self._max_episode_length(),
            episode_length.shape,
            dtype=episode_length.dtype,
            device=episode_length.device,
            generator=generator,
        )
        episode_length.copy_(phases)
        # A nonzero phase makes the first logged return a partial episode.
        self._reward_episode_valid.copy_(phases == 0)
        if self.is_main_process:
            print(
                "[UFO] Randomized initial episode phases: "
                f"min={int(phases.min())} max={int(phases.max())} "
                f"mean={phases.float().mean().item():.1f}",
                flush=True,
            )
        return phases

    def _profile_agent_update(self, global_step, *, synchronize_ranks=False, **update_kwargs):
        """Capture one real update, including its existing distributed collectives."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(self.device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            # Keep profiler startup and its rank skew outside the update marker.
            if synchronize_ranks and self.is_distributed:
                torch.distributed.barrier()
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
            with torch.profiler.record_function("fbcpr/minibatch_update"):
                metrics = self.agent.update(self.replay, **update_kwargs)
        prefix = Path(self.log_dir) / f"update_profile_rank{self.gpu_global_rank}_{global_step}"
        profile.export_chrome_trace(str(prefix) + ".json")
        summary = profile.key_averages().table(sort_by="self_device_time_total", row_limit=60)
        Path(str(prefix) + ".txt").write_text(summary)
        print(f"[PROFILE] Captured one update at env_steps={global_step:,}: {prefix}.json; "
              "exclude this phase from throughput comparisons", flush=True)
        return metrics

    def _learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # Lightweight runner test doubles from older integrations do not run
        # the full constructor; preserve their UFO-only behavior.
        self.context_enabled = getattr(self, "context_enabled", self.visual_enabled)
        if not self._expert_built:
            self._build_expert_buffer()

        obs_buf, _ = self.env.reset()
        obs_buf = {name: value.to(self.device) for name, value in obs_buf.items()}
        self._check_finite_rollout(obs_buf, "reset_obs")
        self.history.reset(torch.arange(self.num_envs, device=self.device))
        self._randomize_initial_episode_phases(init_at_random_ep_len)
        self._z = None
        self._context = None
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Startup/resume resets the simulator; evaluation in a separate process does not.
        truncated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        initial_eval_time = 0.0
        if self.eval_interval_steps > 0:
            eval_start = time.perf_counter()
            world_size = torch.distributed.get_world_size() if self.is_distributed else 1
            self._run_tracking_eval(self.env_iterations * self.num_envs * world_size)
            initial_eval_time = time.perf_counter() - eval_start

        # UFO semantics: agent-update phases are spaced by collected samples
        # (per rank), not by runner iterations.  This is equivalent to the
        # original workspace's EveryNStepsChecker(local_time, update_agent_every).
        update_every_samples = max(1, int(self.cfg.update_agent_every_samples))
        last_update_local_time = self.env_iterations * self.num_envs
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        global_envs = self.num_envs * world_size
        self._mgr_env._ufo_global_step = self.env_iterations * global_envs
        current_global_step = self.env_iterations * global_envs
        last_log_step = current_global_step
        # Validate gradient diagnostics on the first update, including resumes.
        last_diagnostic_step = current_global_step - self.log_interval_steps
        log_every_update = bool(getattr(self.cfg, "log_every_update", False))
        profile_phase = int(getattr(self.cfg, "profile_update_phase", 0))
        profile_rank = int(getattr(self.cfg, "profile_update_rank", 0))
        if profile_phase < 0 or not -1 <= profile_rank < world_size:
            raise ValueError("Invalid update profiling phase or rank")
        initial_update_count = self.tot_iterations
        next_eval_step = (
            ((current_global_step // self.eval_interval_steps) + 1) * self.eval_interval_steps
            if self.eval_interval_steps > 0
            else None
        )
        next_checkpoint_step = (
            ((current_global_step // self.checkpoint_interval_steps) + 1) * self.checkpoint_interval_steps
        )
        num_env_steps = int(getattr(self.cfg, "num_env_steps", 0) or 0)
        if num_env_steps > 0:
            remaining_steps = max(0, num_env_steps - self.env_iterations * global_envs)
            budget_iterations = math.ceil(remaining_steps / global_envs)
            num_learning_iterations = min(num_learning_iterations, budget_iterations)
            if self.is_main_process:
                print(
                    f"[UFO] global step budget: {num_env_steps:,}; "
                    f"{global_envs:,} steps/iteration; running {num_learning_iterations:,} iterations"
                )
        train_start = time.time() - initial_eval_time
        interval_collection_time = 0.0
        interval_update_time = 0.0
        interval_eval_time = initial_eval_time
        interval_env_steps = 0
        interval_update_phases = 0
        metric_sums, metric_counts = {}, {}
        for _ in range(num_learning_iterations):
            collection_start = time.perf_counter()
            obs_dict = self._obs_dict(obs_buf)
            current_obs_values = self._current_obs_values(obs_buf)
            step_count = self._mgr_env.episode_length_buf.clone()
            current_heading = None
            if self.context_enabled:
                current_heading, _ = self._root_heading_velocity()
                self._context = self.agent.advance_rollout_context(
                    self._context, step_count, current_heading, self.replay
                )
                obs_dict.update(self.agent.context_observations(self._context, current_heading))
                self._z = self._context.z
            else:
                self._z = self.agent.maybe_update_rollout_context(self._z, step_count, self.replay)

            with torch.no_grad():
                local_time = self.env_iterations * self.num_envs
                if local_time < self.cfg.num_seed_steps and not getattr(self, "_resumed", False):
                    # Seed actions are already in the environment's normalized
                    # action contract.  Do not apply the policy's physical
                    # action rescale a second time during random exploration.
                    action = torch.empty(
                        self.num_envs, self._mgr_env.action_manager.total_action_dim, device=self.device
                    ).uniform_(-1.0, 1.0) / self.action_rescale
                else:
                    action = self.agent.act(obs_dict, self._z, mean=False)

            self._mgr_env._ufo_global_step = (self.env_iterations + 1) * global_envs
            env_action = self._env_action(action)
            self._check_finite_rollout({"action": action, "env_action": env_action}, "actions")
            before_physics = self._physics_snapshot()
            before_physics["env_action"] = env_action.detach().clone()
            next_obs_buf, _, dones, step_infos = self._step_training_environment(env_action, before_physics)
            dones = dones.to(self.device)
            action_truncated = dones.bool()
            action_terminated = torch.zeros_like(action_truncated)
            if isinstance(step_infos, dict):
                raw_terminated = step_infos.get("terminated")
                raw_truncated = step_infos.get("truncated")
                if raw_terminated is not None or raw_truncated is not None:
                    if raw_terminated is not None:
                        action_terminated = torch.as_tensor(
                            raw_terminated, device=self.device, dtype=torch.bool
                        ).reshape(-1)
                    if raw_truncated is not None:
                        action_truncated = torch.as_tensor(
                            raw_truncated, device=self.device, dtype=torch.bool
                        ).reshape(-1)
                    else:
                        action_truncated = dones.bool() & ~action_terminated
                else:
                    timeout = step_infos.get("time_outs", step_infos.get("timeouts"))
                    if timeout is not None:
                        timeout = torch.as_tensor(timeout, device=self.device, dtype=torch.bool).reshape(-1)
                        action_truncated = timeout
                        action_terminated = dones.bool() & ~timeout
            fixed_timeout = action_truncated & (
                step_count.reshape(-1) >= self._max_episode_length() - 1
            )
            rollout_metrics = {
                "rollout/reset_fraction": dones.float().mean(),
                "rollout/timeout_reset_fraction": fixed_timeout.float().mean(),
                "rollout/other_truncation_fraction": (action_truncated & ~fixed_timeout).float().mean(),
                "rollout/termination_fraction": action_terminated.float().mean(),
            }
            aux = {name: value.to(self.device) for name, value in self._evaluate_reward_terms().items()}
            if self.context_enabled:
                next_heading, actual_velocity = self._root_heading_velocity()
                if bool(dones.any()):
                    kinematics = step_infos["final_obs"]["transition_kinematics"].to(self.device)
                    next_heading = torch.where(dones.bool()[:, None], kinematics[:, :2], next_heading)
                    actual_velocity = torch.where(dones.bool()[:, None], kinematics[:, 2:], actual_velocity)
                aux.update(
                    self.agent.context_transition(
                        self._context,
                        step_count,
                        current_heading,
                        next_heading,
                        actual_velocity,
                        dones,
                    )
                )
            aux_reward = torch.zeros(self.num_envs, device=self.device)
            for name, scale in self._reward_term_scales.items():
                if name not in aux:
                    raise KeyError(
                        f"Auxiliary reward term {name!r} was not returned by the configured reward function."
                    )
                aux_reward += scale * aux[name].reshape(-1)
            aux["aux_reward"] = aux_reward
            self._check_finite_rollout(aux, "aux")
            if self.is_main_process:
                self._reward_buf += aux_reward.detach() * self.control_dt

            transition_obs = obs_dict
            if self.visual_enabled:
                transition_obs = dict(obs_dict)
                if transition_obs["depth_image"].dtype != torch.uint8:
                    raise TypeError("PUFO transition depth must already satisfy the uint8 deployment contract.")
            eval_due = next_eval_step is not None and (self.env_iterations + 1) * global_envs >= next_eval_step
            transition = {
                "observation": transition_obs,
                "action": action,
                "z": self._z,
                "aux_rewards": aux,
                "terminated": action_terminated.unsqueeze(-1),
                "truncated": action_truncated.unsqueeze(-1),
                "transition_terminated": action_terminated.unsqueeze(-1),
                "transition_truncated": action_truncated.unsqueeze(-1),
                "episode_end": dones.bool().unsqueeze(-1),
            }
            if self.visual_enabled:
                transition["depth_episode_start"] = (terminated | truncated).unsqueeze(-1)
            self.history.add(current_obs_values)
            final_observation = None
            if bool(dones.any()):
                final_observation = self._obs_dict(step_infos["final_obs"])
                if self.context_enabled:
                    final_observation.update({
                        "heading": aux["heading_next"], "velocity_command": aux["velocity_next"],
                    })
            self.replay["train"].extend(
                self._time_major(transition),
                final_observation=self._time_major(final_observation) if final_observation is not None else None,
            )

            reset_ids = dones.nonzero(as_tuple=False).flatten()
            self.history.reset(reset_ids)
            self._finish_reward_episodes(reset_ids)
            obs_buf = next_obs_buf
            terminated = action_terminated
            truncated = action_truncated
            self.env_iterations += 1
            global_step = self.env_iterations * global_envs
            interval_collection_time += time.perf_counter() - collection_start
            interval_env_steps += global_envs

            warmup_until = getattr(self, "_replay_warmup_until", 0)
            if warmup_until and local_time >= warmup_until:
                self._replay_warmup_until = 0
                if self.is_main_process:
                    print(f"[RECOVERY] Replay warmup complete: env_steps={global_step:,}, "
                          f"replay_time_steps={len(self.replay['train'])}; enabling updates", flush=True)

            if (
                local_time - last_update_local_time >= update_every_samples
                and local_time > self.cfg.num_seed_steps
                and local_time >= warmup_until
                # Keep the existing minimum collection window on startup/resume.
                and len(self.replay["train"]) >= 3
            ):
                last_update_local_time = local_time
                should_log = log_every_update or global_step - last_log_step >= self.log_interval_steps
                diagnostics_due = (
                    should_log
                    and global_step - last_diagnostic_step >= self.log_interval_steps
                )
                if diagnostics_due:
                    last_diagnostic_step = global_step
                update_start = time.perf_counter()
                for update_index in range(self.cfg.num_agent_updates):
                    update_kwargs = dict(
                        depth_gradient_diagnostics=(
                            self.visual_enabled and diagnostics_due and update_index == 0
                        ),
                        reward_term_scales=self._reward_term_scales,
                    )
                    if (profile_phase > 0 and update_index == 0
                            and (profile_rank == -1 or self.gpu_global_rank == profile_rank)
                            and self.tot_iterations - initial_update_count + 1 == profile_phase):
                        metrics = self._profile_agent_update(
                            global_step, synchronize_ranks=(profile_rank == -1), **update_kwargs
                        )
                    else:
                        metrics = self.agent.update(self.replay, **update_kwargs)
                    self._accumulate_update_metrics(metric_sums, metric_counts, metrics)
                diagnostic_metrics = {}
                if diagnostics_due:
                    replay_metrics = {
                        key: value.to(self.device)
                        for key, value in self.replay["train"].diagnostics().items()
                    }
                    diagnostic_metrics = self._average_metrics({**rollout_metrics, **replay_metrics})
                interval_update_time += time.perf_counter() - update_start
                interval_update_phases += 1
                self.tot_iterations += 1
                if should_log:
                    last_log_step = global_step
                    metrics = {key: value / metric_counts[key] for key, value in metric_sums.items()}
                    metrics = self._average_metrics(metrics)
                    metric_sums, metric_counts = {}, {}
                    training_time = interval_collection_time + interval_update_time
                    fps = interval_env_steps / max(training_time, 1e-9)
                    phases = max(interval_update_phases, 1)
                    if self.is_main_process:
                        values = torch.stack([value.reshape(()) for value in metrics.values()]).detach().cpu().tolist()
                        metrics = dict(zip(metrics, values))
                        logged = {
                            "Perf/total_fps": fps,
                            "Perf/collection_time": interval_collection_time / phases,
                            "Perf/learning_time": interval_update_time / phases,
                            "Perf/eval_time": interval_eval_time,
                            "Perf/elapsed_minutes": (time.time() - train_start) / 60.0,
                        }
                        logged.update({k if k.startswith("Reward/") else f"train/{k}": v
                                       for k, v in metrics.items()})
                        if diagnostic_metrics:
                            diagnostic_values = torch.stack(
                                [value.reshape(()) for value in diagnostic_metrics.values()]
                            ).detach().cpu().tolist()
                            logged.update({
                                f"diagnostics/{key}": value
                                for key, value in zip(diagnostic_metrics, diagnostic_values)
                            })
                        if self._rewbuffer:
                            logged["Train/mean_reward"] = sum(self._rewbuffer) / len(self._rewbuffer)
                        self._write_metrics(logged, global_step)
                        self._print_training_metrics(metrics, global_step, self.tot_iterations, fps)
                    interval_collection_time = 0.0
                    interval_update_time = 0.0
                    interval_eval_time = 0.0
                    interval_env_steps = 0
                    interval_update_phases = 0

            if eval_due:
                eval_start = time.perf_counter()
                # Isolated evaluators leave the collector and its replay trajectory intact.
                self._run_tracking_eval(global_step=global_step)
                interval_eval_time += time.perf_counter() - eval_start
                while next_eval_step <= global_step:
                    next_eval_step += self.eval_interval_steps

            if global_step >= next_checkpoint_step:
                self.save(os.path.join(self.log_dir, f"model_{global_step}.pt"))
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += self.checkpoint_interval_steps

        if self.writer is not None:
            self.writer.flush()
            if hasattr(self.writer, "stop"):
                self.writer.stop()


    # -------------------------------------------------------- tracking eval

    @torch.no_grad()
    def _run_tracking_eval(self, global_step: int):
        """Evaluate disjoint motion shards in persistent per-rank simulators."""
        motion_lib = self._mgr_env.command_manager.get_term("motion").motion_lib
        keys = [str(key) for key in motion_lib._motion_data_keys]
        requested = int(getattr(self.cfg, "eval_num_motions", 0) or len(keys))
        count = min(requested, len(keys))
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        ids = motion_shard(count, self.gpu_global_rank, world_size)
        try:
            result = (self._run_eval_shard(global_step, ids) if ids else
                      {"motion_keys": keys, "motion_ids": [], "sums": {}, "counts": {}, "clips": {}})
        except Exception as exc:
            self.close_evaluator()
            result = {"error": f"rank {self.gpu_global_rank}: {type(exc).__name__}: {exc}"}
        results = [result]
        if self.is_distributed:
            results = [None] * world_size
            torch.distributed.all_gather_object(results, result, group=self._eval_group)
        merged = merge_tracking_shards(results, keys, count)
        metrics = self._finalize_tracking_eval(global_step, merged["metrics"], merged["clips"], count)
        if self.is_main_process:
            self._write_metrics({f"eval/{key}": value for key, value in metrics.items()}, global_step)
            print(f"[FbCpr] tracking evaluation complete: env_steps={global_step:,} ranks={world_size} clips={count} metrics={metrics}", flush=True)
        return metrics

    def _run_eval_shard(self, global_step, motion_ids):
        from robolab.tasks.manager_based.ufo.pufo_env_cfg import PUFOPlaneSceneCfg

        if torch.device(self.device).type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        if self._tracking_evaluator is None:
            entrypoint = getattr(self.cfg, "tracking_eval_entrypoint", None)
            if not entrypoint or not os.path.isfile(entrypoint):
                raise ValueError("Tracking evaluation requires the train.py entrypoint supplied by the launcher.")
            env_cfg = copy.deepcopy(self._mgr_env.cfg)
            env_cfg.scene.terrain = PUFOPlaneSceneCfg().terrain
            env_cfg.scene.num_envs = min(self.num_envs, int(self.cfg.eval_num_envs), len(motion_ids))
            child_device, child_environment = isolated_evaluator_device(
                self.device, self._eval_process_environment()
            )
            env_cfg.sim.device = child_device
            env_cfg.episode_length_s = 100000.0
            cfg = dict(vars(self.cfg))
            cfg.update(logger="tensorboard", eval_interval_steps=0, torch_compile_mode=None, device=child_device)
            self._tracking_evaluator = PersistentTrackingEvaluator(
                entrypoint=entrypoint, request={"env_cfg": env_cfg, "train_cfg": cfg},
                device=child_device, environment=child_environment,
                log_dir=self.log_dir, rank=self.gpu_global_rank,
                timeout=float(getattr(self.cfg, "eval_timeout_s", 1800)),
            )
        print(f"[FbCpr] tracking evaluation shard: env_steps={global_step:,} rank={self.gpu_global_rank} clips={len(motion_ids)}", flush=True)
        return self._tracking_evaluator.evaluate(self.agent.tracking_state_dict(), global_step, motion_ids)

    def _run_eval_subprocess(self, global_step):
        from robolab.tasks.manager_based.ufo.pufo_env_cfg import PUFOPlaneSceneCfg

        entrypoint = getattr(self.cfg, "tracking_eval_entrypoint", None)
        if not entrypoint or not os.path.isfile(entrypoint):
            raise ValueError("Tracking evaluation requires the train.py entrypoint supplied by the launcher.")
        env_cfg = copy.deepcopy(self._mgr_env.cfg)
        env_cfg.scene.terrain = PUFOPlaneSceneCfg().terrain
        env_cfg.scene.num_envs = min(self.num_envs, int(getattr(self.cfg, "eval_num_envs", 64)))
        child_device, child_env = isolated_evaluator_device(self.device, self._eval_process_environment())
        env_cfg.sim.device = child_device
        env_cfg.episode_length_s = 100000.0
        cfg = dict(vars(self.cfg))
        cfg.update(logger="tensorboard", eval_interval_steps=0, torch_compile_mode=None, device=child_device)
        worker_log = os.path.join(self.log_dir, f"tracking_eval_{global_step}.log")
        print(f"[FbCpr] tracking evaluation starting: env_steps={global_step:,}, plane envs={env_cfg.scene.num_envs}, log={worker_log}", flush=True)
        with tempfile.TemporaryDirectory(prefix="tracking_eval_", dir=self.log_dir) as temp_dir:
            request_path = os.path.join(temp_dir, "request.pt")
            result_path = os.path.join(temp_dir, "result.pt")
            torch.save({
                "env_cfg": env_cfg, "train_cfg": cfg, "model_state": self.agent.tracking_state_dict(),
                "global_step": global_step, "log_dir": self.log_dir, "result_path": result_path,
            }, request_path)
            with open(worker_log, "w") as output:
                completed = subprocess.run(
                    [sys.executable, entrypoint, "--tracking_eval_request", request_path,
                     "--device", child_device, "--visualizer", "none"],
                    env=child_env, stdout=output, stderr=subprocess.STDOUT,
                    timeout=float(getattr(self.cfg, "eval_timeout_s", 1800)), check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"Evaluator exited with code {completed.returncode}; see {worker_log}")
            return torch.load(result_path, map_location="cpu", weights_only=False)

    @staticmethod
    def _eval_process_environment():
        env = {key: value for key, value in os.environ.items() if not (
            key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
                    "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
            or key.startswith("TORCHELASTIC_") or key.startswith("WANDB_")
        )}
        env.update(HEADLESS="1", WANDB_MODE="disabled")
        return env

    @torch.no_grad()
    def _run_tracking_eval_on_current_env(self, global_step: int, motion_ids=None, return_shard=False):
        from robolab.tasks.manager_based.ufo.mdp.motion_observations import compute_humanoid_observations_max
        from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
            align_world_vector,
            heading_observation,
            heading_xy,
            velocity_observation,
        )
        from robolab.tasks.manager_based.ufo.motion.torch_utils import quat_rotate_inverse

        mgr = self._mgr_env
        command = mgr.command_manager.get_term("motion")
        motion_lib = command.motion_lib
        requested = int(getattr(self.cfg, "eval_num_motions", 0) or motion_lib._num_unique_motions)
        num_motions = min(requested, motion_lib._num_unique_motions)
        eval_ids = list(range(num_motions)) if motion_ids is None else list(motion_ids)
        motion_ids = torch.tensor(eval_ids, device=self.device, dtype=torch.long)

        default_dof = mgr.scene["robot"].data.default_joint_pos
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        ref_obs_cache = {}
        # Keep reference construction bounded while sharing the GPU with training.
        reference_chunk = min(self.num_envs, self.eval_reference_chunk)
        for reference_ids in motion_ids.split(reference_chunk):
            clip_lengths = [
                max(4, int(motion_lib._motion_lengths[m].item() / self.control_dt))
                for m in reference_ids.tolist()
            ]
            all_times = torch.cat(
                [torch.arange(length, device=self.device) * self.control_dt for length in clip_lengths]
            )
            all_ids = torch.repeat_interleave(reference_ids, torch.tensor(clip_lengths, device=self.device))
            res = motion_lib.get_motion_state(all_ids, all_times)
            obs_dict = compute_humanoid_observations_max(
                res["rg_pos_t"], res["rg_rot_t"], res["body_vel_t"], res["body_ang_vel_t"],
                local_root_obs=True, root_height_obs=True,
            )
            priv = torch.cat([v for v in obs_dict.values()], dim=-1)
            base_quat = res["rg_rot_t"][:, 0]
            ang = quat_rotate_inverse(base_quat, res["body_ang_vel_t"][:, 0], w_last=True)
            grav = quat_rotate_inverse(base_quat, gravity_vec.repeat(base_quat.shape[0], 1), w_last=True)
            dof_rel = res["dof_pos"] - default_dof.to(res["dof_pos"].device)[0]
            ang = ang * _obs_term_scale(mgr.cfg.observations.policy.base_ang_vel)
            state_all = torch.cat([dof_rel, res["dof_vel"], grav, ang], dim=-1)
            norm = self.agent._obs_normalizer.normalize_dict({"state": state_all, "privileged_state": priv})
            b_in = torch.cat([norm["state"], norm["privileged_state"]], dim=-1)
            ctx_all = self.agent.project_z(self.agent._backward_map(b_in))
            offset = 0
            for motion_id, length in zip(reference_ids.tolist(), clip_lengths):
                s0, s1 = offset, offset + length
                ref_obs_cache[motion_id] = {
                    "state": state_all[s0:s1].cpu(),
                    "root_pos": res["rg_pos_t"][s0, 0].cpu(),
                    "root_quat": res["rg_rot_t"][s0, 0].cpu(),
                    "root_lin_vel": res["body_vel_t"][s0, 0].cpu(),
                    "root_ang_vel": res["body_ang_vel_t"][s0, 0].cpu(),
                    "heading_world": heading_xy(base_quat[s0:s1]).cpu(),
                    "velocity_world": res["body_vel_t"][s0:s1, 0, :2].cpu(),
                    "dof_pos": res["dof_pos"][s0:s1].cpu(),
                    "dof_vel": res["dof_vel"][s0].cpu(),
                    "ctx": ctx_all[s0 + 1 : s1].cpu(),
                    "length": length,
                }
                offset = s1

        # Disable mid-rollout truncation: the original UFO eval_overrides set
        # max_episode_length_s=100000 so clip tracking is uninterrupted.
        saved_ep_len = mgr.cfg.episode_length_s
        mgr.cfg.episode_length_s = 100000.0
        robot = mgr.scene["robot"]

        # Chunked rollout over all clips (original UFO pattern: windows beyond
        # num_envs are evaluated in successive batches).
        all_series: dict[int, dict] = {}
        for chunk_start in range(0, len(eval_ids), self.num_envs):
            chunk = eval_ids[chunk_start : chunk_start + self.num_envs]
            self.env.reset()
            assign = torch.arange(self.num_envs, device=self.device) % len(chunk)
            root_states = robot.data.default_root_state.clone()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            chunk_refs = [ref_obs_cache[w] for w in chunk]
            root_states[:, 0:3] = (
                torch.stack([ref["root_pos"] for ref in chunk_refs]).to(root_states.device)[assign]
                + mgr.scene.env_origins
            )
            root_states[:, 3:7] = torch.stack(
                [ref["root_quat"] for ref in chunk_refs]
            ).to(root_states.device)[assign]
            root_states[:, 7:10] = torch.stack(
                [ref["root_lin_vel"] for ref in chunk_refs]
            ).to(root_states.device)[assign]
            root_states[:, 10:13] = torch.stack(
                [ref["root_ang_vel"] for ref in chunk_refs]
            ).to(root_states.device)[assign]
            joint_pos[:] = torch.stack([ref["dof_pos"][0] for ref in chunk_refs]).to(joint_pos.device)[assign]
            joint_vel[:] = torch.stack([ref["dof_vel"] for ref in chunk_refs]).to(joint_vel.device)[assign]
            robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7])
            robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13])
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            mgr.sim.forward()
            mgr.scene.update(dt=0.0)
            # Reapply reset contact clearance after writing the reference pose.
            # This is required by both flat and generated Newton scenes.  The
            # camera refresh is conditional because UFO has no camera sensor.
            from robolab.tasks.manager_based.ufo.mdp.pufo_reset import correct_terrain_penetration
            ids = torch.arange(self.num_envs, device=self.device)
            correct_terrain_penetration(mgr, ids, root_states)
            mgr.sim.forward()
            mgr.scene.update(dt=0.0)
            sensors = getattr(mgr.scene, "sensors", {})
            if "camera" in sensors:
                from robolab.tasks.manager_based.ufo.mdp.pufo_reset import refresh_reset_camera
                refresh_reset_camera(mgr, ids)
            self.history.reset(torch.arange(self.num_envs, device=self.device))
            obs_buf = mgr.observation_manager.compute()
            initial_heading, _ = self._root_heading_velocity()

            max_ctx_chunk = max(ref_obs_cache[w]["ctx"].shape[0] for w in chunk)
            ctx_padded = torch.empty(len(chunk), max_ctx_chunk, self.cfg.z_dim, device=self.device)
            heading_padded = torch.empty(self.num_envs, max_ctx_chunk, 2, device=self.device)
            velocity_padded = torch.empty(self.num_envs, max_ctx_chunk, 2, device=self.device)
            for c_i, w in enumerate(chunk):
                ctx = ref_obs_cache[w]["ctx"]
                ctx_padded[c_i, : ctx.shape[0]] = ctx
                ctx_padded[c_i, ctx.shape[0] :] = ctx[-1]
            for c_i, w in enumerate(chunk):
                ref = ref_obs_cache[w]
                length = ref["ctx"].shape[0]
                heading = ref["heading_world"][:length].to(self.device)
                velocity = ref["velocity_world"][:length].to(self.device)
                env_ids = (assign == c_i).nonzero(as_tuple=False).flatten()
                reference_anchor = heading[0].expand(len(env_ids), -1).unsqueeze(1)
                rollout_anchor = initial_heading[env_ids].unsqueeze(1)
                heading = align_world_vector(
                    heading.unsqueeze(0).expand(len(env_ids), -1, -1),
                    reference_anchor,
                    rollout_anchor,
                )
                velocity = align_world_vector(
                    velocity.unsqueeze(0).expand(len(env_ids), -1, -1),
                    reference_anchor,
                    rollout_anchor,
                )
                heading_padded[env_ids, :length] = heading
                velocity_padded[env_ids, :length] = velocity
                if length < max_ctx_chunk:
                    heading_padded[env_ids, length:] = heading[:, -1:]
                    velocity_padded[env_ids, length:] = velocity[:, -1:]

            initial_state = self._obs_dict(obs_buf, with_history=False)["state"]
            initial_joint = robot.data.joint_pos
            num_dofs = initial_joint.shape[-1]
            len_chunk = len(chunk)
            # GPU-side recording buffers: series are filled on device and
            # transferred to host once per chunk (previously two tiny
            # device->host copies per clip per step).
            buf_state = torch.empty(max_ctx_chunk, len_chunk, num_dofs, device=self.device)
            buf_joint = torch.empty(max_ctx_chunk, len_chunk, initial_joint.shape[-1], device=self.device)
            measure_velocity = bool(getattr(self.agent._actor, "estimate_velocity", False))
            buf_velocity_error = (torch.empty(max_ctx_chunk, len_chunk, 3, device=self.device)
                                  if measure_velocity else None)
            init_state_cpu = _joint_state_observation(initial_state[:len_chunk], num_dofs).cpu()
            init_joint_cpu = initial_joint[:len_chunk].cpu()
            chunk_series: dict[int, dict] = {}
            for c_i, w in enumerate(chunk):
                chunk_series[w] = {}
            for step in range(max_ctx_chunk):
                obs_dict = self._obs_dict(obs_buf)
                current_values = self._current_obs_values(obs_buf)
                if self.context_enabled:
                    current_heading, _ = self._root_heading_velocity()
                    target_heading = heading_padded[:, step]
                    target_velocity = velocity_padded[:, step]
                    valid = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
                    obs_dict["heading"] = heading_observation(current_heading, target_heading, valid)
                    obs_dict["velocity_command"] = velocity_observation(
                        target_velocity, current_heading, valid
                    )
                if measure_velocity:
                    dist = self.agent._actor_forward(self.agent._norm_parts(obs_dict), ctx_padded[assign, step])
                    action = dist.mean
                    buf_velocity_error[step].copy_(
                        dist.velocity_estimate[:len_chunk] - obs_dict["velocity_target_yaw"][:len_chunk]
                    )
                else:
                    action = self.agent.act(obs_dict, ctx_padded[assign, step], mean=True)
                self._check_finite_tree(action, "eval_action")
                next_obs_buf, _, dones, _ = self.env.step(self._env_action(action))
                self._check_finite_tree(dict(next_obs_buf), "eval_obs")
                self.history.add(current_values)
                done_envs = dones.nonzero(as_tuple=False).flatten()
                if len(done_envs):
                    self.history.reset(done_envs)
                obs_buf = next_obs_buf
                state_now = self._obs_dict(obs_buf, with_history=False)["state"]
                buf_state[step].copy_(_joint_state_observation(state_now[:len_chunk], num_dofs))
                buf_joint[step].copy_(robot.data.joint_pos[:len_chunk])
            buf_state_cpu = buf_state.cpu()
            buf_joint_cpu = buf_joint.cpu()
            velocity_error_cpu = buf_velocity_error.cpu() if measure_velocity else None
            for c_i, w in enumerate(chunk):
                ref = ref_obs_cache[w]
                n = min(ref["length"], max_ctx_chunk + 1)
                chunk_series[w]["target_state"] = _joint_state_observation(ref["state"][:n], num_dofs)
                chunk_series[w]["target_joint"] = ref["dof_pos"][:n]
                chunk_series[w]["state"] = torch.cat(
                    [init_state_cpu[c_i].unsqueeze(0), buf_state_cpu[: n - 1, c_i]], dim=0
                )
                chunk_series[w]["joint"] = torch.cat(
                    [init_joint_cpu[c_i].unsqueeze(0), buf_joint_cpu[: n - 1, c_i]], dim=0
                )
                if measure_velocity:
                    chunk_series[w]["velocity_error"] = velocity_error_cpu[: n - 1, c_i]
            all_series.update(chunk_series)

        mgr.cfg.episode_length_s = saved_ep_len
        self.env.reset()
        self.history.reset(torch.arange(self.num_envs, device=self.device))
        self._z = None
        self._context = None

        rank = torch.distributed.get_rank() if self.is_distributed else 0
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        emd_by_clip: dict[int, dict[str, float]] = {}
        for w, values in all_series.items():
            if len(values["joint"]) < 3:
                continue
            state = values["state"]
            joint = values["joint"]
            target_state = values["target_state"]
            target_joint = values["target_joint"]
            local = _distance_proximity(state, target_state, prefix="obs_state_")
            local.update(_joint_pos_metrics(joint, target_joint, self.control_dt))
            if "velocity_error" in values:
                error = values["velocity_error"].float()
                for index, axis in enumerate("xyz"):
                    local[f"velocity_estimator_mae_{axis}_m_s"] = float(error[:, index].abs().mean())
                    local[f"velocity_estimator_rmse_{axis}_m_s"] = float(error[:, index].square().mean().sqrt())
                    local[f"velocity_estimator_bias_{axis}_m_s"] = float(error[:, index].mean())
            if w % world_size == rank:
                # The original prioritization consumes the joint-space EMD
                # (compute_joint_pos_metrics); obs_state EMD is logged only.
                local.update(_emd(state, target_state, prefix="obs_state_"))
                local.update(_emd(joint, target_joint, prefix=""))
                emd_by_clip[w] = {"emd": local["emd"], "obs_state_emd": local["obs_state_emd"]}
            for key, value in local.items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        if return_shard:
            return {"sums": sums, "counts": counts, "clips": emd_by_clip,
                    "motion_ids": eval_ids,
                    "motion_keys": [str(key) for key in motion_lib._motion_data_keys]}
        if not counts:
            return {}
        for missing in ("obs_state_emd", "emd"):
            if missing not in counts:
                sums[missing] = 0.0
                counts[missing] = 0
        if not self.is_distributed:
            metr = {key: sums[key] / counts[key] for key in sums if counts[key] > 0}
        else:
            metr = {}
            for key in sorted(sums):
                total = torch.tensor([sums[key], float(counts[key])], dtype=torch.float64, device=self.device)
                torch.distributed.all_reduce(total)
                denom = total[1].item()
                metr[key] = (total[0].item() / denom) if denom > 0 else 0.0

        return self._finalize_tracking_eval(global_step, metr, emd_by_clip, num_motions)

    def _finalize_tracking_eval(self, global_step, metr, emd_by_clip, num_motions):
        import csv
        import os as _os

        motion_lib = self._mgr_env.command_manager.get_term("motion").motion_lib
        dense_emd = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        dense_obs_emd = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        dense_seen = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        for w, vals in emd_by_clip.items():
            dense_emd[w] = vals["emd"]
            dense_obs_emd[w] = vals["obs_state_emd"]
            dense_seen[w] = 1.0
        if (bool(getattr(self.cfg, "prioritization", False))
                and num_motions == motion_lib._num_unique_motions and bool((dense_seen > 0).all())):
            lo = float(getattr(self.cfg, "prioritization_min_val", 0.5))
            hi = float(getattr(self.cfg, "prioritization_max_val", 2.0))
            scale = float(getattr(self.cfg, "prioritization_scale", 2.0))
            mode = str(getattr(self.cfg, "prioritization_mode", "exp"))
            finite_emd = torch.nan_to_num(dense_emd.float(), nan=hi, posinf=hi, neginf=hi)
            priority = torch.clamp(finite_emd, min=lo, max=hi) * scale
            if mode == "exp":
                weights = 2.0**priority
            elif mode == "lin":
                weights = priority.clone()
            elif mode == "bin":
                bins = torch.floor(priority)
                weights = torch.zeros_like(priority)
                for b in range(int(bins.min().item()), int(bins.max().item()) + 1):
                    mask = bins == b
                    n = int(mask.sum())
                    if n > 0:
                        weights[mask] = 1.0 / n
            else:
                raise ValueError(f"Unsupported prioritization mode {mode}")
            if not motion_lib.update_sampling_weights(weights):
                raise RuntimeError("Motion library rejected finite tracking priorities")
            expert = self.replay.get("expert_slicer")
            if expert is None or not hasattr(expert, "update_priorities"):
                raise RuntimeError("Prioritization requires an expert buffer with priority support")
            motion_to_weight = {motion_id: weights[motion_id] for motion_id in range(num_motions)}
            try:
                expert_weights = torch.stack(
                    [motion_to_weight[int(motion_id)] for motion_id in expert.motion_ids]
                )
            except KeyError as exc:
                raise ValueError(
                    f"Expert buffer motion ID {exc.args[0]} was not covered by full tracking evaluation"
                ) from exc
            expert.update_priorities(expert_weights)
            above = int((weights > weights.mean()).sum().item())
            if self.is_main_process:
                print(
                    f"[UFO] prioritization: {num_motions} clips, "
                    f"{above} above mean weight, emd "
                    f"[{finite_emd.min():.3f}, {finite_emd.median():.3f}, {finite_emd.max():.3f}]"
                )
        if not self.is_main_process:
            return metr
        csv_path = _os.path.join(self.log_dir, "tracking_eval_v2.csv")
        exists = _os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["joint_acceleration_l2_rad_s2", "distance", "emd", "joint_angle_l2_rad", "obs_state_distance", "obs_state_emd",
                            "obs_state_proximity", "proximity", "timestep", "joint_velocity_l2_rad_s", "num_motions"])
            w.writerow([f"{metr['joint_acceleration_l2_rad_s2']:.6f}", f"{metr.get('distance', 0.0):.6f}",
                        f"{metr.get('emd', 0.0):.6f}", f"{metr['joint_angle_l2_rad']:.6f}",
                        f"{metr.get('obs_state_distance', 0.0):.6f}", f"{metr.get('obs_state_emd', 0.0):.6f}",
                        f"{metr.get('obs_state_proximity', 0.0):.6f}", f"{metr.get('proximity', 0.0):.6f}",
                        global_step, f"{metr['joint_velocity_l2_rad_s']:.6f}", num_motions])
        # Per-clip tracking detail (mirrors the original humanoidverse_tracking_eval.csv).
        clips_path = _os.path.join(self.log_dir, "tracking_eval_clips.csv")
        clips_exists = _os.path.exists(clips_path)
        keys_np = motion_lib._motion_data_keys
        with open(clips_path, "a", newline="") as f:
            w = csv.writer(f)
            if not clips_exists:
                w.writerow(["motion_name", "motion_id", "emd", "obs_state_emd", "timestep"])
            for w_i in range(num_motions):
                if dense_seen[w_i].item() <= 0:
                    continue
                name = str(keys_np[w_i]) if w_i < len(keys_np) else str(w_i)
                w.writerow([name, w_i, f"{dense_emd[w_i].item():.6f}",
                            f"{dense_obs_emd[w_i].item():.6f}", global_step])
        return metr

    # ------------------------------------------------------------------ io

    def save(self, path, infos=None):
        """Collective on all ranks; publish the model only after replay shards succeed."""
        path = Path(path)
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        snapshots = self._checkpoint_gather(uuid.uuid4().hex if self.is_main_process else None)
        replay_name = f"replay_{self.env_iterations * self.num_envs * world_size}_{snapshots[0]}"
        replay_dir = path.parent / replay_name
        error = None
        try:
            if self.checkpoint_replay:
                save_replay(self.replay["train"], replay_dir / f"rank_{self.gpu_global_rank}",
                            rank=self.gpu_global_rank, world_size=world_size, env_iterations=self.env_iterations)
        except Exception as exc:
            error = f"rank {self.gpu_global_rank}: {type(exc).__name__}: {exc}"
        self._checkpoint_check_errors(error)
        error = None
        temporary = path.with_name(f".{path.name}.{snapshots[0]}.tmp")
        try:
            if self.is_main_process:
                if self.checkpoint_replay:
                    (replay_dir / "manifest.json").write_text(json.dumps({
                        "kind": "fbcpr_replay", "world_size": world_size,
                        "env_iterations": self.env_iterations,
                    }))
                torch.save({
                    "model_state": self.agent.state_dict(),
                    "optimizer_state": self.agent.optimizer_state_dict(),
                    "iter": self.tot_iterations,
                    "env_iterations": self.env_iterations,
                    "action_rescale": self.action_rescale,
                    "world_size": world_size,
                    "num_envs": self.num_envs,
                    "reward_term_scales": getattr(self, "_reward_term_scales", {}),
                    "replay_checkpoint": replay_name if self.checkpoint_replay else None,
                }, temporary)
                os.replace(temporary, path)
        except Exception as exc:
            error = f"model publication: {type(exc).__name__}: {exc}"
        finally:
            if self.is_main_process and temporary.exists():
                temporary.unlink()
        self._checkpoint_check_errors(error)
        if self.is_main_process:
            print(f"[CHECKPOINT] Saved {path}; replay={'all ranks' if self.checkpoint_replay else 'disabled'}", flush=True)
            if self.checkpoint_replay:
                self._prune_replay_checkpoints(path.parent, replay_name)
        # Retention I/O must finish before other ranks enter gradient collectives.
        self._checkpoint_gather(None)

    def _checkpoint_gather(self, value):
        if not self.is_distributed:
            return [value]
        results = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(results, value, group=self._eval_group)
        return results

    def _checkpoint_check_errors(self, error):
        errors = [value for value in self._checkpoint_gather(error) if value]
        if errors:
            raise RuntimeError("Checkpoint operation failed: " + "; ".join(errors))

    def _prune_replay_checkpoints(self, directory, current):
        try:
            snapshots = []
            for candidate in directory.glob("replay_*"):
                manifest = candidate / "manifest.json"
                if candidate.is_symlink() or not manifest.is_file():
                    continue
                metadata = json.loads(manifest.read_text())
                if metadata.get("kind") == "fbcpr_replay":
                    snapshots.append((candidate.stat().st_mtime_ns, candidate))
            snapshots.sort(reverse=True)
            for _, candidate in snapshots[self.replay_checkpoint_keep:]:
                if candidate.name != current:
                    shutil.rmtree(candidate)
                    print(f"[CHECKPOINT] Removed older replay snapshot {candidate}; model retained", flush=True)
        except Exception as exc:
            print(f"[CHECKPOINT] WARNING: replay retention failed: {exc}", flush=True)

    def load(self, path, load_optimizer=True):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        for key, expected in (("world_size", world_size), ("num_envs", self.num_envs)):
            if load_optimizer and key in payload and payload[key] != expected:
                raise ValueError(f"Checkpoint {key}={payload[key]} does not match {expected}")
        saved_rescale = payload.get("action_rescale")
        if saved_rescale is not None and float(saved_rescale) != self.action_rescale:
            raise ValueError(
                f"Checkpoint action_rescale={saved_rescale} does not match runner action_rescale={self.action_rescale}."
            )
        self.agent.load_state_dict(payload["model_state"])
        if load_optimizer and "optimizer_state" in payload:
            self.agent.load_optimizer_state_dict(payload["optimizer_state"])
        self.tot_iterations = payload.get("iter", 0)
        if "env_iterations" in payload:
            self.env_iterations = payload["env_iterations"]
        elif self.tot_iterations > 0:
            update_every_samples = max(1, int(self.cfg.update_agent_every_samples))
            update_every_iters = max(1, math.ceil(update_every_samples / self.num_envs))
            first_update = self.cfg.num_seed_steps // self.num_envs + 2
            first_update = math.ceil(first_update / update_every_iters) * update_every_iters
            self.env_iterations = first_update + (self.tot_iterations - 1) * update_every_iters
        if not load_optimizer:
            return
        self._resumed = True
        replay_name = payload.get("replay_checkpoint")
        if replay_name and (not isinstance(replay_name, str) or os.path.basename(replay_name) != replay_name):
            raise ValueError("Invalid replay checkpoint directory")
        replay_dir = Path(path).parent / replay_name if replay_name else None
        available = bool(replay_dir and (replay_dir / "manifest.json").is_file() and all(
            (replay_dir / f"rank_{rank}" / "manifest.json").is_file() for rank in range(world_size)
        ))
        # Stored aux_reward cannot be recomputed after changing the reward contract.
        if payload.get("reward_term_scales", {}) != getattr(self, "_reward_term_scales", {}):
            available = False
            if self.is_main_process:
                print("[RECOVERY] Reward contract changed or is unversioned; retain model/optimizer "
                      "and collect fresh replay", flush=True)
        compatibility_error = None
        try:
            if available and self.replay["train"].transition_outcomes:
                shard_manifest = json.loads((replay_dir / f"rank_{self.gpu_global_rank}" / "manifest.json").read_text())
                if not shard_manifest.get("transition_outcomes", False):
                    available = False
                    if self.is_main_process:
                        print("[RECOVERY] Legacy replay lacks pre-reset outcomes; retain model/optimizer and collect fresh replay", flush=True)
        except Exception as exc:
            compatibility_error = f"rank {self.gpu_global_rank} replay compatibility: {type(exc).__name__}: {exc}"
        self._checkpoint_check_errors(compatibility_error)
        available = all(self._checkpoint_gather(available))
        error = None
        if available:
            try:
                manifest = json.loads((replay_dir / "manifest.json").read_text())
                if manifest != {"kind": "fbcpr_replay", "world_size": world_size,
                                "env_iterations": self.env_iterations}:
                    raise ValueError("Replay manifest does not match the model checkpoint")
                load_replay(self.replay["train"], replay_dir / f"rank_{self.gpu_global_rank}",
                            rank=self.gpu_global_rank, world_size=world_size, env_iterations=self.env_iterations)
                self.replay["train"].close_trajectory()
            except Exception as exc:
                error = f"rank {self.gpu_global_rank} replay load: {type(exc).__name__}: {exc}"
        self._checkpoint_check_errors(error)
        sizes = self._checkpoint_gather(len(self.replay["train"]))
        if len(set(sizes)) != 1:
            raise ValueError(f"Replay sizes differ across ranks: {sizes}")
        if not sizes[0]:
            self._replay_warmup_until = (self.env_iterations * self.num_envs
                                         + max(self.resume_replay_warmup_steps, 3 * self.num_envs))
            if self.is_main_process:
                print(f"[RECOVERY] No complete nonempty replay snapshot; collect with loaded policy, "
                      f"no optimizer updates until local_steps={self._replay_warmup_until:,} "
                      f"(warmup={self._replay_warmup_until - self.env_iterations * self.num_envs:,} per rank)", flush=True)
        else:
            self._replay_warmup_until = 0
            if self.is_main_process:
                print(f"[RECOVERY] Restored replay: {sizes[0]} time steps per rank; no warmup", flush=True)

    def get_inference_policy(self, device=None):
        agent = self.agent
        device = device or self.device

        def policy(obs_dict, z):
            return agent.act({k: v.to(device) for k, v in obs_dict.items()}, z.to(device), mean=True)

        return policy

    def add_git_repo_to_log(self, repo_file_path):
        pass

    def export_policy_to_jit(self, path, filename="policy.pt"):
        dummy_obs = {k: torch.zeros(1, d, device=self.device) for k, d in self.obs_dims.items()}
        dummy_z = torch.zeros(1, self.cfg.z_dim, device=self.device)
        module = torch.jit.trace(
            lambda o, z: self.agent.act(o, z, mean=True),
            (dummy_obs, dummy_z),
        )
        os.makedirs(path, exist_ok=True)
        torch.jit.save(module, os.path.join(path, filename))

    def export_policy_to_onnx(self, path, encoder_filename="depth_encoder.onnx",
                              actor_filename="actor.onnx",
                              metadata_filename="pufo_onnx.json",
                              context_filename="backward_encoder.onnx"):
        """Export PUFO depth, context, and actor graphs for deployment."""
        if not self.visual_enabled:
            raise RuntimeError("Separate ONNX export is only defined for visual PUFO.")
        return self.agent.export_visual_onnx(
            path,
            encoder_filename=encoder_filename,
            actor_filename=actor_filename,
            context_filename=context_filename,
            metadata_filename=metadata_filename,
        )
