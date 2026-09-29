"""Off-policy FB-CPR-Aux runner, following the codebase runner conventions
(constructor/learn/save/load interface mirrors OnPolicyRunner)."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from types import SimpleNamespace

import torch
from torch.utils.tensorboard import SummaryWriter

from ..algorithms.fb_cpr_aux import FbCprAuxAgent
from ..storage.replay_buffer import ReplayBuffer, TrajectoryDictBuffer

HISTORY_KEYS = ("actions", "base_ang_vel", "joint_pos", "joint_vel", "projected_gravity")
UFO_POLICY_SLICES = {
    "base_ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_pos": slice(6, 30),
    "joint_vel": slice(30, 54),
    "actions": slice(54, 78),
}
UFO_HISTORY_LENGTHS = dict.fromkeys(HISTORY_KEYS, 4)


def ufo_obs_dims(critic_dim: int) -> dict[str, int]:
    joint_dim = UFO_POLICY_SLICES["joint_pos"].stop - UFO_POLICY_SLICES["joint_pos"].start
    history_dim = sum(
        (part.stop - part.start) * UFO_HISTORY_LENGTHS[name]
        for name, part in UFO_POLICY_SLICES.items()
    )
    return {
        "state": 2 * joint_dim + 6,
        "privileged_state": critic_dim,
        "last_action": joint_dim,
        "history_actor": history_dim,
    }


def make_ufo_history(num_envs: int, device) -> "HistoryHandler":
    dims = {name: part.stop - part.start for name, part in UFO_POLICY_SLICES.items()}
    return HistoryHandler(num_envs, UFO_HISTORY_LENGTHS, dims, device)


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
    }
    if with_history:
        obs["history_actor"] = history.flat()
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


def _joint_pos_metrics(joint_pos, target_joint_pos):
    stats = {}
    stats["mpjpe_l"] = (torch.norm(joint_pos - target_joint_pos, dim=-1).mean() * 1000).item()
    vel_gt = target_joint_pos[:, 1:] - target_joint_pos[:, :-1]
    vel_pred = joint_pos[:, 1:] - joint_pos[:, :-1]
    stats["vel_dist"] = (torch.norm(vel_pred - vel_gt, dim=-1).mean() * 1000).item()
    accel_gt = target_joint_pos[:, :-2] - 2 * target_joint_pos[:, 1:-1] + target_joint_pos[:, 2:]
    accel_pred = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    stats["accel_dist"] = (torch.norm(accel_pred - accel_gt, dim=-1).mean() * 100).item()
    stats.update(_distance_proximity(joint_pos, target_joint_pos, prefix=""))
    return stats


class FbCprRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        if isinstance(train_cfg, dict):
            train_cfg = SimpleNamespace(**train_cfg)
        self.cfg = train_cfg
        self.action_rescale = float(self.cfg.action_rescale)
        self.eval_interval = int(getattr(self.cfg, "eval_interval_iterations", 0) or 0)
        self.save_interval = int(self.cfg.save_interval)
        if self.eval_interval < 0:
            raise ValueError("eval_interval_iterations must be non-negative.")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive.")
        self.log_dir = log_dir or os.path.join("logs", train_cfg.experiment_name)
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

        obs_dims = ufo_obs_dims(int(sum(mgr.observation_manager.group_obs_dim["critic"])))
        self.obs_dims = obs_dims
        self.agent = FbCprAuxAgent(obs_dims, mgr.action_manager.total_action_dim, train_cfg, device=self.device)
        self._broadcast_agent_state()

        self.history = make_ufo_history(self.num_envs, device)

        self.replay = {"train": ReplayBuffer(
            capacity=self.cfg.buffer_size,
            num_envs=self.num_envs,
            device=device,
            output_key_t=("observation", "action", "z", "aux_rewards"),
            output_key_tp1=("observation", "terminated"),
            end_key="truncated",
        )}
        self._expert_built = False
        self._z = None
        self.tot_iterations = 0
        self.env_iterations = 0
        # WBC-style per-reward-term logging: the UFO aux reward terms are the
        # actual training rewards, so accumulate each scaled term over episodes
        # and report `Episode_Reward/<term>` (mean episodic sum / episode length).
        self._reward_term_names = [str(name) for name in getattr(self.cfg, "aux_rewards", [])]
        self._reward_term_scales = {
            name: float(getattr(self.cfg, "aux_rewards_scaling", {}).get(name, 1.0))
            for name in self._reward_term_names
        }
        self._reward_ep_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self._reward_term_names
        }
        self._ep_extras: list[dict[str, float]] = []
        # Episode-return buffer for the PPO-style `Train/mean_reward` curve.
        self._reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._rewbuffer: deque[float] = deque(maxlen=100)

    def _configure_multi_gpu(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.environ.get("RANK", str(local_rank)))
        self.is_distributed = world_size > 1
        if self.is_distributed:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl")
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(self.device)

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
        averaged = {}
        world_size = torch.distributed.get_world_size()
        for key, value in metrics.items():
            scalar = value.clone()
            torch.distributed.all_reduce(scalar)
            averaged[key] = scalar / world_size
        return averaged

    def _mean_update_metrics(self, updates):
        """Average the minibatch metrics from one agent-update phase."""
        return {key: torch.stack([metrics[key] for metrics in updates]).mean() for key in updates[0]}

    # ------------------------------------------------------------- helpers

    def _build_expert_buffer(self):
        from robolab.tasks.manager_based.ufo.expert_motion_loader import load_expert_trajectories_from_motion_lib

        command = self._mgr_env.command_manager.get_term("motion")
        default_dof_pos = self._mgr_env.scene["robot"].data.default_joint_pos
        episodes, file_names = load_expert_trajectories_from_motion_lib(
            command.motion_lib, self.control_dt, default_dof_pos, device=self.device
        )
        buffer = TrajectoryDictBuffer(episodes=episodes, device=self.device, seq_length=self.cfg.seq_length)
        buffer.file_names = file_names
        self.replay["expert_slicer"] = buffer
        self._expert_built = True

    def _obs_dict(self, obs_buf: dict, with_history=True):
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

    # ---------------------------------------------------------------- learn

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        from robolab.tasks.manager_based.ufo.mdp.rewards import compute_aux_rewards, initialize_aux_rewards

        if not self._expert_built:
            self._build_expert_buffer()

        obs_buf, _ = self.env.reset()
        obs_buf = {name: value.to(self.device) for name, value in obs_buf.items()}
        initialize_aux_rewards(self._mgr_env)
        self.history.reset(torch.arange(self.num_envs, device=self.device))
        self._z = None
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # UFO semantics: agent-update phases are spaced by collected samples
        # (per rank), not by runner iterations.  This is equivalent to the
        # original workspace's EveryNStepsChecker(local_time, update_agent_every).
        update_every_samples = max(1, int(self.cfg.update_agent_every_samples))
        last_update_local_time = self.env_iterations * self.num_envs
        world_size = torch.distributed.get_world_size() if self.is_distributed else 1
        global_envs = self.num_envs * world_size
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
        train_start = time.time()
        interval_collection_time = 0.0
        interval_update_time = 0.0
        interval_eval_time = 0.0
        interval_env_steps = 0
        interval_update_phases = 0
        for _ in range(num_learning_iterations):
            collection_start = time.perf_counter()
            obs_dict = self._obs_dict(obs_buf)
            current_obs_values = self._current_obs_values(obs_buf)
            step_count = self._mgr_env.episode_length_buf
            self._z = self.agent.maybe_update_rollout_context(self._z, step_count, self.replay)

            with torch.no_grad():
                local_time = self.env_iterations * self.num_envs
                if local_time < self.cfg.num_seed_steps:
                    action = torch.empty(
                        self.num_envs, self._mgr_env.action_manager.total_action_dim, device=self.device
                    ).uniform_(-1.0, 1.0)
                else:
                    action = self.agent.act(obs_dict, self._z, mean=False)

            next_obs_buf, _, dones, infos = self.env.step(self._env_action(action))
            next_obs_buf = {name: value.to(self.device) for name, value in next_obs_buf.items()}
            dones = dones.to(self.device)
            aux = {name: value.to(self.device) for name, value in compute_aux_rewards(self._mgr_env).items()}

            if self.is_main_process and self._reward_ep_sums:
                step_reward = None
                for name, total in self._reward_ep_sums.items():
                    term = self._reward_term_scales[name] * aux[name].reshape(-1).detach() * self.control_dt
                    total += term
                    step_reward = term if step_reward is None else step_reward + term
                self._reward_buf += step_reward

            transition = {
                "observation": obs_dict,
                "action": action,
                "z": self._z,
                "aux_rewards": aux,
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
            }
            self.replay["train"].extend(self._time_major(transition))

            self.history.add(current_obs_values)
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            self.history.reset(reset_ids)
            if self.is_main_process and self._reward_ep_sums and len(reset_ids) > 0:
                episode_extras = {}
                for name, total in self._reward_ep_sums.items():
                    episode_extras[f"Episode_Reward/{name}"] = (
                        total[reset_ids].mean() / self.max_episode_length_s
                    ).item()
                    total[reset_ids] = 0.0
                self._rewbuffer.extend(self._reward_buf[reset_ids].tolist())
                self._reward_buf[reset_ids] = 0.0
                self._ep_extras.append(episode_extras)
            obs_buf = next_obs_buf
            terminated = torch.zeros_like(dones, dtype=torch.bool)
            truncated = dones.bool()
            self.env_iterations += 1
            global_step = self.env_iterations * global_envs
            interval_collection_time += time.perf_counter() - collection_start
            interval_env_steps += global_envs

            if self.eval_interval > 0 and self.env_iterations % self.eval_interval == 0:
                eval_start = time.perf_counter()
                pre_eval_steps = self._mgr_env.episode_length_buf.clamp(min=1)
                self._run_tracking_eval(global_step=global_step)
                interval_eval_time += time.perf_counter() - eval_start
                # [RESYNC] the eval ends with a full env.reset() whose fresh
                # observations were discarded; recompute them so the next
                # action is not computed from pre-eval stale observations,
                # and mark the next transition as an episode boundary so
                # replay sequences are never joined across the eval reset.
                obs_buf = {
                    name: value.to(self.device)
                    for name, value in self._mgr_env.observation_manager.compute().items()
                }
                truncated = torch.ones_like(truncated)
                # [FLUSH] the eval resets all envs without emitting dones, so
                # pending episodes can never reach time_out; report them now.
                if self.is_main_process and self._reward_ep_sums:
                    lengths_s = (pre_eval_steps * self.control_dt).clamp(min=self.control_dt)
                    episode_extras = {}
                    for name, total in self._reward_ep_sums.items():
                        episode_extras[f"Episode_Reward/{name}"] = float((total / lengths_s).mean().item())
                        total.zero_()
                    self._rewbuffer.extend(self._reward_buf.tolist())
                    self._reward_buf.zero_()
                    self._ep_extras.append(episode_extras)

            if (
                local_time - last_update_local_time >= update_every_samples
                and local_time > self.cfg.num_seed_steps
                and len(self.replay["train"]) >= 2
            ):
                last_update_local_time = local_time
                should_log = (self.tot_iterations + 1) % 20 == 0
                update_metrics = []
                update_start = time.perf_counter()
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(self.replay)
                    if should_log:
                        update_metrics.append(metrics)
                interval_update_time += time.perf_counter() - update_start
                interval_update_phases += 1
                self.tot_iterations += 1
                if should_log:
                    metrics = self._mean_update_metrics(update_metrics)
                    metrics = self._average_metrics(metrics)
                    training_time = interval_collection_time + interval_update_time
                    fps = interval_env_steps / max(training_time, 1e-9)
                    phases = max(interval_update_phases, 1)
                    if self.is_main_process:
                        self.writer.add_scalar("Perf/total_fps", fps, global_step)
                        self.writer.add_scalar(
                            "Perf/collection_time", interval_collection_time / phases, global_step
                        )
                        self.writer.add_scalar(
                            "Perf/learning_time", interval_update_time / phases, global_step
                        )
                        self.writer.add_scalar("Perf/eval_time", interval_eval_time, global_step)
                        self.writer.add_scalar(
                            "Perf/elapsed_minutes", (time.time() - train_start) / 60.0, global_step
                        )
                        for k, v in metrics.items():
                            self.writer.add_scalar(
                                f"train/{k}", float(v), global_step
                            )
                        if self._rewbuffer:
                            self.writer.add_scalar(
                                "Train/mean_reward", sum(self._rewbuffer) / len(self._rewbuffer), global_step
                            )
                        if self._ep_extras:
                            for key in dict.fromkeys(k for ep in self._ep_extras for k in ep):
                                values = torch.tensor(
                                    [ep[key] for ep in self._ep_extras if key in ep], device=self.device
                                )
                                self.writer.add_scalar(key, values.mean().item(), global_step)
                            self._ep_extras.clear()
                    interval_collection_time = 0.0
                    interval_update_time = 0.0
                    interval_eval_time = 0.0
                    interval_env_steps = 0
                    interval_update_phases = 0

            if self.is_main_process and self.env_iterations % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.tot_iterations}.pt"))

        if self.writer is not None and hasattr(self.writer, "stop"):
            self.writer.stop()


    # -------------------------------------------------------- tracking eval

    @torch.no_grad()
    def _run_tracking_eval(self, global_step: int):
        import csv
        import os as _os

        from robolab.tasks.manager_based.ufo.mdp.motion_observations import compute_humanoid_observations_max
        from robolab.tasks.manager_based.ufo.motion.torch_utils import quat_rotate_inverse

        mgr = self._mgr_env
        command = mgr.command_manager.get_term("motion")
        motion_lib = command.motion_lib
        requested = int(getattr(self.cfg, "eval_num_motions", 0) or motion_lib._num_unique_motions)
        num_motions = min(requested, motion_lib._num_unique_motions)
        motion_ids = torch.arange(num_motions, device=self.device, dtype=torch.long)

        default_dof = mgr.scene["robot"].data.default_joint_pos
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        ref_obs_cache = {}
        # Batched reference build: one motion query, one observation pass and
        # one backward pass for all clips (previously per-clip calls; identical
        # math, far less launch overhead).
        clip_lengths = [
            max(4, int(motion_lib._motion_lengths[m].item() / self.control_dt)) for m in motion_ids.tolist()
        ]
        all_times = torch.cat(
            [torch.arange(length, device=self.device) * self.control_dt for length in clip_lengths]
        )
        all_ids = torch.repeat_interleave(motion_ids, torch.tensor(clip_lengths, device=self.device))
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
        state_all = torch.cat([dof_rel, res["dof_vel"], grav, ang], dim=-1)
        norm = self.agent._obs_normalizer.normalize_dict({"state": state_all, "privileged_state": priv})
        b_in = torch.cat([norm["state"], norm["privileged_state"]], dim=-1)
        ctx_all = self.agent.project_z(self.agent._backward_map(b_in))
        offsets = [0]
        for length in clip_lengths:
            offsets.append(offsets[-1] + length)
        for i, m in enumerate(motion_ids.tolist()):
            s0, s1 = offsets[i], offsets[i + 1]
            length = clip_lengths[i]
            ref_obs_cache[i] = {
                "state": state_all[s0:s1].cpu(),
                "root_pos": res["rg_pos_t"][s0, 0].cpu(),
                "root_quat": res["rg_rot_t"][s0, 0].cpu(),
                "root_lin_vel": res["body_vel_t"][s0, 0].cpu(),
                "root_ang_vel": res["body_ang_vel_t"][s0, 0].cpu(),
                "dof_pos": res["dof_pos"][s0:s1].cpu(),
                "dof_vel": res["dof_vel"][s0].cpu(),
                "ctx": ctx_all[s0 + 1 : s1],
                "length": length,
            }

        term_cfgs = mgr.observation_manager._group_obs_term_cfgs["policy"]
        saved_noise = [t.noise for t in term_cfgs]
        for t in term_cfgs:
            t.noise = None
        # Disable mid-rollout truncation: the original UFO eval_overrides set
        # max_episode_length_s=100000 so clip tracking is uninterrupted.
        saved_ep_len = mgr.cfg.episode_length_s
        mgr.cfg.episode_length_s = 100000.0
        robot = mgr.scene["robot"]

        # Chunked rollout over all clips (original UFO pattern: windows beyond
        # num_envs are evaluated in successive batches).
        all_series: dict[int, dict] = {}
        for chunk_start in range(0, num_motions, self.num_envs):
            chunk = list(range(chunk_start, min(chunk_start + self.num_envs, num_motions)))
            self.env.reset()
            assign = torch.arange(self.num_envs, device=self.device) % len(chunk)
            root_states = robot.data.default_root_state.clone()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            for env_i in range(self.num_envs):
                ref = ref_obs_cache[chunk[int(assign[env_i])]]
                root_states[env_i, 0:3] = ref["root_pos"].to(root_states.device)
                root_states[env_i, 3:7] = ref["root_quat"].to(root_states.device)
                root_states[env_i, 7:10] = ref["root_lin_vel"].to(root_states.device)
                root_states[env_i, 10:13] = ref["root_ang_vel"].to(root_states.device)
                joint_pos[env_i] = ref["dof_pos"][0].to(joint_pos.device)
                joint_vel[env_i] = ref["dof_vel"].to(joint_vel.device)
            robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7])
            robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13])
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            mgr.sim.forward()
            self.history.reset(torch.arange(self.num_envs, device=self.device))
            obs_buf = mgr.observation_manager.compute()

            max_ctx_chunk = max(ref_obs_cache[w]["ctx"].shape[0] for w in chunk)
            ctx_padded = torch.empty(len(chunk), max_ctx_chunk, self.cfg.z_dim, device=self.device)
            for c_i, w in enumerate(chunk):
                ctx = ref_obs_cache[w]["ctx"]
                ctx_padded[c_i, : ctx.shape[0]] = ctx
                ctx_padded[c_i, ctx.shape[0] :] = ctx[-1]

            initial_state = self._obs_dict(obs_buf, with_history=False)["state"]
            initial_joint = robot.data.joint_pos
            len_chunk = len(chunk)
            # GPU-side recording buffers: series are filled on device and
            # transferred to host once per chunk (previously two tiny
            # device->host copies per clip per step).
            buf_state = torch.empty(max_ctx_chunk, len_chunk, 23, device=self.device)
            buf_joint = torch.empty(max_ctx_chunk, len_chunk, initial_joint.shape[-1], device=self.device)
            init_state_cpu = [initial_state[c_i, :23].cpu() for c_i in range(len_chunk)]
            init_joint_cpu = [initial_joint[c_i].cpu() for c_i in range(len_chunk)]
            chunk_series: dict[int, dict] = {}
            for c_i, w in enumerate(chunk):
                ref = ref_obs_cache[w]
                chunk_series[w] = {
                    "target_state": [ref["state"][0, :23]],
                    "target_joint": [ref["dof_pos"][0]],
                }
            for step in range(max_ctx_chunk):
                obs_dict = self._obs_dict(obs_buf)
                current_values = self._current_obs_values(obs_buf)
                action = self.agent.act(obs_dict, ctx_padded[assign, step], mean=True)
                next_obs_buf, _, dones, _ = self.env.step(self._env_action(action))
                self.history.add(current_values)
                done_envs = dones.nonzero(as_tuple=False).flatten()
                if len(done_envs):
                    self.history.reset(done_envs)
                obs_buf = next_obs_buf
                state_now = self._obs_dict(obs_buf, with_history=False)["state"]
                buf_state[step].copy_(state_now[:len_chunk, :23])
                buf_joint[step].copy_(robot.data.joint_pos[:len_chunk])
                for c_i, w in enumerate(chunk):
                    ref = ref_obs_cache[w]
                    ref_t = step + 1
                    if ref_t < ref["length"]:
                        chunk_series[w]["target_state"].append(ref["state"][ref_t, :23])
                        chunk_series[w]["target_joint"].append(ref["dof_pos"][ref_t])
            buf_state_cpu = buf_state.cpu()
            buf_joint_cpu = buf_joint.cpu()
            for c_i, w in enumerate(chunk):
                ref = ref_obs_cache[w]
                n = min(ref["length"], max_ctx_chunk + 1)
                chunk_series[w]["state"] = torch.cat(
                    [init_state_cpu[c_i].unsqueeze(0), buf_state_cpu[: n - 1, c_i]], dim=0
                )
                chunk_series[w]["joint"] = torch.cat(
                    [init_joint_cpu[c_i].unsqueeze(0), buf_joint_cpu[: n - 1, c_i]], dim=0
                )
            all_series.update(chunk_series)

        for t, n in zip(term_cfgs, saved_noise):
            t.noise = n
        mgr.cfg.episode_length_s = saved_ep_len
        self.env.reset()
        self.history.reset(torch.arange(self.num_envs, device=self.device))
        self._z = None

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
            target_state = torch.stack(values["target_state"])
            target_joint = torch.stack(values["target_joint"])
            local = _distance_proximity(state, target_state, prefix="obs_state_")
            local.update(_joint_pos_metrics(joint, target_joint))
            if w % world_size == rank:
                # The original prioritization consumes the joint-space EMD
                # (compute_joint_pos_metrics); obs_state EMD is logged only.
                local.update(_emd(state, target_state, prefix="obs_state_"))
                local.update(_emd(joint, target_joint, prefix=""))
                emd_by_clip[w] = {"emd": local["emd"], "obs_state_emd": local["obs_state_emd"]}
            for key, value in local.items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
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

        # Original-UFO EMD prioritization: gather per-clip joint-space EMD on
        # every rank and translate it into clip sampling probabilities
        # (workspace.py: priorities = clamp(emd, min, max) * scale; exp => 2**p).
        dense_emd = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        dense_obs_emd = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        dense_seen = torch.zeros(num_motions, dtype=torch.float64, device=self.device)
        for w, vals in emd_by_clip.items():
            dense_emd[w] = vals["emd"]
            dense_obs_emd[w] = vals["obs_state_emd"]
            dense_seen[w] = 1.0
        if self.is_distributed:
            torch.distributed.all_reduce(dense_emd)
            torch.distributed.all_reduce(dense_obs_emd)
            torch.distributed.all_reduce(dense_seen)
        if bool(getattr(self.cfg, "prioritization", False)) and bool((dense_seen > 0).all()):
            lo = float(getattr(self.cfg, "prioritization_min_val", 0.5))
            hi = float(getattr(self.cfg, "prioritization_max_val", 2.0))
            scale = float(getattr(self.cfg, "prioritization_scale", 2.0))
            mode = str(getattr(self.cfg, "prioritization_mode", "exp"))
            priority = torch.clamp(dense_emd.float(), min=lo, max=hi) * scale
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
            if motion_lib.update_sampling_weights(weights):
                above = int((weights > weights.mean()).sum().item())
                if self.is_main_process:
                    print(
                        f"[UFO] prioritization: {num_motions} clips, "
                        f"{above} above mean weight, emd [{dense_emd.min():.3f}, {dense_emd.median():.3f}, {dense_emd.max():.3f}]"
                    )
        if not self.is_main_process:
            return metr
        csv_path = _os.path.join(self.log_dir, "tracking_eval.csv")
        exists = _os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["accel_dist", "distance", "emd", "mpjpe_l", "obs_state_distance", "obs_state_emd",
                            "obs_state_proximity", "proximity", "timestep", "vel_dist", "num_motions"])
            w.writerow([f"{metr.get('accel_dist', 0.0):.6f}", f"{metr.get('distance', 0.0):.6f}",
                        f"{metr.get('emd', 0.0):.6f}", f"{metr.get('mpjpe_l', 0.0):.6f}",
                        f"{metr.get('obs_state_distance', 0.0):.6f}", f"{metr.get('obs_state_emd', 0.0):.6f}",
                        f"{metr.get('obs_state_proximity', 0.0):.6f}", f"{metr.get('proximity', 0.0):.6f}",
                        global_step, f"{metr.get('vel_dist', 0.0):.6f}", num_motions])
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
        for k, v in metr.items():
            self.writer.add_scalar(f"eval/{k}", v, global_step)
        return metr

    # ------------------------------------------------------------------ io

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state": self.agent.state_dict(),
                "optimizer_state": self.agent.optimizer_state_dict(),
                "iter": self.tot_iterations,
                "env_iterations": self.env_iterations,
                "action_rescale": self.action_rescale,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        payload = torch.load(path, map_location=self.device)
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
