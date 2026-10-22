# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import argparse
import os
import sys
from pathlib import Path

from robolab.utils.kitless_runtime import configure_kitless_runtime, parse_isaaclab_cli, run_resolved_task

_REPO_ROOT = configure_kitless_runtime()

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--data_root", type=str, default=None,
    help="Motion-data root containing motions; defaults to ROBO_TRAIN_DATA_ROOT or robo_newton/robolab/data/.",
)
parser.add_argument(
    "--motion_dir", "--motion_manifest", dest="motion_dir", type=str, default=None,
    help="UFO/PUFO NPZ directory; overrides UFO_MOTION_DIR (--motion_manifest is a legacy alias).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--resume_learning_rate", type=float, default=None,
    help="Override PPO learning rate after loading optimizer state; the configured schedule still applies.",
)
parser.add_argument(
    "--reset_amp_discriminators",
    action="store_true",
    default=False,
    help="Resume PPOAMP actor/critic state while reinitializing AMP discriminators and their optimizers.",
)
parser.add_argument(
    "--tracking_eval_request", type=str, default=None,
    help="Internal evaluation request from FbCprRunner; runs in an isolated simulator process.",
)
parser.add_argument(
    "--skip_tracking_eval", action="store_true", default=False,
    help="Skip tracking evaluation for a smoke test; the training default remains unchanged.",
)
parser.add_argument(
    "--diagnostic_steps", type=int, default=0,
    help="Run zero-action physics checks without a runner or replay buffer.",
)
parser.add_argument(
    "--diagnostic_action_scale", type=float, default=0.0,
    help="Sample diagnostic actions uniformly within +/- this environment-action amplitude.",
)
parser.add_argument(
    "--profile_physics_steps", type=int, default=0,
    help="Profile Newton memory and contact capacity without constructing a training agent.",
)
parser.add_argument("--profile_physics_trace", type=str, default=None,
                    help="Optional output path for fixed-seed physics comparison tensors.")
parser.add_argument("--profile_physics_nan", type=str, default=None,
                    help="Opt-in graph-compatible solver-stage checks; save first invalid state to this path.")
parser.add_argument("--profile_policy_checkpoint", type=str, default=None,
                    help="Use a trained PUFO actor during bounded physics profiling, without optimizer updates.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes. Auto-enabled when launched via torchrun (WORLD_SIZE > 1).",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
args_cli = parse_isaaclab_cli(parser)

if args_cli.data_root is not None:
    os.environ["ROBO_TRAIN_DATA_ROOT"] = str(Path(args_cli.data_root).expanduser().resolve())
if args_cli.motion_dir is not None:
    os.environ["UFO_MOTION_DIR"] = str(Path(args_cli.motion_dir).expanduser().resolve())

# auto-enable distributed training when launched with torchrun / torch.distributed.run
if int(os.getenv("WORLD_SIZE", "1")) > 1:
    args_cli.distributed = True

from packaging import version

from isaaclab_rl.rsl_rl import check_rsl_rl_version

installed_version = check_rsl_rl_version()

import logging
import re
from datetime import datetime

import torch
import gymnasium as gym

from rsl_rl.runners import DistillationRunner, FbCprRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab.app import launch_simulation
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

import robolab.tasks  # noqa: F401

DEFAULT_RSL_RL_LOG_ROOT = str(_REPO_ROOT / "logs" / "rsl_rl")


def _rsl_rl_log_root(experiment_name: str) -> str:
    """Use the data disk for run logs and checkpoints by default."""
    root = os.environ.get("ROBO_TRAIN_LOG_ROOT", DEFAULT_RSL_RL_LOG_ROOT)
    return os.path.abspath(os.path.join(root, experiment_name))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def get_resume_checkpoint_path(log_path: str, run_dir: str, checkpoint: str) -> str:
    """Find the latest checkpoint, skipping empty run directories.

    Runs are ordered by last-modified time (newest first), matching Isaac Lab ``get_checkpoint_path(sort_alpha=False)``.
    """
    runs = [
        os.path.join(log_path, run.name)
        for run in os.scandir(log_path)
        if run.is_dir() and re.match(run_dir, run.name)
    ]
    if not runs:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")
    runs = sorted(runs, key=os.path.getmtime)
    for run_path in reversed(runs):
        model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
        if model_checkpoints:
            model_checkpoints.sort(key=lambda m: f"{m:0>15}")
            return os.path.join(run_path, model_checkpoints[-1])
    raise ValueError(f"No checkpoints in the directory: '{log_path}' match '{checkpoint}'.")


def _init_process_group_if_needed() -> None:
    """Initialize torch distributed once when launched via torchrun."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1 or torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))


def _broadcast_log_run_name(log_run_name: str, global_rank: int) -> str:
    """Broadcast the log run name from rank 0 to every distributed worker."""
    _init_process_group_if_needed()
    payload = [log_run_name if global_rank == 0 else ""]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _broadcast_resume_path(resume_path: str | None, global_rank: int) -> str:
    """Broadcast the resume checkpoint path from rank 0 to every distributed worker."""
    _init_process_group_if_needed()
    payload = [resume_path if global_rank == 0 else ""]
    torch.distributed.broadcast_object_list(payload, src=0)
    path = payload[0]
    if not path:
        raise ValueError("Rank 0 failed to resolve resume checkpoint path.")
    return path


def _resolve_log_dir(
    log_root_path: str,
    *,
    resume: bool,
    load_run: str,
    load_checkpoint: str,
    run_name: str | None,
    distributed: bool,
    global_rank: int,
) -> tuple[str, str | None]:
    """Resolve the log directory and optional checkpoint path for this run."""
    os.makedirs(log_root_path, exist_ok=True)

    # Resolve checkpoint from prior runs before creating a new log directory (Isaac Lab behavior).
    resume_path = None
    if resume:
        if global_rank == 0:
            resume_path = get_resume_checkpoint_path(log_root_path, load_run, load_checkpoint)
        if distributed:
            resume_path = _broadcast_resume_path(resume_path, global_rank)
        print(f"[INFO] Resuming weights from checkpoint: {resume_path}")

    log_run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_run_name += f"_{run_name}"
    if distributed:
        log_run_name = _broadcast_log_run_name(log_run_name, global_rank)
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_run_name}")
    log_dir = os.path.join(log_root_path, log_run_name)
    os.makedirs(log_dir, exist_ok=True)
    if resume:
        print(f"[INFO] Logging new run to directory: {log_dir}")
    return log_dir, resume_path


def _diagnose_physics(env, steps):
    """Check the simulator before any learner or replay can affect the result."""
    robot = env.scene["robot"]
    from robolab.utils.tensors import as_torch_tensor

    def check_state(label):
        values = {
            "root": as_torch_tensor(robot.data.root_state_w),
            "joint_pos": as_torch_tensor(robot.data.joint_pos),
            "joint_vel": as_torch_tensor(robot.data.joint_vel),
            "effort": as_torch_tensor(robot.data.applied_torque),
        }
        for name, value in values.items():
            finite = torch.isfinite(value).reshape(env.num_envs, -1).all(-1)
            if not finite.all():
                ids = (~finite).nonzero().flatten()[:8]
                print(f"[PHYSICS] {label} invalid={name} ids={ids.tolist()}", flush=True)
                for key, tensor in values.items():
                    print(f"[PHYSICS] {key}={tensor[ids].tolist()}", flush=True)
                raise FloatingPointError(f"{label}: non-finite {name}")
        return values

    env.reset()
    def check_reset_frame(env_ids):
        command = env.command_manager.get_term("motion")
        reference = command.get_motion_state(command.reset_motion_ids[env_ids], command.reset_motion_times[env_ids])
        position = as_torch_tensor(robot.data.joint_pos)[env_ids]
        expected = reference["dof_pos"].clone()
        if hasattr(env, "_pufo_reset_lying"):
            lying = env._pufo_reset_lying[env_ids]
            expected[lying] = as_torch_tensor(robot.data.default_joint_pos)[env_ids][lying]
        error = (position - expected).abs().amax().item()
        if error > 1.0e-5:
            raise AssertionError(f"Reset pose does not match sampled motion: max_joint_error={error}")
        if command._reset_prepared[env_ids].any():
            raise AssertionError("Command reset did not consume the prepared motion")
        print(f"[PHYSICS] reset frame PASS envs={len(env_ids)} max_joint_error={error:.6g}", flush=True)
        if hasattr(env, "_pufo_reset_lift"):
            import mujoco_warp as mjwarp
            import warp as wp
            from isaaclab_newton.physics.newton_manager import NewtonManager

            solver = NewtonManager._solver
            data = solver.mjw_data
            # Diagnostic-only verification of the final pose, not another reset correction.
            solver._update_mjc_data(data, solver.model, NewtonManager.backend.state_0)
            mjwarp.fwd_position(solver.mjw_model, data)
            count = int(wp.to_torch(data.nacon)[0])
            residual = 0.0
            if count:
                pairs = wp.to_torch(data.contact.geom)[:count].long()
                worlds = wp.to_torch(data.contact.worldid)[:count].long()
                distance = wp.to_torch(data.contact.dist)[:count]
                dynamic = torch.as_tensor(solver.mj_model.geom_bodyid > 0, device=env.device)
                valid = ((pairs >= 0) & (pairs < len(dynamic))).all(-1)
                safe = pairs.clamp(0, len(dynamic) - 1)
                valid &= dynamic[safe[:, 0]] ^ dynamic[safe[:, 1]]
                valid &= torch.isin(worlds, env_ids)
                residual = (-distance[valid]).clamp_min(0).max().item() if valid.any() else 0.0
            lift = env._pufo_reset_lift[env_ids].max().item()
            print(f"[PHYSICS] reset clearance: max_lift={lift:.6g} max_residual={residual:.6g}", flush=True)
            if residual > .005:
                raise AssertionError(f"Reset terrain penetration exceeds 5 mm: {residual} m")

    check_reset_frame(torch.arange(env.num_envs, device=env.device))
    # A round-trip tests the actual backend convention, not only its docstring.
    ids = torch.arange(min(2, env.num_envs), device=env.device)
    saved_pose = as_torch_tensor(robot.data.root_state_w)[ids, :7].clone()
    probe_pose = saved_pose.clone()
    probe_pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    robot.write_root_pose_to_sim_index(root_pose=probe_pose, env_ids=ids)
    env.sim.forward()
    readback = as_torch_tensor(robot.data.root_quat_w)[ids].clone()
    torch.testing.assert_close(readback, probe_pose[:, 3:7])
    from isaaclab.utils.math import quat_apply
    up = torch.zeros((len(ids), 3), device=env.device)
    up[:, 2] = 1.0
    torch.testing.assert_close(quat_apply(readback, up), up)
    robot.write_root_pose_to_sim_index(root_pose=saved_pose, env_ids=ids)
    env.sim.forward()
    print("[PHYSICS] XYZW identity round-trip and up-vector PASS", flush=True)
    check_state("reset")
    from isaaclab_newton.physics.newton_manager import NewtonManager
    import warp as wp
    solver = NewtonManager._solver
    model = solver.mj_model
    print(f"[PHYSICS] reset_roots={as_torch_tensor(robot.data.root_state_w)[:8, :7].tolist()}", flush=True)
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            torch.testing.assert_close(
                wp.to_torch(solver.mjw_model.geom_pos)[:, geom_id],
                torch.tensor(model.geom_pos[geom_id], device=env.device, dtype=torch.float32)
                .expand(solver.mjw_model.geom_pos.shape[0], -1),
            )
            print(f"[PHYSICS] static_geom={model.geom(geom_id).name} type={model.geom_type[geom_id]} "
                  f"pos={model.geom_pos[geom_id].tolist()} size={model.geom_size[geom_id].tolist()} "
                  f"gpu_pos={wp.to_torch(solver.mjw_model.geom_pos)[0, geom_id].tolist()}", flush=True)
    print(f"[PHYSICS] hfield_size={model.hfield_size.tolist()}", flush=True)
    original_step = env.sim.step
    original_update = env.scene.update
    substep = 0

    def checked_step(*args, **kwargs):
        nonlocal substep
        before = check_state(f"before_substep={substep}")
        peak_speed = before["joint_vel"].abs().max().item()
        if peak_speed > 100.0:
            flat_index = before["joint_vel"].abs().argmax().item()
            row, joint = divmod(flat_index, robot.num_joints)
            print(
                f"[PHYSICS] high_speed substep={substep} env={row} joint={robot.joint_names[joint]} "
                f"speed={peak_speed} root={before['root'][row].tolist()} "
                f"joints={before['joint_pos'][row].tolist()}", flush=True,
            )
            data = solver.mjw_data
            count = int(wp.to_torch(data.nacon)[0])
            contact = data.contact
            mask = wp.to_torch(contact.worldid)[:count] == row
            indices = mask.nonzero().flatten()
            if indices.numel():
                distances = wp.to_torch(contact.dist)[indices]
                for index in indices[distances.argsort()[:4]].tolist():
                    geom_ids = wp.to_torch(contact.geom)[index].tolist()
                    print(f"[PHYSICS] high_speed_contact distance={float(wp.to_torch(contact.dist)[index])} "
                          f"geoms={[model.geom(g).name for g in geom_ids]}", flush=True)
        if substep < 24:
            print(
                f"[PHYSICS] substep={substep} "
                f"max_joint_vel={before['joint_vel'].abs().max().item():.6g} "
                f"max_effort={before['effort'].abs().max().item():.6g}",
                flush=True,
            )
        result = original_step(*args, **kwargs)
        substep += 1
        return result

    def checked_update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if substep == 1:
            from isaaclab_newton.physics.newton_manager import NewtonManager
            import warp as wp
            solver = NewtonManager._solver
            model, data = solver.mj_model, solver.mjw_data
            print(f"[PHYSICS] armature={model.dof_armature.tolist()}", flush=True)
            count = int(wp.to_torch(data.nacon)[0])
            if count:
                contact = data.contact
                distances = wp.to_torch(contact.dist)[:count]
                for index in distances.argsort()[:8].tolist():
                    geom_ids = wp.to_torch(contact.geom)[index].tolist()
                    bodies = [model.body(int(model.geom_bodyid[g])).name for g in geom_ids]
                    geoms = [model.geom(g).name for g in geom_ids]
                    print(
                        f"[PHYSICS] contact world={int(wp.to_torch(contact.worldid)[index])} "
                        f"distance={float(distances[index]):.6g} bodies={bodies} geoms={geoms}", flush=True,
                    )
        check_state(f"after_substep={substep - 1}")
        return result

    env.sim.step = checked_step
    env.scene.update = checked_update
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    try:
        for step in range(steps):
            if args_cli.diagnostic_action_scale > 0.0:
                actions.uniform_(-args_cli.diagnostic_action_scale, args_cli.diagnostic_action_scale)
            obs, reward, terminated, truncated, _ = env.step(actions)
            reset_ids = (terminated | truncated).nonzero().flatten()
            if len(reset_ids):
                check_reset_frame(reset_ids)
            if not torch.isfinite(reward).all():
                raise FloatingPointError(f"step={step}: non-finite reward")
            for name, value in obs.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"step={step}: non-finite observation {name}")
            if step < 10 or (step + 1) % 100 == 0:
                print(f"[PHYSICS] step={step + 1} finite; resets={(terminated | truncated).sum().item()}", flush=True)
    finally:
        env.sim.step = original_step
        env.scene.update = original_update
    print(f"[PHYSICS] PASS steps={steps} substeps={substep} envs={env.num_envs}", flush=True)


def _run_tracking_eval_worker(request_path):
    # This payload is generated locally by the training runner, not an external checkpoint.
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    cfg = request["train_cfg"]
    with launch_simulation(request["env_cfg"], args_cli):
        _run_tracking_eval(request, cfg)


def _run_tracking_eval(request, cfg):
    env = gym.make("RP1-PUFO-EVAL", cfg=request["env_cfg"])
    runner = None
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=cfg.get("clip_actions"))
        runner = FbCprRunner(
            env, cfg, log_dir=request["log_dir"], device=request["env_cfg"].sim.device,
            inference_only=True,
        )
        if request.get("persistent", False):
            import json

            print(f"[FbCpr] persistent evaluator ready: pid={os.getpid()} envs={env.num_envs}", flush=True)
            for line in sys.stdin:
                message = json.loads(line)
                round_request = torch.load(message["request_path"], map_location="cpu", weights_only=False)
                runner.agent.load_state_dict(round_request.pop("model_state"))
                step = round_request["global_step"]
                try:
                    result = runner._run_tracking_eval_on_current_env(
                        step, motion_ids=round_request["motion_ids"], return_shard=True,
                    )
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                result["global_step"] = step
                result["worker_pid"] = os.getpid()
                result_path = round_request["result_path"]
                # Keep the evaluator alive without retaining unused inference allocations.
                if torch.device(runner.device).type == "cuda":
                    with torch.cuda.device(runner.device):
                        torch.cuda.empty_cache()
                torch.save(result, result_path + ".tmp")
                os.replace(result_path + ".tmp", result_path)
                print(f"[FbCpr] evaluation worker round complete: step={step} pid={os.getpid()}", flush=True)
            return
        runner.agent.load_state_dict(request["model_state"])
        del request["model_state"]
        metrics = runner._run_tracking_eval_on_current_env(request["global_step"])
        if not metrics or not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError(f"Invalid tracking evaluation metrics: {metrics}")
        motion_lib = env.unwrapped.command_manager.get_term("motion").motion_lib
        probabilities = motion_lib._sampling_prob
        torch.save({
            "metrics": metrics,
            "motion_keys": [str(key) for key in motion_lib._motion_data_keys],
            "sampling_weights": probabilities.cpu() if probabilities is not None else None,
        }, request["result_path"])
        print(f"[FbCpr] evaluation worker PASS: {metrics}", flush=True)
    finally:
        if runner is not None:
            runner.writer.close()
        env.close()


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.skip_tracking_eval:
        agent_cfg.eval_interval_steps = 0
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # rsl-rl 3.x 的 PPO/PPOAMP 不支持 Isaac Lab 2.3 为 5.0 预留的字段
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration; launch_simulation already bound each rank's device
    if args_cli.distributed:
        agent_cfg.device = env_cfg.sim.device
        seed = agent_cfg.seed + int(os.getenv("LOCAL_RANK", "0"))
        env_cfg.seed = seed
        agent_cfg.seed = seed

    global_rank = int(os.getenv("RANK", "0")) if args_cli.distributed else 0

    # specify directory for logging experiments
    log_root_path = _rsl_rl_log_root(agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    should_resume = agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
    log_dir, resume_path = _resolve_log_dir(
        log_root_path,
        resume=should_resume,
        load_run=agent_cfg.load_run,
        load_checkpoint=agent_cfg.load_checkpoint,
        run_name=agent_cfg.run_name or None,
        distributed=args_cli.distributed,
        global_rank=global_rank,
    )

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.profile_physics_steps > 0:
        from robolab.utils.physics_memory import make_profile_policy, profile_physics

        writer = None
        try:
            policy = None
            if args_cli.profile_policy_checkpoint:
                policy, writer = make_profile_policy(env.unwrapped, agent_cfg, args_cli.profile_policy_checkpoint)
            profile_physics(env.unwrapped, args_cli.profile_physics_steps,
                            args_cli.diagnostic_action_scale, args_cli.profile_physics_trace,
                            args_cli.profile_physics_nan, policy=policy)
        finally:
            if writer is not None:
                writer.close()
            env.close()
        return

    if args_cli.diagnostic_steps > 0:
        try:
            _diagnose_physics(env.unwrapped, args_cli.diagnostic_steps)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            raise
        finally:
            env.close()
        return

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "FbCprRunner":
        runner_cfg = agent_cfg.to_dict()
        runner_cfg["tracking_eval_entrypoint"] = os.path.abspath(__file__)
        runner = FbCprRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if isinstance(runner, FbCprRunner):
            runner.load(resume_path)
        elif args_cli.reset_amp_discriminators:
            if agent_cfg.algorithm.class_name != "PPOAMP":
                raise ValueError("--reset_amp_discriminators is only supported for PPOAMP training")
            print("[INFO]: Reinitializing AMP discriminators and discriminator optimizers.")
            runner.load(
                resume_path,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": True,
                    "iteration": True,
                    "rnd": True,
                    "discriminators": False,
                    "disc_optimizers": False,
                },
                map_location=agent_cfg.device,
            )
        else:
            runner.load(resume_path, map_location=agent_cfg.device)

    if args_cli.resume_learning_rate is not None:
        if not agent_cfg.resume or agent_cfg.algorithm.class_name not in {"PPO", "PPOAMP"}:
            raise ValueError("--resume_learning_rate requires a resumed PPO or PPOAMP run")
        cli_args.override_resumed_learning_rate(runner.alg, args_cli.resume_learning_rate)
        agent_cfg.algorithm.learning_rate = args_cli.resume_learning_rate
        print(f"[INFO]: Resumed PPO learning rate: {runner.alg.learning_rate}, schedule: {runner.alg.schedule}")

    # dump the configuration into log-directory (rank 0 only in distributed mode)
    if not args_cli.distributed or global_rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except BaseException:
        # Simulator shutdown may terminate Python before the outer handler runs.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if isinstance(runner, FbCprRunner):
            runner.close_evaluator()
        env.close()


if __name__ == "__main__":
    try:
        if args_cli.tracking_eval_request:
            _run_tracking_eval_worker(args_cli.tracking_eval_request)
        else:
            run_resolved_task(args_cli, main)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
