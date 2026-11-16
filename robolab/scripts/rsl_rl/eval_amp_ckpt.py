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

import argparse
import sys
from pathlib import Path

from robolab.utils.kitless_runtime import configure_kitless_runtime, parse_isaaclab_cli, run_resolved_task, sim_is_running

configure_kitless_runtime()

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--plane", action='store_true', help="Use plane terrain")
parser.add_argument("--push_robot", action='store_true', help="Push robot during playing")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many policy steps; zero runs continuously.")
parser.add_argument("--out", type=str, default=None, help="Output dir (or JSON path in single mode).")
parser.add_argument("--checkpoints-file", type=str, default=None, help="File with one checkpoint path per line (batch mode).")
parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                    help="Hold one user velocity command during playback.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
args_cli = parse_isaaclab_cli(parser)

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, FbCprRunner, OnPolicyRunner
from rsl_rl.runners.fbcpr_runner import default_rsl_rl_log_root

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.markers import VisualizationMarkers

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401

def visualize_attention(map_scan, root_pose, output_attn, visualizer):
    root_pos = root_pose[:, :3]  # shape (B, 3)
    map_scan_world = -map_scan + root_pos.unsqueeze(1).unsqueeze(1)  # shape (B, W, L, 3)
    B = root_pos.shape[0]

    max_attn_per_env, _ = torch.max(output_attn.view(B, -1), dim=1)
    max_attn_per_env[max_attn_per_env == 0] = 1.0
    normalized_attn = output_attn / max_attn_per_env.view(B, 1, 1)
    normalized_attn = normalized_attn.view(-1)
    # print(normalized_attn)
    attention_indices = torch.zeros_like(normalized_attn, dtype=torch.int)
    for i in range(10):
        color_mask = (normalized_attn > 0.1 * i)
        color_mask = torch.bitwise_and(color_mask, normalized_attn < 0.1 * (i + 1))
        attention_indices[color_mask] = i
    visualizer.visualize(translations=map_scan_world.view(-1, 3), marker_indices=attention_indices)

def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    if agent_cfg.class_name != "FbCprRunner":
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

    manager_based = isinstance(env_cfg, ManagerBasedRLEnvCfg)
    if manager_based:
        for group in vars(env_cfg.observations).values():
            if hasattr(group, 'enable_corruption'):
                group.enable_corruption = False
    else:
        env_cfg.noise.add_noise = False
    if not args_cli.push_robot:
        env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 40.0
    if manager_based:
        env_cfg.commands.base_velocity.heading_command = False
    else:
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_standing_envs = 0.0

    if args_cli.plane:
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.terrain_type = "plane"

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

    if hasattr(env_cfg, "attn_enc"):
        visualizer = VisualizationMarkers(env_cfg.attn_enc.marker_cfg)

    if hasattr(env_cfg, 'interrupt') and env_cfg.interrupt.use_interrupt:
        env_cfg.interrupt.interrupt_ratio = 1.0

    # specify directory for logging experiments
    log_root_path = os.path.join(default_rsl_rl_log_root(), agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    from robolab.tasks.manager_based.amp.mdp.force_estimation import configure_force_estimation
    configure_force_estimation(env_cfg, agent_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if not manager_based:
        env.unwrapped.command_generator.command[:] = 0.0

    if hasattr(env_cfg, 'interrupt') and env_cfg.interrupt.use_interrupt:
        env.unwrapped.interrupt_rad_curriculum = torch.ones(env_cfg.scene.num_envs, dtype=torch.float, device=env.unwrapped.device, requires_grad=False)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.velocity is not None:
        velocity = torch.tensor(args_cli.velocity, device=env.unwrapped.device)
        if manager_based:
            command = env.unwrapped.command_manager.get_term('base_velocity')
            command._update_command = lambda: command.vel_command_b.copy_(velocity.expand_as(command.vel_command_b))
            command._update_command()
            env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
        else:
            env.unwrapped.command_generator.command[:] = velocity

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "FbCprRunner":
        runner = FbCprRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if isinstance(runner, FbCprRunner):
        runner.load(resume_path, load_optimizer=False)
    else:
        runner.load(resume_path, map_location=agent_cfg.device)

    # ---- batch rollout: 一个进程内循环加载多个 ckpt ----
    import json as _json
    import torch as _t

    _ckpts = (
        [ln.strip() for ln in open(args_cli.checkpoints_file) if ln.strip()]
        if args_cli.checkpoints_file
        else [resume_path]
    )
    _outdir = args_cli.out or "/data/zym/eval_out"
    os.makedirs(_outdir, exist_ok=True)

    _robot = env.unwrapped.scene["robot"]
    _bn = list(_robot.data.body_names)
    _li = next(i for i, n in enumerate(_bn) if "left_ankle_roll_link" in n)
    _ri = next(i for i, n in enumerate(_bn) if "right_ankle_roll_link" in n)
    _orig = env.unwrapped.scene.env_origins[:, 2]
    _cmdterm = env.unwrapped.command_manager.get_term("base_velocity")

    print(f"[BATCH] {len(_ckpts)} checkpoints -> {_outdir}", flush=True)
    for _k, _ck in enumerate(_ckpts):
        _name = os.path.basename(os.path.dirname(_ck))[-24:] + "__" + os.path.basename(_ck).replace(".pt", "")
        _out = os.path.join(_outdir, _name + ".json")
        if os.path.exists(_out):
            print(f"[SKIP] {_k+1}/{len(_ckpts)} {_name}", flush=True)
            continue
        try:
            runner.load(_ck, map_location=agent_cfg.device)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            obs = env.get_observations()
            rec = {"foot_z": [], "root_z": [], "yaw_rate": [], "cmd": []}
            for _ in range(args_cli.steps):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                rec["foot_z"].append(
                    (_robot.data.body_pos_w[:, [_li, _ri], 2] - _orig.unsqueeze(1)).detach().cpu()
                )
                rec["root_z"].append((_robot.data.root_pos_w[:, 2] - _orig).detach().cpu())
                rec["yaw_rate"].append(_robot.data.root_ang_vel_w[:, 2].detach().cpu())
                rec["cmd"].append(_cmdterm.command.detach().cpu())
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {_k+1}/{len(_ckpts)} {_name}: {type(e).__name__}: {e}", flush=True)
            continue

        fz = _t.stack(rec["foot_z"]).float()
        rz = _t.stack(rec["root_z"]).float()
        wz = _t.stack(rec["yaw_rate"]).float()
        cmd = _t.stack(rec["cmd"]).float()
        _skip = max(1, int(fz.shape[0] * 0.2))
        fz, rz, wz, cmd = fz[_skip:], rz[_skip:], wz[_skip:], cmd[_skip:]
        _gnd = _t.quantile(fz.reshape(-1), 0.05)
        lift = (fz - _gnd).clamp(min=0.0)
        air = (lift > 0.02).float()
        out = {
            "checkpoint": _ck,
            "steps": int(fz.shape[0]),
            "num_envs": int(fz.shape[1]),
            "air_ratio": float(air.mean()),
            "air_ratio_per_foot": [float(air[..., i].mean()) for i in range(2)],
            "foot_clear_p95": float(lift.flatten().quantile(0.95)),
            "foot_clear_max": float(lift.max()),
            "foot_clear_p95_l": float(lift[..., 0].flatten().quantile(0.95)),
            "foot_clear_p95_r": float(lift[..., 1].flatten().quantile(0.95)),
            "foot_clear_max_l": float(lift[..., 0].max()),
            "foot_clear_max_r": float(lift[..., 1].max()),
            "foot_sym_p95_gap": float(abs(float(lift[..., 0].flatten().quantile(0.95)) - float(lift[..., 1].flatten().quantile(0.95)))),
            "foot_sym_p95_ratio": float(min(float(lift[..., 0].flatten().quantile(0.95)), float(lift[..., 1].flatten().quantile(0.95))) / max(1e-6, max(float(lift[..., 0].flatten().quantile(0.95)), float(lift[..., 1].flatten().quantile(0.95))))),
            "ground_ref": float(_gnd),
            "root_z_mean": float(rz.mean()),
            "root_z_std": float(rz.std()),
            "yaw_cmd_abs_mean": float(cmd[..., 2].abs().mean()),
            "yaw_rate_abs_mean": float(wz.abs().mean()),
            "yaw_track_err": float((wz - cmd[..., 2]).abs().mean()),
        }
        open(_out, "w").write(_json.dumps(out, indent=2))
        print(
            f"[OK] {_k+1}/{len(_ckpts)} {_name} air={out['air_ratio']:.3f} "
            f"p95={out['foot_clear_p95']:.3f} yawerr={out['yaw_track_err']:.3f}",
            flush=True,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    run_resolved_task(args_cli, main)
