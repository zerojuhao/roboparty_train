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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--eval_steps", type=int, default=0,
                    help="Run bounded headless evaluation and print tracking metrics without exporting models.")
parser.add_argument("--amp_observation_dump", type=str,
                    help="Save discriminator inputs every 10 steps during bounded evaluation for offline diagnosis.")
parser.add_argument("--stochastic_actions", action="store_true",
                    help="Sample the policy's learned action distribution instead of using its mean.")
parser.add_argument(
    "--speed_ramp_max",
    type=float,
    default=None,
    help="Override playback commands with a forward-speed ramp ending at this speed in m/s.",
)
parser.add_argument(
    "--video_renderer",
    choices=("newton_gl", "newton_rtx"),
    default="newton_gl",
    help="Newton visualizer used for video capture.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.amp_observation_dump and args_cli.eval_steps <= 0:
    parser.error("--amp_observation_dump requires --eval_steps > 0")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import copy
import json
import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import robolab.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            input_size = getattr(self.actor, "input_dim", None)
            if input_size is None:
                input_size = self.actor[0].in_features
            obs = torch.zeros(1, input_size)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )



@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.speed_ramp_max is not None:
        if args_cli.speed_ramp_max <= 0.0:
            raise ValueError("--speed_ramp_max must be positive")
        command_cfg = getattr(env_cfg.commands, "single_axis_velocity", None)
        if command_cfg is None:
            raise ValueError("--speed_ramp_max requires a single_axis_velocity command")
        command_cfg.resampling_time_range = (1.0e6, 1.0e6)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)

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

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Video recording is driven from env.step() in Isaac Lab 3.0. The legacy
    # gymnasium RecordVideo wrapper receives no frames from ManagerBasedRLEnv.
    if args_cli.video:
        # Keep the training collision plane and friction unchanged, but use a
        # quieter recording-only material so the robots remain legible.
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.16, 0.18, 0.20), roughness=0.85
        )
        if args_cli.speed_ramp_max is not None:
            env_cfg.scene.env_spacing = 1.6
            resolution = (1920, 1080)
            camera_eye = (5.5, -7.0, 3.2)
            camera_lookat = (0.0, 0.0, 1.0)
        else:
            resolution = tuple(env_cfg.viewer.resolution)
            camera_eye = tuple(env_cfg.viewer.eye)
            camera_lookat = tuple(env_cfg.viewer.lookat)
        visualizer_cls = NewtonRTXVisualizerCfg if args_cli.video_renderer == "newton_rtx" else NewtonGLVisualizerCfg
        visualizer_kwargs = {
            "eye": camera_eye,
            "lookat": camera_lookat,
            "max_visible_envs": env_cfg.scene.num_envs,
            "headless": True,
            "window_width": resolution[0],
            "window_height": resolution[1],
            "focal_length": 26.0,
            "streaming_view": False,
            "enable_live_plots": False,
            "enable_markers": False,
        }
        if args_cli.video_renderer == "newton_rtx":
            visualizer_kwargs["rtx_environment"] = "studio"
        env_cfg.sim.visualizer_cfgs = [
            visualizer_cls(**visualizer_kwargs)
        ]
        video_cfg = VideoRecorderCfg(
            source=f"visualizer:{args_cli.video_renderer}",
            output_dir=os.path.join(log_dir, "videos", "play"),
            video_length=args_cli.video_length,
            video_interval=0,
            output_filename_prefix=(
                f"rp1_amp_speed_ramp_{args_cli.speed_ramp_max:g}_"
                if args_cli.speed_ramp_max is not None
                else f"rp1_amp_many_robots_{args_cli.video_renderer}_"
            )
            + (
                f"{os.path.splitext(os.path.basename(resume_path))[0]}"
            ),
        )
        env_cfg.video_recorders = [video_cfg]
        print("[INFO] Recording video with Isaac Lab's internal recorder.")
        print_dict(video_cfg.to_dict(), nesting=4)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = runner.alg.get_policy()

    # Export through the runner so model-specific estimators are preserved.
    if not args_cli.eval_steps:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    robot = env.unwrapped.scene["robot"]
    command_term = None
    if args_cli.speed_ramp_max is not None:
        command_term = env.unwrapped.command_manager.get_term("single_axis_velocity")
        command_term.vel_command_b.zero_()
        command_term.is_standing_env.zero_()
    initial_joint_pos = robot.data.joint_pos.torch.clone()
    initial_root_pos = robot.data.root_pos_w.torch.clone()
    evaluation_samples = []
    evaluation_resets = 0
    amp_samples = {}
    if args_cli.amp_observation_dump:
        amp_settle_steps = min(round(1.0 / dt), args_cli.eval_steps // 10)
        amp_group_names = list(dict.fromkeys(
            group
            for route in runner.alg.routes.values()
            for obs_set in (route["policy_obs_set"], route["demonstration_obs_set"])
            for group in runner.alg.amp_obs_groups[obs_set]
        ))
        amp_samples = {name: [] for name in amp_group_names}
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if command_term is not None:
                hold_start = max(1, round(args_cli.video_length * 0.1))
                hold_end = max(hold_start + 1, round(args_cli.video_length * 0.8))
                ramp_fraction = min(1.0, max(0.0, (timestep - hold_start) / (hold_end - hold_start)))
                commanded_speed = args_cli.speed_ramp_max * ramp_fraction
                command_term.vel_command_b[:, 0] = commanded_speed
                command_term.vel_command_b[:, 1:] = 0.0
                command_term.is_standing_env.zero_()
                if env.unwrapped.sim.visualizers:
                    group_x = robot.data.root_pos_w.torch[:, 0].mean().item()
                    env.unwrapped.sim.visualizers[0].set_camera_view(
                        (group_x + 5.5, -7.0, 3.2), (group_x, 0.0, 1.0)
                    )
            # agent stepping
            actions = policy(obs, stochastic_output=True) if args_cli.stochastic_actions else policy(obs)
            if args_cli.eval_steps:
                commanded = env.unwrapped.command_manager.get_command("single_axis_velocity").clone()
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if args_cli.eval_steps:
                actual = torch.cat((robot.data.root_lin_vel_b.torch[:, :2],
                                    robot.data.root_ang_vel_b.torch[:, 2:3]), dim=-1)
                evaluation_samples.append(torch.stack((commanded, actual), dim=1).cpu())
                evaluation_resets += int(dones.sum().item())
            if amp_samples and timestep >= amp_settle_steps and (timestep - amp_settle_steps) % 10 == 0:
                # Skip reset histories; retain raw, unnormalized discriminator inputs.
                history_steps = max(route["disc_obs_steps"] for route in runner.alg.routes.values())
                valid = env.unwrapped.episode_length_buf > history_steps
                for name in amp_samples:
                    amp_samples[name].append(obs[name][valid].detach().cpu())
        timestep += 1
        if args_cli.video:
            if args_cli.speed_ramp_max is not None and timestep in {
                1,
                args_cli.video_length // 3,
                2 * args_cli.video_length // 3,
                args_cli.video_length,
            }:
                actual_speed = robot.data.root_lin_vel_b.torch[:, 0]
                print(
                    f"[INFO] Speed ramp at step {timestep}: command={commanded_speed:.3f} m/s, "
                    f"actual_mean={actual_speed.mean().item():.3f} m/s, "
                    f"actual_std={actual_speed.std().item():.3f} m/s"
                )
            if timestep in {max(1, args_cli.video_length // 2), args_cli.video_length}:
                joint_delta = (robot.data.joint_pos.torch - initial_joint_pos).abs()
                root_delta_xy = torch.linalg.vector_norm(
                    robot.data.root_pos_w.torch[:, :2] - initial_root_pos[:, :2], dim=1
                )
                print(
                    f"[INFO] Playback motion at step {timestep}: "
                    f"joint_delta_mean={joint_delta.mean().item():.4f} rad, "
                    f"joint_delta_max={joint_delta.max().item():.4f} rad, "
                    f"root_xy_displacement_mean={root_delta_xy.mean().item():.4f} m, "
                    f"action_abs_mean={actions.abs().mean().item():.4f}"
                )
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.eval_steps and timestep >= args_cli.eval_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if evaluation_samples:
        # Match the MuJoCo recorder's one-second (or 10%) settling period.
        settle_steps = min(round(1.0 / dt), len(evaluation_samples) // 10)
        samples = torch.stack(evaluation_samples[settle_steps:])
        command, actual = samples.unbind(dim=2)
        evaluation_summary = {
            "checkpoint": resume_path, "steps": timestep, "num_envs": env.unwrapped.num_envs,
            "stochastic_actions": args_cli.stochastic_actions,
            "reset_count": evaluation_resets,
            "command_mean_xy_yaw": command.mean(dim=(0, 1)).tolist(),
            "actual_mean_xy_yaw": actual.mean(dim=(0, 1)).tolist(),
            "tracking_mae_xy_yaw": (actual - command).abs().mean(dim=(0, 1)).tolist(),
        }
        print("EVALUATION=" + json.dumps(evaluation_summary), flush=True)

    if amp_samples:
        manager = env.unwrapped.observation_manager
        dump = {
            "checkpoint": resume_path,
            "evaluation": evaluation_summary,
            "task": args_cli.task,
            "seed": env_cfg.seed,
            "settle_steps": amp_settle_steps,
            "sample_interval_steps": 10,
            "step_dt": dt,
            "routes": runner.alg.routes,
            "amp_cfg": runner.alg.amp_cfg,
            "obs_groups": runner.alg.amp_obs_groups,
            "layouts": {
                name: {"terms": manager.active_terms[name], "shapes": manager.group_obs_term_dim[name]}
                for name in amp_samples
            },
            "observations": {name: torch.cat(samples) for name, samples in amp_samples.items()},
        }
        dump_path = os.path.abspath(args_cli.amp_observation_dump)
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "xb") as output:
            torch.save(dump, output)
        print(f"AMP_OBSERVATION_DUMP={dump_path}", flush=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
