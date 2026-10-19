"""Replay one UFO motion with a checkpoint and optionally record a video."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Replay and record an RP1 UFO motion.")
parser.add_argument("--task", type=str, default="RP1-UFO")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--motion", type=str, default="dance1_subject2")
parser.add_argument("--video", action="store_true", help="Record an RGB video.")
parser.add_argument("--video_length", type=int, default=0, help="Frames to record; 0 records the whole motion.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for the recorded video.")
parser.add_argument("--real-time", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
record_video = bool(args_cli.video)
# IsaacLab treats a ``video`` attribute as its MoviePy-based recorder switch;
# VideoRecorderCfg below is configured explicitly after the app starts.
delattr(args_cli, "video")
if record_video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg, VideoRecorderCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from rsl_rl.runners import FbCprRunner
from rsl_rl.runners.fbcpr_runner import default_rsl_rl_log_root, _obs_term_scale

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401
from robolab.tasks.manager_based.ufo.mdp.expert_motion_loader import load_expert_trajectories_from_motion_lib
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
    heading_observation,
    velocity_observation,
    world_to_yaw,
    yaw_to_world,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.0.0")
    agent_cfg.logger = "tensorboard"
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.motion.motion_names = (args_cli.motion,)
    env_cfg.commands.motion.clip_seconds = 0.0
    env_cfg.commands.motion.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = 100000.0
    # Keep the recording deterministic and avoid external pushes or startup DR.
    for name in (
        "push_robot", "randomize_rigid_body_material", "randomize_rigid_body_com",
        "scale_link_mass", "scale_actuator_gains",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.join(default_rsl_rl_log_root(), agent_cfg.experiment_name)
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    checkpoint = os.path.abspath(checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)
    video_path = None
    if record_video:
        video_dir = args_cli.output_dir or os.path.join(os.path.dirname(checkpoint), "videos", "ufo")
        os.makedirs(video_dir, exist_ok=True)
        prefix = f"{args_cli.motion}_{os.path.basename(checkpoint).replace('.pt', '')}"
        video_path = os.path.join(video_dir, prefix + "_0000.mp4")
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=(4.5, -6.0, 2.8),
            lookat=(0.0, 0.0, 1.0),
            max_visible_envs=1,
            headless=True,
            window_width=1280,
            window_height=720,
            streaming_view=False,
            enable_live_plots=False,
            enable_markers=False,
        )]
        env_cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:newton_gl",
            output_dir=video_dir,
            video_length=args_cli.video_length if args_cli.video_length > 0 else 100000000,
            video_interval=0,
            output_filename_prefix=prefix,
        )]
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = FbCprRunner(env, agent_cfg.to_dict(), log_dir=tempfile.mkdtemp(prefix="play_ufo_"), device=agent_cfg.device)
    runner.load(checkpoint, load_optimizer=False)
    agent = runner.agent

    robot = env.unwrapped.scene["robot"]
    episodes, _ = load_expert_trajectories_from_motion_lib(
        env.unwrapped.command_manager.get_term("motion").motion_lib,
        runner.control_dt,
        robot.data.default_joint_pos,
        device=runner.device,
        base_ang_vel_scale=_obs_term_scale(env_cfg.observations.policy.base_ang_vel),
    )
    episode = episodes[0]
    with torch.inference_mode():
        # Match the training context generator: backward-map next expert obs,
        # then average the causal seq_length window before projection.
        privileged = {
            key: value[1:] for key, value in episode["observation"].items()
            if key in agent.privileged_keys
        }
        z_raw = agent._backward_map(agent._b_in(privileged))
        z = torch.stack([
            z_raw[i:min(i + agent.cfg.seq_length, len(z_raw))].mean(dim=0)
            for i in range(len(z_raw))
        ])
        z = agent.project_z(z)
    headings = episode["heading_forward_xy"][1:]
    velocities = episode["reference_velocity_world_xy"][1:]
    frames = min(len(z), len(headings), len(velocities))
    if args_cli.video_length > 0:
        frames = min(frames, args_cli.video_length)
    obs_buf, _ = env.reset()
    runner.history.reset(torch.arange(env.num_envs, device=runner.device))
    with torch.inference_mode():
        current_heading, _ = runner._root_heading_velocity()
        # The sidecar-style relative targets are anchored once at playback
        # start.  Reprojecting them with the changing current heading makes
        # the target rotate along with the robot and can trigger a spin.
        start_heading = current_heading.clone()
        anchor = headings[0:1].to(runner.device)
        relative_heading = world_to_yaw(headings[:frames].to(runner.device), anchor.expand(frames, -1))
        relative_velocity = world_to_yaw(velocities[:frames].to(runner.device), anchor.expand(frames, -1))
        for step in range(frames):
            current_heading, _ = runner._root_heading_velocity()
            obs = runner._obs_dict(obs_buf)
            target_heading = yaw_to_world(relative_heading[step][None], start_heading)
            target_velocity = yaw_to_world(relative_velocity[step][None], start_heading)
            valid = torch.ones((1, 1), dtype=torch.bool, device=runner.device)
            obs.update({
                "heading": heading_observation(current_heading, target_heading, valid),
                "velocity_command": velocity_observation(target_velocity, current_heading, valid),
            })
            action = agent.act(obs, z[step][None], mean=True)
            current_values = runner._current_obs_values(obs_buf)
            obs_buf, _, dones, _ = env.step(runner._env_action(action))
            runner.history.add(current_values)
            if bool(dones.any()):
                runner.history.reset(dones.nonzero(as_tuple=False).flatten())
            if args_cli.real_time:
                time.sleep(max(0.0, runner.control_dt))
    env.close()
    print(f"[UFO playback] motion={args_cli.motion} frames={frames} checkpoint={checkpoint} video={video_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
