# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a PUFO checkpoint and export everything the rp1 robot needs.

Written to ``<run>/exported/`` (next to the training exporter's depth_encoder.onnx,
backward_encoder.onnx and pufo_onnx.json):
    pbfm_actor.onnx    (state, last_action, history_actor, heading, velocity_command, z, depth_latent) -> actions
    pbfm_encoder.onnx  depth_m float32 meters [B, 8, 18, 36] -> depth_latent; clip/normalize/uint8 in graph
    context_1.npz      tracking context of ``--motion``: z, heading, velocity (start-heading frame), dt
    export_meta.json

Then tracks the exported context in simulation. For the same assets without launching Isaac,
use export_pufo_deploy.py.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a PUFO agent and export the rp1 deployment assets.")
parser.add_argument("--task", type=str, default="RP1-PUFO", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--motion", type=str, default="dance1_subject2", help="Motion clip exported as the context.")
parser.add_argument("--start_frame", type=int, default=50, help="First motion frame of the context.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import tempfile
import time
import zipfile

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

from rsl_rl.algorithms.fb_networks import DepthEncoderOnnxExporter
from rsl_rl.runners import FbCprRunner
from rsl_rl.runners.fbcpr_runner import _obs_term_scale, default_rsl_rl_log_root

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robolab.tasks  # noqa: F401
from robolab.tasks.manager_based.ufo.expert_motion_loader import load_expert_trajectories_from_motion_lib
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
    heading_observation,
    velocity_observation,
    world_to_yaw,
    yaw_to_world,
)


class MetricDepthEncoder(torch.nn.Module):
    """Meters in; clip, normalize and uint8-quantize exactly as the training env and runner do."""

    def __init__(self, encoder, low: float, high: float):
        super().__init__()
        self.encoder, self.low, self.high = encoder, low, high

    def forward(self, depth_m):
        normalized = (depth_m.clip(self.low, self.high) - self.low) / (self.high - self.low)
        return self.encoder(normalized.mul(255.0).round().to(torch.uint8))


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.logger = "tensorboard"
    agent_cfg.torch_compile_mode = None  # ONNX tracing cannot pass through torch.compile

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # One unwindowed clip, so the context spans the whole motion as on the robot.
    env_cfg.commands.motion.motion_names = (args_cli.motion,)
    env_cfg.commands.motion.clip_seconds = 0.0
    env_cfg.episode_length_s = 100000.0

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join(default_rsl_rl_log_root(), agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = FbCprRunner(env, agent_cfg.to_dict(), log_dir=tempfile.mkdtemp(prefix="play_pufo_"), device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    agent = runner.agent

    # ---- actor (training exporter) and metric-depth encoder
    out = os.path.join(env_cfg.log_dir, "exported")
    actor_path = runner.export_policy_to_onnx(out, actor_filename="pbfm_actor.onnx")["actor"]
    encoder_path = os.path.join(out, "pbfm_encoder.onnx")
    actor = copy.deepcopy(agent._actor).cpu().eval()
    normalizer = copy.deepcopy(agent._obs_normalizer).cpu().eval()
    pipeline = env_cfg.scene.camera.noise_pipeline
    low, high = (float(v) for v in pipeline["depth_normalization"].depth_range)
    depth_shape = tuple(agent.depth_shape)
    torch.onnx.export(
        MetricDepthEncoder(DepthEncoderOnnxExporter(actor.depth_encoder), low, high).eval(),
        (torch.full((1, *depth_shape), high),), encoder_path,
        opset_version=18, input_names=["depth_m"], output_names=["depth_latent"],
        dynamic_axes={"depth_m": {0: "batch"}, "depth_latent": {0: "batch"}},
        external_data=False, do_constant_folding=True, dynamo=False,
    )

    # ---- tracking context: training's expert episode, single-frame z[t] = project(B(next_obs[t]));
    # the robot smooths it over its own window_size.
    robot = env.unwrapped.scene["robot"]
    episode = load_expert_trajectories_from_motion_lib(
        env.unwrapped.command_manager.get_term("motion").motion_lib, runner.control_dt,
        robot.data.default_joint_pos, device=runner.device,
        base_ang_vel_scale=_obs_term_scale(env_cfg.observations.policy.base_ang_vel),
    )[0][0]
    start = args_cli.start_frame
    with torch.inference_mode():
        z = agent.project_z(agent._backward_map(agent._b_in({k: v[start + 1:] for k, v in episode["observation"].items()})))
    frames = len(z)
    heading = episode["heading_forward_xy"][start : start + frames]
    anchor = heading[:1].expand_as(heading)
    heading_rel = world_to_yaw(heading, anchor)
    velocity_rel = world_to_yaw(episode["reference_velocity_world_xy"][start : start + frames], anchor)
    valid = np.ones((frames, 1), dtype=np.uint8)
    members = {
        "z": z, "heading_relative_xy": heading_rel, "heading_valid": valid,
        "reference_velocity_heading0_xy": velocity_rel, "velocity_valid": valid,
        "dt": np.array([runner.control_dt], dtype=np.float64),
    }
    context_path = os.path.join(out, "context_1.npz")
    # Plain stored zip without zip64 (cnpy cannot read numpy's forced zip64) and fixed timestamps.
    with zipfile.ZipFile(context_path, "w", allowZip64=False) as archive:
        for name, value in members.items():
            array = value.float().cpu().numpy() if torch.is_tensor(value) else value
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite context {name}")
            handle = io.BytesIO()
            np.save(handle, np.ascontiguousarray(array))
            archive.writestr(zipfile.ZipInfo(f"{name}.npy"), handle.getvalue())

    # ---- ONNX contract and parity against the CPU torch modules
    sess = ort.InferenceSession(actor_path, providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    actor_keys = list(agent.actor_keys)
    if [i.name for i in sess.get_inputs()] != actor_keys + ["z", "depth_latent"] or \
            [o.name for o in sess.get_outputs()] != ["actions"] or \
            [(i.name, i.type) for i in enc.get_inputs()] != [("depth_m", "tensor(float)")]:
        raise RuntimeError("Exported ONNX contract mismatch")
    rng = torch.Generator().manual_seed(0)
    parity = []
    with torch.inference_mode():
        for _ in range(4):
            depth_m = torch.rand((1, *depth_shape), generator=rng) * 3.0  # exercises the in-graph clip
            depth_u8 = ((depth_m.clip(low, high) - low) / (high - low)).mul(255.0).round().to(torch.uint8)
            latent = actor.depth_encoder(depth_u8)
            obs = {k: torch.randn((1, runner.obs_dims[k]), generator=rng) for k in actor_keys}
            obs["heading"] = torch.nn.functional.normalize(obs["heading"], dim=-1)
            obs["velocity_command"][:, 2] = 1.0
            z_sample = agent.project_z(torch.randn((1, z.shape[-1]), generator=rng))
            action = actor.mean_from_depth_latent(normalizer.normalize_dict(obs), z_sample, latent)
            feeds = {k: v.numpy() for k, v in obs.items()} | {"z": z_sample.numpy(), "depth_latent": latent.numpy()}
            parity.append({
                "encoder": float(np.abs(enc.run(None, {"depth_m": depth_m.numpy()})[0] - latent.numpy()).max()),
                "action": float(np.abs(sess.run(None, feeds)[0] - action.numpy()).max()),
            })
    print(f"[INFO]: ONNX parity {parity}")
    if any(max(item.values()) > 1e-4 for item in parity):
        raise RuntimeError(f"ONNX parity failed: {parity}")

    meta = {
        "checkpoint": resume_path,
        "motion": args_cli.motion,
        "start_frame": start,
        "frames": frames,
        "dt": runner.control_dt,
        "action_rescale": runner.action_rescale,
        "action_scale": float(env_cfg.actions.joint_pos.scale),
        "z": "single-frame project_z(B(next_obs[t])); training seq_length average not applied",
        "depth_shape": list(depth_shape),
        "depth_crop_up_down_left_right": list(pipeline["crop_and_resize"].crop_region),
        "depth_range_m": [low, high],
        "encoder_input": "depth_m float32 meters; clip/normalize/uint8 quantize in graph",
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "parity": parity,
        "sha256": {os.path.basename(p): sha256(p) for p in (actor_path, encoder_path, context_path)},
    }
    with open(os.path.join(out, "export_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[INFO]: rp1 deployment assets exported to {out}\n{json.dumps(meta['sha256'], indent=2)}")

    # ---- track the exported context like the robot: lock the start heading, then step through z[t]
    device = runner.device
    z, heading_rel, velocity_rel = z.to(device), heading_rel.to(device), velocity_rel.to(device)
    always = torch.ones(env.num_envs, 1, dtype=torch.bool, device=device)
    obs_buf, t = env.get_observations(), 0
    runner.history.reset(torch.arange(env.num_envs, device=device))
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            current, _ = runner._root_heading_velocity()
            if t == 0:
                start_heading = current.clone()
            obs = runner._obs_dict(obs_buf)
            target_heading = yaw_to_world(heading_rel[t].expand_as(start_heading), start_heading)
            target_velocity = yaw_to_world(velocity_rel[t].expand_as(start_heading), start_heading)
            obs["heading"] = heading_observation(current, target_heading, always)
            obs["velocity_command"] = velocity_observation(target_velocity, current, always)
            action = agent.act(obs, z[t].expand(env.num_envs, -1), mean=True)
            current_values = runner._current_obs_values(obs_buf)
            obs_buf, _, dones, _ = env.step(runner._env_action(action))
            runner.history.add(current_values)
            if dones.any():
                runner.history.reset(dones.nonzero(as_tuple=False).flatten())
            t = 0 if dones.any() or t + 1 == frames else t + 1
        sleep_time = runner.control_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
