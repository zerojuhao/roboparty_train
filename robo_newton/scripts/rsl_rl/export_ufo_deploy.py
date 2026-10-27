#!/usr/bin/env python3
"""Export an UFO checkpoint and motion context for CPU/MuJoCo deployment.

The visual PUFO branch exports a depth encoder in addition to the actor. UFO
has no depth input, so this exporter writes only the proprioceptive actor,
backward context encoder, and a precomputed motion context sidecar:

    ufo_actor.onnx          state/history/heading/velocity/z -> actions
    ufo_backward_encoder.onnx  state/privileged_state -> z
    context_1.npz            z, heading and velocity targets for one motion
    export_meta.json          observation, actuator and timing contract

The exporter does not start IsaacLab or consume a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rsl_rl.algorithms.fb_networks import (
    BackwardContextOnnxExporter,
    BackwardMap,
    ObsNormalizer,
    ResidualActor,
)
from rsl_rl.runners.fbcpr_runner import default_rsl_rl_log_root
from robolab.tasks.manager_based.ufo.mdp.expert_motion_loader import (
    load_expert_trajectories_from_motion_lib,
)
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import world_to_yaw
from robolab.tasks.manager_based.ufo.motion.motion_lib_npz import MotionLibNPZ


class Loader(yaml.SafeLoader):
    pass


for _tag in ("python/tuple", "python/object/apply:builtins.slice"):
    Loader.add_constructor(
        "tag:yaml.org,2002:" + _tag,
        lambda loader, node: loader.construct_sequence(node),
    )


def read_yaml(path: Path) -> dict:
    return yaml.load(path.read_text(), Loader=Loader)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_run(log_root: Path, load_run: str | None) -> Path:
    if load_run is None:
        candidates = sorted(
            (path for path in log_root.iterdir() if (path / "params/agent.yaml").is_file()),
            key=lambda path: path.name,
        )
        if not candidates:
            raise FileNotFoundError(f"No UFO runs found below {log_root}")
        return candidates[-1]
    candidate = Path(load_run)
    return candidate if candidate.is_dir() else log_root / load_run


def resolve_checkpoint(run: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        candidate = Path(checkpoint)
        return candidate if candidate.is_file() else run / checkpoint
    files = [path for path in run.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", path.name)]
    if not files:
        raise FileNotFoundError(f"No model checkpoint found in {run}")
    return max(files, key=lambda path: int(path.stem.split("_")[-1]))


def joint_values(value, names: list[str]) -> list[float]:
    result = []
    for name in names:
        if isinstance(value, (int, float)):
            result.append(float(value))
            continue
        matches = [float(item) for pattern, item in value.items() if re.fullmatch(pattern, name)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous default joint position for {name}: {matches}")
        result.append(matches[0] if matches else 0.0)
    return result


class ActorOnnxExporter(nn.Module):
    """Keep normalization and the velocity estimator inside the actor graph."""

    def __init__(self, actor, normalizer, actor_keys):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.actor_keys = tuple(actor_keys)

    def forward(self, state, last_action, history_actor, heading, velocity_command, z):
        values = {
            "state": state,
            "last_action": last_action,
            "history_actor": history_actor,
            "heading": heading,
            "velocity_command": velocity_command,
        }
        normalized = self.normalizer.normalize_dict(values)
        actor_input = torch.cat([normalized[key] for key in self.actor_keys], dim=-1)
        return self.actor(actor_input, z, 0.0).mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load_run", default=None, help="Run directory or run name below the UFO log root.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or model_N.pt name.")
    parser.add_argument("--motion", default="dance1_subject2")
    parser.add_argument("--start_frame", type=int, default=50)
    parser.add_argument("--motion-dir", "--manifest", dest="motion_dir", type=Path, default=None,
                        help="Override the motion directory saved in env.yaml (--manifest is a legacy alias).")
    args = parser.parse_args()
    torch.set_num_threads(4)

    log_root = Path(default_rsl_rl_log_root()) / "rp1_ufo"
    run = resolve_run(log_root, args.load_run)
    checkpoint = resolve_checkpoint(run, args.checkpoint).resolve()
    agent_cfg = read_yaml(run / "params/agent.yaml")
    env_cfg = read_yaml(run / "params/env.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    weights = state["model_state"]
    print(f"[INFO] {checkpoint} -> {run / 'exported'}", flush=True)

    suffix = ".running_mean"
    dims = {
        key[len("normalizers.") : -len(suffix)]: value.numel()
        for key, value in weights["obs_normalizer"].items()
        if key.endswith(suffix)
    }
    dims.update(heading=2, velocity_command=3, velocity_target_yaw=3)
    actor_keys = tuple(agent_cfg["actor_obs_keys"])
    privileged_keys = tuple(agent_cfg["privileged_obs_keys"])
    if actor_keys != ("state", "last_action", "history_actor", "heading", "velocity_command"):
        raise ValueError(f"Unexpected UFO actor keys: {actor_keys}")

    normalizer = ObsNormalizer(
        dims,
        identity_keys=("heading", "velocity_command", "velocity_target_yaw"),
    )
    actor = ResidualActor(
        sum(dims[key] for key in actor_keys),
        agent_cfg["z_dim"],
        dims["last_action"],
        estimator_cfg=agent_cfg["velocity_estimator"],
        estimate_velocity=agent_cfg["estimated_velocity_enabled"],
        velocity_dim=dims["velocity_target_yaw"],
        **agent_cfg["actor"],
    )
    backward = BackwardMap(
        sum(dims[key] for key in privileged_keys),
        agent_cfg["z_dim"],
        norm=agent_cfg["norm_z"],
        **agent_cfg["backward_map"],
    )
    for module, key in ((actor, "actor"), (backward, "backward_map"), (normalizer, "obs_normalizer")):
        module.load_state_dict(weights[key], strict=True)
        module.eval().requires_grad_(False)

    motion_cfg = env_cfg["commands"]["motion"]
    sources = args.motion_dir or motion_cfg.get("motion_dir") or motion_cfg["manifest_path"]
    if isinstance(sources, (str, Path)):
        sources = [sources]
    motion_dirs = [Path(source) for source in sources]
    motion_dirs = [path if path.is_absolute() else (run / path).resolve() for path in motion_dirs]
    motion_lib = MotionLibNPZ(
        {
            "motion_file": tuple(str(path) for path in motion_dirs),
            "step_dt": motion_cfg["step_dt"],
            "clip_seconds": 0.0,
            "min_clip_seconds": motion_cfg["min_clip_seconds"],
            "motion_names": [args.motion],
        },
        num_envs=1,
        device="cpu",
    )
    joint_order = list(motion_lib._motion_joints)
    default = joint_values(env_cfg["scene"]["robot"]["init_state"]["joint_pos"], joint_order)
    control_dt = float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"])
    episodes, _ = load_expert_trajectories_from_motion_lib(
        motion_lib,
        control_dt,
        torch.tensor([default], dtype=torch.float32),
        device="cpu",
        base_ang_vel_scale=float(env_cfg["observations"]["policy"]["base_ang_vel"].get("scale") or 1.0),
    )
    episode = episodes[0]
    observation = episode["observation"]
    if args.start_frame < 0 or args.start_frame + 1 >= len(observation["state"]):
        raise ValueError(f"start_frame={args.start_frame} is outside the {len(observation['state'])}-frame clip")

    with torch.inference_mode():
        normalized = normalizer.normalize_dict(
            {key: observation[key][args.start_frame + 1 :] for key in privileged_keys}
        )
        backward_input = torch.cat([normalized[key] for key in privileged_keys], dim=-1)
        z_parts = [backward(chunk) for chunk in backward_input.split(256)]
        z = torch.cat(z_parts, dim=0)
        if agent_cfg["norm_z"]:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)

    frames = len(z)
    heading = episode["heading_forward_xy"][args.start_frame + 1 : args.start_frame + 1 + frames]
    velocity = episode["reference_velocity_world_xy"][args.start_frame + 1 : args.start_frame + 1 + frames]
    anchor = episode["heading_forward_xy"][args.start_frame : args.start_frame + 1].expand_as(heading)
    valid = np.ones((frames, 1), dtype=np.uint8)

    out = run / "exported"
    out.mkdir(parents=True, exist_ok=True)
    context_path = out / "context_1.npz"
    np.savez(
        context_path,
        z=z.numpy().astype(np.float32),
        heading_relative_xy=world_to_yaw(heading, anchor).numpy().astype(np.float32),
        heading_valid=valid,
        reference_velocity_heading0_xy=world_to_yaw(velocity, anchor).numpy().astype(np.float32),
        velocity_valid=valid,
        start_frame=np.asarray(args.start_frame, dtype=np.int64),
        dt=np.asarray(control_dt, dtype=np.float64),
    )

    actor_path = out / "ufo_actor.onnx"
    actor_exporter = ActorOnnxExporter(actor, normalizer, actor_keys).eval()
    dummy = tuple(torch.zeros((1, dims[key])) for key in actor_keys) + (
        torch.zeros((1, agent_cfg["z_dim"])),
    )
    torch.onnx.export(
        actor_exporter,
        dummy,
        actor_path,
        opset_version=18,
        input_names=[*actor_keys, "z"],
        output_names=["actions"],
        dynamic_axes={name: {0: "batch"} for name in [*actor_keys, "z", "actions"]},
        external_data=False,
        do_constant_folding=True,
        dynamo=False,
    )

    context_path_onnx = out / "ufo_backward_encoder.onnx"
    context_exporter = BackwardContextOnnxExporter(
        backward,
        normalizer,
        privileged_keys,
        norm_z=False,
    ).eval()
    context_dummy = tuple(torch.zeros((1, dims[key])) for key in privileged_keys)
    torch.onnx.export(
        context_exporter,
        context_dummy,
        context_path_onnx,
        opset_version=18,
        input_names=list(privileged_keys),
        output_names=["z"],
        dynamic_axes={key: {0: "batch"} for key in [*privileged_keys, "z"]},
        external_data=False,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        rng = torch.Generator().manual_seed(0)
        samples = []
        for _ in range(4):
            values = {key: torch.randn((1, dims[key]), generator=rng) for key in actor_keys}
            values["heading"] = F.normalize(values["heading"], dim=-1)
            values["velocity_command"][:, 2] = 1.0
            z_sample = F.normalize(torch.randn((1, agent_cfg["z_dim"]), generator=rng), dim=-1) * math.sqrt(agent_cfg["z_dim"])
            torch_action = actor_exporter(*tuple(values[key] for key in actor_keys), z_sample)
            feeds = {key: values[key].numpy() for key in actor_keys} | {"z": z_sample.numpy()}
            onnx_action = session.run(None, feeds)[0]
            samples.append(float(np.abs(onnx_action - torch_action.numpy()).max()))
    if max(samples) > 1.0e-4:
        raise RuntimeError(f"UFO ONNX parity failed: {samples}")
    print(f"[INFO] actor ONNX parity max errors: {samples}", flush=True)

    actuator_cfg = env_cfg["scene"]["robot"]["actuators"]["all_joints"]
    meta = {
        "format": "ufo_deploy_v1",
        "checkpoint": str(checkpoint),
        "global_step": int(state["env_iterations"] * state["num_envs"] * state["world_size"]),
        "motion": args.motion,
        "start_frame": args.start_frame,
        "frames": frames,
        "dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "action_rescale": float(state["action_rescale"]),
        "action_scale": float(env_cfg["actions"]["joint_pos"]["scale"]),
        "action_dim": int(dims["last_action"]),
        "z_dim": int(agent_cfg["z_dim"]),
        "actor_inputs": {key: int(dims[key]) for key in actor_keys} | {"z": int(agent_cfg["z_dim"])},
        "actor_output": "actions",
        "actor_file": actor_path.name,
        "backward_encoder_file": context_path_onnx.name,
        "context_file": context_path.name,
        "context_z": "project_z(B(next_expert_obs[t]))",
        "joint_order": joint_order,
        "default_joint_pos": default,
        "body_order": list(motion_lib._motion_bodies),
        "base_ang_vel_scale": float(env_cfg["observations"]["policy"]["base_ang_vel"].get("scale") or 1.0),
        "actuator_cfg": actuator_cfg,
        "physics_dt": float(env_cfg["sim"]["dt"]),
        "decimation": int(env_cfg["decimation"]),
        "parity_action_max_abs": samples,
        "sha256": {path.name: sha256(path) for path in (actor_path, context_path_onnx, context_path)},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[INFO] deployment assets written to {out}", flush=True)


if __name__ == "__main__":
    main()
