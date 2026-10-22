#!/usr/bin/env python3
"""Export a PUFO checkpoint for rp1 real-robot deployment (CPU only, no simulator).

Networks and the tracking context are rebuilt from the run's saved params
(agent.yaml / env.yaml), the checkpoint and training's own motion library /
expert-trajectory code, so observation scales, joints and depth settings follow the run.

Written to ``<run>/exported/``:
    pbfm_actor.onnx    (state, last_action, history_actor, heading, velocity_command, z, depth_latent) -> actions
    pbfm_encoder.onnx  depth_m float32 meters [B, 8, 18, 36] -> depth_latent; clip/normalize/uint8 in graph
    context_1.npz      tracking context of ``--motion``: z, heading, velocity (start-heading frame), dt
    export_meta.json
next to the training exporter's depth_encoder.onnx (uint8 input), backward_encoder.onnx and pufo_onnx.json.

Usage:
    python robo_newton/scripts/rsl_rl/export_pufo_deploy.py [--load_run RUN] [--checkpoint model_N.pt] [--motion NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import yaml

from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent
from rsl_rl.algorithms.fb_networks import BackwardMap, DepthEncoderOnnxExporter, ObsNormalizer, VisualResidualActor
from rsl_rl.runners.fbcpr_runner import default_rsl_rl_log_root
from robolab.tasks.manager_based.ufo.mdp.expert_motion_loader import load_expert_trajectories_from_motion_lib
from robolab.tasks.manager_based.ufo.mdp.pufo_observations import world_to_yaw
from robolab.tasks.manager_based.ufo.motion.motion_lib_npz import MotionLibNPZ


class MetricDepthEncoder(torch.nn.Module):
    """Meters in; clip, normalize and uint8-quantize exactly as the training env and runner do."""

    def __init__(self, encoder, low: float, high: float):
        super().__init__()
        self.encoder, self.low, self.high = encoder, low, high

    def forward(self, depth_m):
        normalized = (depth_m.clip(self.low, self.high) - self.low) / (self.high - self.low)
        return self.encoder(normalized.mul(255.0).round().to(torch.uint8))


class Loader(yaml.SafeLoader):
    pass


for tag in ("python/tuple", "python/object/apply:builtins.slice"):
    Loader.add_constructor("tag:yaml.org,2002:" + tag, lambda loader, node: loader.construct_sequence(node))


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load_run", default=None, help="Run directory or name under the rp1_pufo log root (default: latest).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or file name in the run (default: latest model_N.pt).")
    parser.add_argument("--motion", default="dance1_subject2", help="Motion clip exported as the context.")
    parser.add_argument("--start_frame", type=int, default=50, help="First motion frame of the context.")
    parser.add_argument("--motion-dir", "--manifest", dest="motion_dir", type=Path, default=None,
                        help="Override the motion directory saved in env.yaml (--manifest is a legacy alias).")
    args = parser.parse_args()
    torch.set_num_threads(4)

    log_root = Path(default_rsl_rl_log_root()) / "rp1_pufo"
    if args.load_run is None:
        run = max((p for p in log_root.iterdir() if (p / "params/agent.yaml").is_file()), key=lambda p: p.name)
    else:
        run = Path(args.load_run) if Path(args.load_run).is_dir() else log_root / args.load_run
    if args.checkpoint is None:
        checkpoint = max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    else:
        checkpoint = Path(args.checkpoint) if Path(args.checkpoint).is_file() else run / args.checkpoint
    out = run / "exported"
    print(f"[INFO]: {checkpoint} -> {out}", flush=True)

    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=Loader)
    env = yaml.load((run / "params/env.yaml").read_text(), Loader=Loader)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    weights = state["model_state"]
    suffix = ".running_mean"
    dims = {
        key[len("normalizers."):-len(suffix)]: value.numel()
        for key, value in weights["obs_normalizer"].items()
        if key.endswith(suffix)
    }
    dims.update(heading=2, velocity_command=3, velocity_target_yaw=3)
    encoder_cfg = dict(agent["depth_encoder"])
    depth_shape = tuple(encoder_cfg.pop("input_shape"))
    action_dim = dims["last_action"]

    normalizer = ObsNormalizer(dims, identity_keys=("heading", "velocity_command", "velocity_target_yaw"))
    actor = VisualResidualActor(
        dims, agent["actor_obs_keys"], agent["z_dim"], action_dim, depth_shape=depth_shape,
        encoder_cfg=encoder_cfg, estimator_cfg=agent["velocity_estimator"],
        estimate_velocity=agent["estimated_velocity_enabled"], **agent["actor"],
    )
    backward = BackwardMap(
        sum(dims[k] for k in agent["privileged_obs_keys"]), agent["z_dim"],
        norm=agent["norm_z"], **agent["backward_map"],
    )
    for module, key in ((actor, "actor"), (backward, "backward_map"), (normalizer, "obs_normalizer")):
        module.load_state_dict(weights[key], strict=True)
        module.eval().requires_grad_(False)

    def project_z(z: torch.Tensor) -> torch.Tensor:
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1) if agent["norm_z"] else z

    # ---- tracking context: training's expert episode, single-frame z[t] = project(B(next_obs[t]));
    # the robot smooths it over its own window_size.
    motion_cfg = env["commands"]["motion"]
    dt = float(env["sim"]["dt"]) * int(env["decimation"])
    # One unwindowed clip, so the context spans the whole motion as on the robot.
    motion_dir = args.motion_dir or Path(motion_cfg.get("motion_dir") or motion_cfg["manifest_path"])
    if not motion_dir.is_absolute():
        motion_dir = (run / motion_dir).resolve()
    motion_lib = MotionLibNPZ({
        "motion_file": (str(motion_dir),), "step_dt": motion_cfg["step_dt"],
        "clip_seconds": 0.0, "min_clip_seconds": motion_cfg["min_clip_seconds"], "motion_names": [args.motion],
    }, num_envs=1, device="cpu")
    policy_terms = env["observations"]["policy"]
    # The expert loader models only the base_ang_vel scale; refuse configs it cannot reproduce.
    if any(isinstance(term, dict) and (term.get("clip") is not None or (name != "base_ang_vel" and term.get("scale") is not None))
           for name, term in policy_terms.items()):
        raise ValueError("Policy observation scale/clip not reproduced by the expert loader")
    joint_pos_cfg = env["scene"]["robot"]["init_state"]["joint_pos"]
    default = [
        next((float(v) for pattern, v in joint_pos_cfg.items() if re.fullmatch(pattern, name)), 0.0)
        for name in motion_lib._motion_joints
    ]
    episode = load_expert_trajectories_from_motion_lib(
        motion_lib, dt, torch.tensor([default]),
        base_ang_vel_scale=float(policy_terms["base_ang_vel"].get("scale") or 1.0),
    )[0][0]
    observation = episode["observation"]
    if observation["state"].shape[-1] != dims["state"] or observation["privileged_state"].shape[-1] != dims["privileged_state"]:
        raise ValueError("Expert observation dims do not match the checkpoint normalizer")
    start = args.start_frame
    with torch.inference_mode():
        normalized = normalizer.normalize_dict({k: observation[k][start + 1:] for k in ("state", "privileged_state")})
        b_in = torch.cat([normalized[k] for k in agent["privileged_obs_keys"]], -1)
        z = project_z(torch.cat([backward(chunk) for chunk in b_in.split(256)]))
    frames = len(z)
    heading = episode["heading_forward_xy"][start : start + frames]
    anchor = heading[:1].expand_as(heading)
    valid = np.ones((frames, 1), dtype=np.uint8)
    members = {
        "z": z, "heading_relative_xy": world_to_yaw(heading, anchor), "heading_valid": valid,
        "reference_velocity_heading0_xy": world_to_yaw(episode["reference_velocity_world_xy"][start : start + frames], anchor),
        "velocity_valid": valid, "dt": np.array([dt], dtype=np.float64),
    }
    out.mkdir(parents=True, exist_ok=True)
    context_path = out / "context_1.npz"
    # Plain stored zip without zip64 (cnpy cannot read numpy's forced zip64) and fixed timestamps.
    with zipfile.ZipFile(context_path, "w", allowZip64=False) as archive:
        for name, value in members.items():
            array = value.float().numpy() if torch.is_tensor(value) else value
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite context {name}")
            handle = io.BytesIO()
            np.save(handle, np.ascontiguousarray(array))
            archive.writestr(zipfile.ZipInfo(f"{name}.npy"), handle.getvalue())

    # ---- actor (training exporter) and metric-depth encoder
    exporter = SimpleNamespace(
        visual_enabled=True, actor_keys=tuple(agent["actor_obs_keys"]), _actor=actor,
        _obs_normalizer=normalizer, _backward_map=backward, depth_shape=depth_shape,
        obs_dims=dims, z_dim=agent["z_dim"], action_dim=action_dim, device="cpu",
        privileged_keys=tuple(agent["privileged_obs_keys"]),
        cfg=SimpleNamespace(action_rescale=float(state["action_rescale"]), norm_z=agent["norm_z"]),
    )
    actor_path = Path(FbCprAuxAgent.export_visual_onnx(exporter, str(out), actor_filename="pbfm_actor.onnx")["actor"])
    encoder_path = out / "pbfm_encoder.onnx"
    pipeline = env["scene"]["camera"]["noise_pipeline"]
    low, high = (float(v) for v in pipeline["depth_normalization"]["depth_range"])
    torch.onnx.export(
        MetricDepthEncoder(DepthEncoderOnnxExporter(actor.depth_encoder), low, high).eval(),
        (torch.full((1, *depth_shape), high),), str(encoder_path),
        opset_version=18, input_names=["depth_m"], output_names=["depth_latent"],
        dynamic_axes={"depth_m": {0: "batch"}, "depth_latent": {0: "batch"}},
        external_data=False, do_constant_folding=True, dynamo=False,
    )

    # ---- ONNX contract and parity against the torch modules
    sess = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    actor_keys = list(agent["actor_obs_keys"])
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
            obs = {k: torch.randn((1, dims[k]), generator=rng) for k in actor_keys}
            obs["heading"] = F.normalize(obs["heading"], dim=-1)
            obs["velocity_command"][:, 2] = 1.0
            z_sample = project_z(torch.randn((1, agent["z_dim"]), generator=rng))
            action = actor.mean_from_depth_latent(normalizer.normalize_dict(obs), z_sample, latent)
            feeds = {k: v.numpy() for k, v in obs.items()} | {"z": z_sample.numpy(), "depth_latent": latent.numpy()}
            parity.append({
                "encoder": float(np.abs(enc.run(None, {"depth_m": depth_m.numpy()})[0] - latent.numpy()).max()),
                "action": float(np.abs(sess.run(None, feeds)[0] - action.numpy()).max()),
            })
    print(f"[INFO]: ONNX parity {parity}", flush=True)
    if any(max(item.values()) > 1e-4 for item in parity):
        raise RuntimeError(f"ONNX parity failed: {parity}")

    meta = {
        "checkpoint": str(checkpoint),
        "global_step": int(state["env_iterations"] * state["num_envs"] * state["world_size"]),
        "motion": args.motion,
        "start_frame": start,
        "frames": frames,
        "dt": dt,
        "action_rescale": float(state["action_rescale"]),
        "action_scale": float(env["actions"]["joint_pos"]["scale"]),
        "z": "single-frame project_z(B(next_obs[t])); training seq_length average not applied",
        "depth_shape": list(depth_shape),
        "depth_crop_up_down_left_right": list(pipeline["crop_and_resize"]["crop_region"]),
        "depth_range_m": [low, high],
        "encoder_input": "depth_m float32 meters; clip/normalize/uint8 quantize in graph",
        "default_joint_pos": default,
        "parity": parity,
        "sha256": {p.name: sha256(p) for p in (actor_path, encoder_path, context_path)},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[INFO]: rp1 deployment assets exported to {out}\n{json.dumps(meta['sha256'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
