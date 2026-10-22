#!/usr/bin/env python3
"""Export a Parkour/SSR checkpoint to deployable ONNX on CPU, without Isaac Lab.

Writes the official ``ParkourModel.export_as_onnx`` graphs:

    depth_encoder.onnx   depth_normalized (1, C, H, W) -> depth_latent
    policy_parkour.onnx  concat(proprio, depth_latent) -> actions
    export_meta.json

The exporter does not start Isaac Lab or consume a GPU.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from tensordict import TensorDict

from export_deploy_common import actor_kwargs, log_root, read_yaml, resolve_checkpoint, resolve_run, sha256
from rsl_rl.models.parkour_model import ParkourActorOnnxExporter, ParkourModel

PROPRIO_WIDTHS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
}


def _term_history(term: dict, group: dict) -> int:
    return max(int(term.get("history_length") or group.get("history_length") or 1), 1)


def _action_dim(weights: dict) -> int:
    keys = [key for key in weights if key.startswith("mlp.experts.0.") and key.endswith(".weight")]
    last = max(int(key.split(".")[3]) for key in keys)
    return int(weights[f"mlp.experts.0.{last}.weight"].shape[0])


def _depth_shape(env_cfg: dict) -> tuple[int, int, int]:
    frames = int(env_cfg["observations"]["policy"]["depth_image"]["params"].get("num_output_frames") or 1)
    camera = env_cfg["scene"]["camera"]
    height = int(camera["pattern_cfg"]["height"])
    width = int(camera["pattern_cfg"]["width"])
    up, down, left, right = camera["noise_pipeline"]["crop_and_resize"]["crop_region"]
    return frames, height - int(up) - int(down), width - int(left) - int(right)


def _policy_obs(env_cfg: dict, action_dim: int) -> TensorDict:
    group = env_cfg["observations"]["policy"]
    widths = dict(PROPRIO_WIDTHS)
    widths.update(joint_pos=action_dim, joint_vel=action_dim, actions=action_dim)
    tensors = {}
    for name, term in group.items():
        if not isinstance(term, dict) or "func" not in term or name == "depth_image":
            continue
        width = widths.get(name)
        if width is None:
            raise ValueError(f"Unknown parkour proprio term '{name}'")
        tensors[name] = torch.zeros(1, width * _term_history(term, group))
    channels, height, width = _depth_shape(env_cfg)
    tensors["depth_image"] = torch.zeros(1, channels, height, width, dtype=torch.uint8)
    return TensorDict(tensors, batch_size=[1])


def _critic_obs(weights: dict, actor_cfg: dict) -> TensorDict | None:
    if not actor_cfg.get("enable_ssr_estimation"):
        return None
    keys = [key for key in weights if "foot_height_decoder" in key and key.endswith(".weight")]
    last = max(int(key.split(".")[-2]) for key in keys)
    weight = weights[next(key for key in keys if key.endswith(f".{last}.weight"))]
    foot_dim = int(weight.shape[0] // 2)
    return TensorDict(
        {
            "left_foot_height_map": torch.zeros(1, foot_dim),
            "right_foot_height_map": torch.zeros(1, foot_dim),
        },
        batch_size=[1],
    )


def build_actor(agent_cfg: dict, env_cfg: dict, weights: dict) -> tuple[ParkourModel, TensorDict]:
    cfg = actor_kwargs(agent_cfg)
    action_dim = _action_dim(weights)
    policy = _policy_obs(env_cfg, action_dim)
    obs = TensorDict({"policy": policy}, batch_size=[1])
    critic = _critic_obs(weights, cfg)
    if critic is not None:
        obs["critic"] = critic
    model = ParkourModel(obs, {"actor": ["policy"], "critic": ["critic"]}, "actor", action_dim, **cfg)
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False), obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="rp1_parkour", help="Log folder under logs/rsl_rl.")
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)

    run = resolve_run(log_root(args.experiment), args.load_run)
    checkpoint = resolve_checkpoint(run, args.checkpoint).resolve()
    agent_cfg = read_yaml(run / "params/agent.yaml")
    env_cfg = read_yaml(run / "params/env.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    out = run / "exported"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {checkpoint} -> {out}", flush=True)

    model, obs = build_actor(agent_cfg, env_cfg, state["actor_state_dict"])
    model.export_as_onnx(obs, str(out))
    encoder_path = out / model.encoder_onnx_filename
    actor_path = out / model.actor_onnx_filename

    import onnxruntime as ort

    enc = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    act = ort.InferenceSession(str(actor_path), providers=["CPUExecutionProvider"])
    actor = ParkourActorOnnxExporter(model).eval()
    rng = torch.Generator().manual_seed(0)
    errors = []
    with torch.inference_mode():
        for _ in range(4):
            depth_n = torch.rand((1, *model.image_shape), generator=rng)
            depth_u8 = depth_n.clamp(0, 1).mul(255).round().to(torch.uint8)
            torch_latent = model.depth_encoder(depth_u8)
            proprio = torch.randn((1, model.obs_dim), generator=rng)
            actor_in = torch.cat((proprio, torch_latent), dim=-1)
            errors.append(
                {
                    "encoder": float(np.abs(enc.run(None, {"depth_normalized": depth_n.numpy()})[0] - torch_latent.numpy()).max()),
                    "action": float(np.abs(act.run(None, {"input": actor_in.numpy()})[0] - actor(actor_in).numpy()).max()),
                }
            )
    if any(max(item.values()) > 1.0e-4 for item in errors):
        raise RuntimeError(f"Parkour ONNX parity failed: {errors}")
    print(f"[INFO] ONNX parity max errors: {errors}", flush=True)

    control_dt = float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"])
    meta = {
        "format": "parkour_deploy_v1",
        "checkpoint": str(checkpoint),
        "iteration": int(state["iter"]),
        "obs_dim": int(model.obs_dim),
        "image_shape": list(model.image_shape),
        "depth_latent_dim": int(model.image_dim),
        "action_dim": int(_action_dim(state["actor_state_dict"])),
        "encoder_file": encoder_path.name,
        "actor_file": actor_path.name,
        "dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "parity": errors,
        "sha256": {path.name: sha256(path) for path in (encoder_path, actor_path)},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[INFO] deployment assets written to {out}", flush=True)


if __name__ == "__main__":
    main()
