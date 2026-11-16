#!/usr/bin/env python3
"""Export an RPO / attention-encoder checkpoint to ONNX on CPU, without Isaac Lab.

MLP policies write ``policy.onnx`` (obs -> actions) and ``policy.pt``.
Attention policies write a single ``obs`` vector ``[policy | height_scan] -> actions``.

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
from torch import nn

from export_deploy_common import (
    actor_kwargs,
    log_root,
    mlp_io_dims,
    onnx_export,
    read_yaml,
    resolve_checkpoint,
    resolve_run,
    sha256,
)
from rsl_rl.models.attention_model import AttentionModel
from rsl_rl.models.mlp_model import MLPModel


class OnnxAttentionPolicy(nn.Module):
    """Single concatenated ``obs`` matching sim2sim_rpo_attn_enc."""

    def __init__(self, model: AttentionModel):
        super().__init__()
        self.obs_normalizer = model.obs_normalizer
        self.obs_encoder = model.obs_encoder
        self.estimator = model.estimator
        self.encoder = model.encoder
        self.mlp = model.mlp
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.obs_dim = model.obs_dim
        self.actor_step_dim = model.actor_step_dim
        self.enable_obs_encoder = model.enable_obs_encoder
        self.enable_critic_estimation = model.enable_critic_estimation
        self.map_elems = int(model.encoder.length * model.encoder.width)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = self.obs_normalizer(obs[:, : self.obs_dim])
        perception = obs[:, self.obs_dim :]
        last = proprio[:, -self.actor_step_dim :]
        parts = [last]
        esti_in = proprio
        if self.obs_encoder is not None:
            latent = self.obs_encoder(proprio)
            parts.append(latent)
            esti_in = latent
        if self.estimator is not None:
            parts.append(self.estimator(esti_in))
        enc_input = torch.cat(parts, dim=-1)
        embedding = self.encoder(enc_input, perception)
        if self.enable_critic_estimation or self.enable_obs_encoder:
            features = torch.cat((enc_input, embedding), dim=-1)
        else:
            features = torch.cat((proprio, embedding), dim=-1)
        return self.deterministic_output(self.mlp(features))


def _policy_dim(env_cfg: dict, actor_cfg: dict, weights: dict) -> int:
    if "obs_normalizer._mean" in weights:
        return int(weights["obs_normalizer._mean"].shape[-1])
    if "obs_encoder.0.weight" in weights:
        return int(weights["obs_encoder.0.weight"].shape[1])
    single = env_cfg.get("observation_space")
    history = int(actor_cfg.get("actor_history_length") or 1)
    if single is not None:
        return int(single) * history
    return mlp_io_dims(weights)[0]


def build_mlp(agent_cfg: dict, weights: dict) -> MLPModel:
    cfg = actor_kwargs(agent_cfg)
    if "obs_normalizer._mean" in weights:
        obs_dim = int(weights["obs_normalizer._mean"].shape[-1])
        action_dim = mlp_io_dims(weights)[1]
    else:
        obs_dim, action_dim = mlp_io_dims(weights)
    obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    model = MLPModel(obs, {"actor": ["policy"]}, "actor", action_dim, **cfg)
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)


def build_attention(agent_cfg: dict, env_cfg: dict, weights: dict) -> AttentionModel:
    cfg = actor_kwargs(agent_cfg)
    policy_dim = _policy_dim(env_cfg, cfg, weights)
    history = int(cfg.get("critic_history_length") or cfg.get("actor_history_length") or 5)
    slice_end = max(cfg.get("estimation_slice") or [80]) + 1
    critic_dim = int(env_cfg.get("state_space") or slice_end) * history
    length, width = cfg.get("map_size") or (17, 11)
    obs = TensorDict(
        {
            "policy": torch.zeros(1, policy_dim),
            "critic": torch.zeros(1, critic_dim),
            "perception_a": torch.zeros(1, int(length) * int(width)),
        },
        batch_size=[1],
    )
    model = AttentionModel(
        obs,
        {"actor": ["policy"], "critic": ["critic"]},
        "actor",
        env_cfg.get("action_space") or mlp_io_dims(weights)[1],
        **cfg,
    )
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False)


def export_mlp(model: MLPModel, out) -> tuple:
    exporter = model.as_onnx()
    onnx_path = out / "policy.onnx"
    jit_path = out / "policy.pt"
    onnx_export(exporter, exporter.get_dummy_inputs(), onnx_path, list(exporter.input_names), list(exporter.output_names))
    torch.jit.script(model.as_jit().to("cpu")).save(str(jit_path))
    return onnx_path, jit_path, list(exporter.input_names), int(model.obs_dim), int(model.mlp[-1].out_features)


def export_attention(model: AttentionModel, out) -> tuple:
    exporter = OnnxAttentionPolicy(model).eval()
    onnx_path = out / "policy.onnx"
    dummy = torch.zeros(1, exporter.obs_dim + exporter.map_elems)
    onnx_export(exporter, (dummy,), onnx_path, ["obs"], ["actions"])
    jit_path = out / "policy.pt"
    try:
        torch.jit.script(exporter.to("cpu")).save(str(jit_path))
    except Exception as error:
        print(f"[WARN] TorchScript export skipped: {error}", flush=True)
        jit_path = None
    return onnx_path, jit_path, ["obs"], int(dummy.shape[-1]), int(model.mlp[-1].out_features)


def parity(exporter, onnx_path, dummy_fn, samples: int = 4) -> list[float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = torch.Generator().manual_seed(0)
    errors = []
    with torch.inference_mode():
        for _ in range(samples):
            obs = dummy_fn(rng)
            torch_action = exporter(obs)
            name = session.get_inputs()[0].name
            onnx_action = session.run(None, {name: obs.numpy()})[0]
            errors.append(float(np.abs(onnx_action - torch_action.numpy()).max()))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="rpo_flat", help="Log folder under logs/rsl_rl.")
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

    class_name = str(agent_cfg["actor"]["class_name"])
    weights = state["actor_state_dict"]
    if class_name.endswith("AttentionModel"):
        model = build_attention(agent_cfg, env_cfg, weights)
        exporter = OnnxAttentionPolicy(model).eval()
        onnx_path, jit_path, inputs, obs_dim, action_dim = export_attention(model, out)
        dummy_fn = lambda rng: torch.randn((1, exporter.obs_dim + exporter.map_elems), generator=rng)
    else:
        model = build_mlp(agent_cfg, weights)
        exporter = model.as_onnx().eval()
        onnx_path, jit_path, inputs, obs_dim, action_dim = export_mlp(model, out)
        dummy_fn = lambda rng: torch.randn((1, model.obs_dim), generator=rng)

    errors = parity(exporter, onnx_path, dummy_fn)
    if max(errors) > 1.0e-4:
        raise RuntimeError(f"RPO ONNX parity failed: {errors}")
    print(f"[INFO] ONNX parity max errors: {errors}", flush=True)

    control_dt = float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"])
    meta = {
        "format": "rpo_deploy_v1",
        "checkpoint": str(checkpoint),
        "iteration": int(state["iter"]),
        "actor_class": class_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "actor_inputs": {name: obs_dim for name in inputs},
        "actor_file": onnx_path.name,
        "jit_file": None if jit_path is None else jit_path.name,
        "dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "parity_action_max_abs": errors,
        "sha256": {path.name: sha256(path) for path in ((onnx_path,) if jit_path is None else (onnx_path, jit_path))},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[INFO] deployment assets written to {out}", flush=True)


if __name__ == "__main__":
    main()
