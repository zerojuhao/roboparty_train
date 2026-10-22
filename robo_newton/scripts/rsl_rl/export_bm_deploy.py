#!/usr/bin/env python3
"""Export a BeyondMimic checkpoint to deployable ONNX on CPU, without Isaac Lab.

Writes ``policy.onnx`` (obs -> actions), ``policy.pt``, and ``export_meta.json``.
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
from rsl_rl.models.mlp_model import MLPModel


def build_actor(agent_cfg: dict, weights: dict) -> MLPModel:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="rpo_beyondmimic", help="Log folder under logs/rsl_rl.")
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

    model = build_actor(agent_cfg, state["actor_state_dict"])
    exporter = model.as_onnx()
    onnx_path = out / "policy.onnx"
    jit_path = out / "policy.pt"
    onnx_export(exporter, exporter.get_dummy_inputs(), onnx_path, list(exporter.input_names), list(exporter.output_names))
    torch.jit.script(model.as_jit().to("cpu")).save(str(jit_path))

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = torch.Generator().manual_seed(0)
    errors = []
    with torch.inference_mode():
        for _ in range(4):
            obs = torch.randn((1, model.obs_dim), generator=rng)
            torch_action = exporter(obs)
            onnx_action = session.run(None, {"obs": obs.numpy()})[0]
            errors.append(float(np.abs(onnx_action - torch_action.numpy()).max()))
    if max(errors) > 1.0e-4:
        raise RuntimeError(f"BeyondMimic ONNX parity failed: {errors}")
    print(f"[INFO] ONNX parity max errors: {errors}", flush=True)

    control_dt = float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"])
    meta = {
        "format": "bm_deploy_v1",
        "checkpoint": str(checkpoint),
        "iteration": int(state["iter"]),
        "obs_dim": int(model.obs_dim),
        "action_dim": int(model.mlp[-1].out_features),
        "actor_file": onnx_path.name,
        "jit_file": jit_path.name,
        "dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "parity_action_max_abs": errors,
        "sha256": {path.name: sha256(path) for path in (onnx_path, jit_path)},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[INFO] deployment assets written to {out}", flush=True)


if __name__ == "__main__":
    main()
