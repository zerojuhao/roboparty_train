#!/usr/bin/env python3
"""Export an AMP checkpoint to deployable ONNX/JIT on CPU, without Isaac Lab.

Reconstructs the official ``AMPModel.as_onnx()`` / ``as_jit()`` graphs from the
run's ``params/agent.yaml`` and ``actor_state_dict``, then writes:

    policy.onnx   obs (1, obs_dim) -> actions (1, action_dim)
    policy.pt     TorchScript of the same graph
    export_meta.json

The exporter does not start Isaac Lab or consume a GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from rsl_rl.models.amp_model import AMPModel

DEFAULT_LOG_ROOT = "/data/zym/roboparty_train/logs/rsl_rl"


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
            raise FileNotFoundError(f"No AMP runs found below {log_root}")
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


def build_actor(agent_cfg: dict, weights: dict) -> AMPModel:
    obs_dim = int(weights["obs_normalizer._mean"].shape[-1])
    mlp_weight_keys = [key for key in weights if key.startswith("mlp.") and key.endswith(".weight")]
    last_layer = max(int(key.split(".")[1]) for key in mlp_weight_keys)
    action_dim = int(weights[f"mlp.{last_layer}.weight"].shape[0])
    actor_cfg = copy.deepcopy(agent_cfg["actor"])
    actor_cfg.pop("class_name", None)
    dist_cfg = actor_cfg.get("distribution_cfg")
    if dist_cfg is not None and "." not in str(dist_cfg.get("class_name", "")) and ":" not in str(dist_cfg.get("class_name", "")):
        dist_cfg["class_name"] = f"rsl_rl.modules.distribution:{dist_cfg['class_name']}"
    if actor_cfg.get("enable_velocity_estimation") and actor_cfg.get("velocity_target_indices") is None:
        actor_cfg["velocity_target_indices"] = (0, 1)
    obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    model = AMPModel(obs, {"actor": ["policy"]}, "actor", action_dim, **actor_cfg)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    return model


def export_onnx(model: AMPModel, path: Path) -> None:
    exporter = model.as_onnx()
    exporter.to("cpu").eval()
    torch.onnx.export(
        exporter,
        exporter.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=list(exporter.input_names),
        output_names=list(exporter.output_names),
        external_data=False,
        do_constant_folding=True,
        dynamo=False,
    )
    sidecar = path.with_name(path.name + ".data")
    if sidecar.is_file():
        import onnx

        onnx.save(onnx.load(str(path)), str(path))
        sidecar.unlink()


def export_jit(model: AMPModel, path: Path) -> None:
    traced = torch.jit.script(model.as_jit().to("cpu"))
    traced.save(str(path))


def parity_errors(model: AMPModel, onnx_path: Path, samples: int = 4) -> list[float]:
    import onnxruntime as ort

    exporter = model.as_onnx().eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = torch.Generator().manual_seed(0)
    errors = []
    with torch.inference_mode():
        for _ in range(samples):
            obs = torch.randn((1, model.obs_dim), generator=rng)
            torch_action = exporter(obs)
            onnx_action = session.run(None, {"obs": obs.numpy()})[0]
            errors.append(float(np.abs(onnx_action - torch_action.numpy()).max()))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", default="rp1_amp", help="Log folder under logs/rsl_rl.")
    parser.add_argument("--load_run", default=None, help="Run directory or run name below the AMP log root.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or model_N.pt name.")
    args = parser.parse_args()
    torch.set_num_threads(4)

    log_root = Path(os.environ.get("ROBO_TRAIN_LOG_ROOT", DEFAULT_LOG_ROOT)) / args.experiment
    run = resolve_run(log_root, args.load_run)
    checkpoint = resolve_checkpoint(run, args.checkpoint).resolve()
    agent_cfg = read_yaml(run / "params/agent.yaml")
    env_cfg = read_yaml(run / "params/env.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    out = run / "exported"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {checkpoint} -> {out}", flush=True)

    model = build_actor(agent_cfg, state["actor_state_dict"])
    onnx_path = out / "policy.onnx"
    jit_path = out / "policy.pt"
    export_onnx(model, onnx_path)
    export_jit(model, jit_path)

    errors = parity_errors(model, onnx_path)
    if max(errors) > 1.0e-4:
        raise RuntimeError(f"AMP ONNX parity failed: {errors}")
    print(f"[INFO] ONNX parity max errors: {errors}", flush=True)

    control_dt = float(env_cfg["sim"]["dt"]) * int(env_cfg["decimation"])
    meta = {
        "format": "amp_deploy_v1",
        "checkpoint": str(checkpoint),
        "iteration": int(state["iter"]),
        "obs_dim": int(model.obs_dim),
        "action_dim": int(model.mlp[-1].out_features),
        "actor_inputs": {"obs": int(model.obs_dim)},
        "actor_output": "actions",
        "history_length": env_cfg["observations"]["policy"].get("history_length"),
        "action_scale": float(env_cfg["actions"]["joint_pos"]["scale"]),
        "dt": control_dt,
        "control_hz": 1.0 / control_dt,
        "physics_dt": float(env_cfg["sim"]["dt"]),
        "decimation": int(env_cfg["decimation"]),
        "actor_file": onnx_path.name,
        "jit_file": jit_path.name,
        "parity_action_max_abs": errors,
        "sha256": {path.name: sha256(path) for path in (onnx_path, jit_path)},
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[INFO] deployment assets written to {out}", flush=True)


if __name__ == "__main__":
    main()
