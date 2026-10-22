"""Shared CPU-export helpers for ``export_*_deploy.py`` scripts."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch
import yaml

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


def log_root(*parts: str) -> Path:
    return Path(os.environ.get("ROBO_TRAIN_LOG_ROOT", DEFAULT_LOG_ROOT)).joinpath(*parts)


def resolve_run(root: Path, load_run: str | None) -> Path:
    if load_run is None:
        candidates = sorted(
            (path for path in root.iterdir() if (path / "params/agent.yaml").is_file()),
            key=lambda path: path.name,
        )
        if not candidates:
            raise FileNotFoundError(f"No runs found below {root}")
        return candidates[-1]
    candidate = Path(load_run)
    return candidate if candidate.is_dir() else root / load_run


def resolve_checkpoint(run: Path, checkpoint: str | None) -> Path:
    if checkpoint is not None:
        candidate = Path(checkpoint)
        return candidate if candidate.is_file() else run / checkpoint
    files = [path for path in run.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", path.name)]
    if not files:
        raise FileNotFoundError(f"No model checkpoint found in {run}")
    return max(files, key=lambda path: int(path.stem.split("_")[-1]))


def qualify_distribution(cfg: dict | None) -> dict | None:
    if cfg is None:
        return None
    name = str(cfg.get("class_name", ""))
    if name and "." not in name and ":" not in name:
        cfg = dict(cfg)
        cfg["class_name"] = f"rsl_rl.modules.distribution:{name}"
    return cfg


def actor_kwargs(agent_cfg: dict) -> dict:
    cfg = dict(agent_cfg["actor"])
    cfg.pop("class_name", None)
    if "distribution_cfg" in cfg:
        cfg["distribution_cfg"] = qualify_distribution(cfg["distribution_cfg"])
    return cfg


def mlp_io_dims(weights: dict, prefix: str = "mlp") -> tuple[int, int]:
    keys = [key for key in weights if key.startswith(f"{prefix}.") and key.endswith(".weight")]
    first = min(int(key.split(".")[1]) for key in keys)
    last = max(int(key.split(".")[1]) for key in keys)
    return int(weights[f"{prefix}.{first}.weight"].shape[1]), int(weights[f"{prefix}.{last}.weight"].shape[0])


def onnx_export(module, dummy, path: Path, input_names: list[str], output_names: list[str]) -> None:
    module.to("cpu").eval()
    torch.onnx.export(
        module,
        dummy,
        path,
        export_params=True,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        external_data=False,
        do_constant_folding=True,
        dynamo=False,
    )
    sidecar = path.with_name(path.name + ".data")
    if sidecar.is_file():
        import onnx

        onnx.save(onnx.load(str(path)), str(path))
        sidecar.unlink()
