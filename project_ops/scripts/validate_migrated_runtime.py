"""Validate a migrated PUFO runtime without starting a training run."""

import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp


@wp.kernel
def add_one(values: wp.array(dtype=float)):
    index = wp.tid()
    values[index] = values[index] + 1.0


def validate_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        rank = torch.distributed.get_rank()
        count = torch.distributed.get_world_size()
        value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(value)
        assert value.item() == count * (count + 1) / 2
        torch.distributed.barrier()
        print(f"Rank {rank}/{count}: NCCL all-reduce PASS", flush=True)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    if args.distributed:
        validate_distributed()
        return
    packages = ("torch", "warp-lang", "newton", "mujoco", "mujoco-warp", "rsl-rl-lib")
    print(json.dumps({
        "python": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in packages},
    }, indent=2), flush=True)
    assert torch.cuda.is_available()
    for index in range(torch.cuda.device_count()):
        values = torch.ones(32, 32, device=f"cuda:{index}")
        torch.testing.assert_close(values @ values.T, torch.full_like(values, 32))
        torch.cuda.synchronize(index)
        print(f"CUDA {index}: {torch.cuda.get_device_name(index)} matmul PASS", flush=True)

    values = torch.ones(64, 64, device="cuda:0")
    compiled = torch.compile(lambda x: x.sin() + x.square())
    torch.testing.assert_close(compiled(values), values.sin() + values.square())
    torch.cuda.synchronize()
    print("Torch compile PASS", flush=True)
    wp.init()
    buffer = wp.zeros(128, dtype=float, device="cuda:0")
    wp.launch(add_one, dim=128, inputs=[buffer], device="cuda:0")
    wp.synchronize_device("cuda:0")
    torch.testing.assert_close(wp.to_torch(buffer), torch.ones(128, device="cuda:0"))
    assert Path(wp.config.kernel_cache_dir).is_relative_to(Path(os.environ["XDG_CACHE_HOME"]))
    print("Warp CUDA kernel and cache path PASS", flush=True)

    from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg
    from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg

    cfg, agent = RP1PUFOEnvCfg(), RP1PUFORunnerCfg()
    assert agent.visual_enabled and agent.estimated_velocity_enabled
    assert agent.prioritization and agent.shared_value_net and agent.shared_context_value_net
    assert tuple(agent.depth_encoder.input_shape) == (8, 18, 36)
    manifest_path = Path(cfg.commands.motion.manifest_path)
    assert manifest_path.is_file(), manifest_path
    asset_path = Path(cfg.scene.robot.spawn.asset_path)
    assert asset_path.is_file(), asset_path
    manifest = json.loads(manifest_path.read_text())
    for clip in manifest["clips"]:
        path = manifest_path.parent / clip["npz"]
        with np.load(path) as motion:
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w"):
                assert np.isfinite(motion[key]).all(), (path, key)
    print(f"PUFO configuration, robot assets and {len(manifest['clips'])} motion files PASS", flush=True)
    print("MIGRATED RUNTIME PASS", flush=True)


if __name__ == "__main__":
    main()
