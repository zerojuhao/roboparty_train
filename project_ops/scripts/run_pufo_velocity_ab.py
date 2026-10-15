"""Run equal-budget B then A continuations from one immutable PUFO snapshot."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, default=9_600_000)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if args.additional_steps <= 0:
        raise ValueError("Additional steps must be positive")
    if (args.output / "experiment.json").exists():
        raise FileExistsError(args.output / "experiment.json")
    from sim2sim_rp1_pufo import read_config
    saved = read_config(checkpoint.parent / "params/agent.yaml")
    saved_env = read_config(checkpoint.parent / "params/env.yaml")
    if "camera" not in saved_env["scene"]:
        raise ValueError("Expected the saved PUFO scene.camera sensor")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    global_envs = payload["world_size"] * payload["num_envs"]
    if (payload["world_size"], payload["num_envs"]) != (8, 1024):
        raise ValueError("This launcher requires the existing 8 x 1024 training layout")
    start_step = payload["env_iterations"] * global_envs
    interval = int(saved["checkpoint_interval_steps"])
    # Stop on a normal publication boundary so both groups have a final checkpoint.
    publication = (start_step // interval + math.ceil(args.additional_steps / interval)) * interval
    end_step = math.ceil(publication / global_envs) * global_envs
    del payload
    experiment_root = checkpoint.parent.parent
    sources = [ROOT / name for name in (
        "rsl_rl/rsl_rl/algorithms/fb_networks.py",
        "rsl_rl/rsl_rl/algorithms/fb_cpr_aux.py",
        "rsl_rl/rsl_rl/runners/fbcpr_runner.py",
        "robo_newton/robolab/tasks/manager_based/ufo/agents/rp1_pufo_agent_cfg.py",
        "robo_newton/robolab/tasks/manager_based/ufo/agents/rp1_pufo_slim512_agent_cfg.py",
    )]
    manifest = dict(checkpoint=str(checkpoint), sha256=checksum(checkpoint),
                    start_step=start_step, end_step=end_step, additional_steps=end_step-start_step,
                    world_size=8, num_envs_per_rank=1024, seed=saved["seed"],
                    source_sha256={str(path): checksum(path) for path in sources},
                    corpus_sha256={str(path.resolve()): checksum(path) for path in args.corpus},
                    order=["B_stopgrad", "A_sharedgrad"], groups={}, status="prepared")
    args.output.mkdir(parents=True, exist_ok=True)

    def publish():
        temporary = args.output / "experiment.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output / "experiment.json")

    publish()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    for group, updates in (("B_stopgrad", False), ("A_sharedgrad", True)):
        if checksum(checkpoint) != manifest["sha256"]:
            raise RuntimeError("Initial checkpoint changed between groups")
        if any(checksum(Path(path)) != value for path, value in manifest["source_sha256"].items()):
            raise RuntimeError("Training source changed between groups; refusing a confounded comparison")
        run_name = f"velocity_ab_{args.output.name}_{group}"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                   "robo_newton/scripts/rsl_rl/train.py", "--task", "RP1-PUFO-SLIM512",
                   "--num_envs", "1024", "--device", "cuda", "--distributed", "--visualizer", "none",
                   "--logger", saved["logger"], "--resume", "--load_run", "^"+re.escape(checkpoint.parent.name)+"$",
                   "--checkpoint", "^"+re.escape(checkpoint.name)+"$", "--run_name", run_name,
                   f"agent.velocity_estimator_updates_encoder={str(updates).lower()}",
                   f"agent.num_env_steps={end_step}", f"agent.seed={saved['seed']}",
                   "env.scene.camera.ray_alignment=base"]
        record = dict(status="starting", command=command, run_name=run_name,
                      velocity_estimator_updates_encoder=updates, started_at=time.time())
        manifest["groups"][group] = record
        manifest["status"] = f"running_{group}"
        publish()
        print(f"Starting {group}: {start_step:,} -> {end_step:,}", flush=True)
        with (args.output / f"{group}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record["torchrun_pid"] = process.pid
            publish()
            returncode = process.wait()
        record.update(returncode=returncode, finished_at=time.time())
        runs = list(experiment_root.glob(f"*_{run_name}"))
        if returncode != 0 or len(runs) != 1 or not (runs[0] / f"model_{end_step}.pt").is_file():
            record["status"] = "failed"
            manifest["status"] = "failed"
            publish()
            raise RuntimeError(f"{group} failed or did not publish its final checkpoint; see its log")
        cfg = read_config(runs[0] / "params/agent.yaml")
        if cfg["velocity_estimator_updates_encoder"] != updates:
            raise RuntimeError("Recorded training configuration has the wrong gradient setting")
        record.update(status="completed", run_dir=str(runs[0]),
                      final_checkpoint=str(runs[0] / f"model_{end_step}.pt"))
        publish()
        if any(checksum(Path(path)) != value for path, value in manifest["corpus_sha256"].items()):
            raise RuntimeError("Fixed evaluation corpus changed")
        report = args.output / f"{group}_fixed_input_velocity.json"
        comparison = [sys.executable, str(ROOT / "project_ops/scripts/compare_pufo_velocity.py"),
                      "--checkpoint", str(checkpoint), record["final_checkpoint"],
                      "--corpus", *manifest["corpus_sha256"], "--output", str(report)]
        with (args.output / f"{group}_fixed_input_velocity.log").open("x") as log:
            subprocess.run(comparison, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=""),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record["fixed_input_velocity_report"] = str(report)
        publish()
    manifest["status"] = "completed"
    publish()
    print("Both continuation groups completed", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
    main()
