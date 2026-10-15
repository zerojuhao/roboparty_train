"""Evaluate a baseline and each saved AMP checkpoint without changing training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]


def checkpoint_ready(path):
    if not path.exists() or time.time() - path.stat().st_mtime < 3:
        return False
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
        return False
    if path.name != f"model_{saved['iter']}.pt":
        raise ValueError(f"Checkpoint iteration mismatch: {path}")
    return True


def planned_iterations(start, final, interval=500):
    return sorted(set(range((start // interval + 1) * interval, final + 1, interval)) | {final})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--final-iteration", type=int, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turn-tests", action="store_true")
    parser.add_argument("--step-tests", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=3.0)
    args = parser.parse_args()
    start = int(args.baseline.stem.removeprefix("model_"))
    if args.final_iteration <= start:
        raise ValueError("Final iteration must follow the baseline")
    if args.timeout_hours <= 0:
        raise ValueError("Timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_hours * 3600
    environment = os.environ.copy()
    environment.update(MUJOCO_GL="egl", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES="")
    completed = []

    def status(phase, **extra):
        data = {"phase": phase, "run_name": args.run_name,
                "completed": completed, **extra}
        (args.output_dir / "status.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)

    def wait_for(predicate):
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError("Checkpoint evaluation deadline exceeded")
            time.sleep(10)

    def evaluate(checkpoint, label):
        wait_for(lambda: checkpoint_ready(checkpoint))
        output = args.output_dir / f"{label}_{checkpoint.stem}.json"
        status("evaluating", checkpoint=str(checkpoint))
        if not output.exists():
            command = [
                sys.executable, str(ROOT / "project_ops/scripts/compare_amp_stops.py"),
                "--checkpoint", str(checkpoint), "--xml", str(args.xml),
                "--output", str(output),
            ]
            if args.turn_tests:
                command.append("--turn-tests")
            subprocess.run(command, cwd=ROOT, env=environment, check=True, timeout=300)
        report = json.loads(output.read_text())
        result = report["checkpoints"][0]
        if Path(result["checkpoint"]) != checkpoint.resolve():
            raise ValueError("Existing report belongs to a different checkpoint")
        if args.turn_tests and "turn_trials" not in result:
            raise ValueError("Existing report lacks the requested turning tests")
        step_result = None
        step_output = args.output_dir / f"{label}_{checkpoint.stem}_step.json"
        if args.step_tests:
            status("evaluating_steps", checkpoint=str(checkpoint))
            if not step_output.exists():
                subprocess.run([
                    sys.executable, str(ROOT / "project_ops/scripts/compare_amp_stops.py"),
                    "--checkpoint", str(checkpoint), "--xml", str(args.xml),
                    "--output", str(step_output), "--step-start",
                ], cwd=ROOT, env=environment, check=True, timeout=300)
            step_report = json.loads(step_output.read_text())
            step_result = step_report["checkpoints"][0]
            if Path(step_result["checkpoint"]) != checkpoint.resolve() or not step_report["protocol"].get("raw_step_start"):
                raise ValueError("Existing step report has an incompatible checkpoint or protocol")
        completed.append({"checkpoint": str(checkpoint), "report": str(output),
                          "summary": result["summary"], "turn_trials": result.get("turn_trials", []),
                          "step_report": str(step_output) if step_result else None,
                          "step_summary": step_result["summary"] if step_result else None})
        status("evaluated", checkpoint=str(checkpoint))

    with (args.output_dir / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            evaluate(args.baseline, "baseline")
            status("waiting_for_run")
            wait_for(lambda: bool(list(args.run_root.glob(f"*_{args.run_name}"))))
            runs = list(args.run_root.glob(f"*_{args.run_name}"))
            if len(runs) != 1:
                raise ValueError("Run name must identify exactly one run")
            for iteration in planned_iterations(start, args.final_iteration):
                checkpoint = runs[0] / f"model_{iteration}.pt"
                status("waiting_for_checkpoint", checkpoint=str(checkpoint))
                evaluate(checkpoint, "resumed")
            status("complete")
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
