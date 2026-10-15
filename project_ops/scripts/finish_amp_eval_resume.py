"""Finish an AMP run, repeat its saved benchmark, then resume with two new weights."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
from sim2sim_rp1_amp import read_config


def training_process_alive(pid, run_name):
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes()
    except FileNotFoundError:
        return False
    return b"torch.distributed.run" in command and run_name.encode() in command


def build_commands(source_run, checkpoint, reference, output, next_run_name, iterations):
    evaluation = [
        sys.executable, str(ROOT / "robo_newton/scripts/mujoco/sim2sim_rp1_amp.py"),
        "--checkpoint", str(checkpoint), "--mujoco-xml", reference["mujoco_xml"],
        "--output", str(output), "--max-speed", "4",
        "--ramp-seconds", str(reference["ramp_seconds_override"]),
        "--deceleration-seconds", str(reference["deceleration_seconds"]),
        "--standing-seconds", str(reference["standing_seconds"]),
        "--duration", str(reference["duration_s"]),
        "--fps", str(reference["video_fps"]),
        "--width", str(reference["resolution"][0]),
        "--height", str(reference["resolution"][1]),
        "--initial-height", str(reference["initial_root_height_m"]), "--threads", "2",
        "--camera-distance", "2.15",
    ]
    training = [
        "bash", str(ROOT / "project_ops/train_amp_proprio.sh"),
        "--resume", "--load_run", source_run.name, "--checkpoint", checkpoint.name,
        "--run_name", next_run_name, "--max_iterations", str(iterations),
        "--resume_learning_rate", "5e-5",
        "agent.algorithm.learning_rate=5e-5", "agent.algorithm.schedule=fixed",
        "env.rewards.track_lin_vel_xy_exp.weight=1.2",
        "env.rewards.lateral_command_offset.weight=-0.2",
        "env.rewards.lateral_command_offset.params.std=0.5",
        "env.rewards.feet_slide.weight=-0.5",
        "env.rewards.feet_close_y_gauss.params.threshold=0.18",
        "env.rewards.feet_close_y_gauss.weight=-0.5",
        "env.rewards.knee_close_y_gauss.params.threshold=0.18",
        "env.rewards.knee_close_y_gauss.weight=-0.5",
        "env.rewards.sound_suppression.weight=-5e-4",
    ]
    return evaluation, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--final-iteration", type=int, default=24499)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--next-run-name", required=True)
    parser.add_argument("--next-session", default="amp_speed_drift_finetune")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.source_run.resolve()
    checkpoint = source / f"model_{args.final_iteration}.pt"
    output = args.output.resolve()
    reference = json.loads(args.reference.read_text())
    if reference["command"] != "linear ramp 0 to 4 m/s, cruise, brake to zero, stand":
        raise ValueError("Reference must be the saved 0-to-4-to-0 benchmark")
    old_cfg = read_config(source / "params/env.yaml")
    old_agent = read_config(source / "params/agent.yaml")
    for name, weight in (("track_lin_vel_xy_exp", 0.8), ("lateral_command_offset", -0.1)):
        if old_cfg["rewards"][name]["weight"] != weight:
            raise ValueError(f"Unexpected source reward weight for {name}")
    evaluation, training = build_commands(
        source, checkpoint, reference, output, args.next_run_name, args.iterations,
    )
    if args.dry_run:
        print(json.dumps({"evaluation": evaluation, "training": training}, indent=2))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    # A second invocation must not launch a duplicate eight-GPU job.
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output.with_suffix(".workflow.json")

        def status(phase, **details):
            result = {"phase": phase, "checkpoint": str(checkpoint),
                      "next_run": args.next_run_name, **details}
            status_path.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

        try:
            if output.exists() or subprocess.run(
                ["tmux", "has-session", "-t", args.next_session], capture_output=True,
            ).returncode == 0:
                raise FileExistsError("Benchmark output or next training session already exists")
            status("waiting_for_source_training")
            deadline = time.monotonic() + 7200
            while training_process_alive(args.source_pid, old_agent["run_name"]):
                if time.monotonic() > deadline:
                    raise TimeoutError("Source training did not finish within two hours")
                time.sleep(10)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["iter"] != args.final_iteration:
                raise ValueError("Final checkpoint iteration does not match")
            status("evaluating_final_checkpoint", source_curriculum=saved.get("curriculum_state"))
            environment = os.environ.copy()
            environment.update(MUJOCO_GL="egl", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
            subprocess.run(evaluation, cwd=ROOT, env=environment, check=True)
            current = json.loads(output.with_suffix(".json").read_text())
            compared = ("duration_s", "ramp_seconds_override", "deceleration_seconds",
                        "standing_seconds", "video_fps", "resolution", "initial_root_height_m", "mujoco_xml")
            if any(current[key] != reference[key] for key in compared):
                raise ValueError("Benchmark conditions differ from the saved reference")
            comparison = {"reference": reference, "final_checkpoint": current}
            output.with_suffix(".comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

            launch_log = source.parents[2] / "launch" / f"{args.next_run_name}.log"
            status("starting_new_training", video=str(output), launch_log=str(launch_log))
            subprocess.run([
                "tmux", "new-session", "-d", "-s", args.next_session, "-c", str(ROOT),
                shlex.join(training) + " > " + shlex.quote(str(launch_log)) + " 2>&1",
            ], check=True)
            deadline = time.monotonic() + 300
            # The next multiple-of-500 checkpoint proves rollout and learning
            # actually started, rather than merely accepting the tmux command.
            first_saved_iteration = ((args.final_iteration + 499) // 500) * 500
            while time.monotonic() < deadline:
                runs = list(source.parent.glob(f"*_{args.next_run_name}"))
                if len(runs) == 1 and (runs[0] / f"model_{first_saved_iteration}.pt").exists():
                    try:
                        resumed = torch.load(runs[0] / f"model_{first_saved_iteration}.pt",
                                             map_location="cpu", weights_only=False)
                    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
                        time.sleep(5)
                        continue
                    cfg = read_config(runs[0] / "params/env.yaml")
                    assert cfg["rewards"]["track_lin_vel_xy_exp"]["weight"] == 1.2
                    assert cfg["rewards"]["lateral_command_offset"]["weight"] == -0.2
                    for name in ("feet_slide", "feet_close_y_gauss", "knee_close_y_gauss",
                                 "sound_suppression", "lean_orientation_l2", "high_speed_double_flight"):
                        assert cfg["rewards"][name] == old_cfg["rewards"][name], name
                    assert all(g["lr"] == 5e-5 for g in resumed["optimizer_state_dict"]["param_groups"])
                    assert resumed.get("curriculum_state") == saved.get("curriculum_state")
                    status("new_training_verified", run_dir=str(runs[0]), video=str(output),
                           comparison=str(output.with_suffix(".comparison.json")),
                           curriculum=resumed.get("curriculum_state"))
                    return
                time.sleep(5)
            raise TimeoutError("New training did not produce its initial checkpoint within five minutes")
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
