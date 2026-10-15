"""Queue an unchanged AMP continuation after an existing run finishes."""

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from finish_amp_eval_resume import training_process_alive
from watch_amp_checkpoint_evaluations import checkpoint_ready


ROOT = Path(__file__).resolve().parents[2]


def continuation_command(source_command, source_run, checkpoint, run_name, iterations):
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    script = "robo_newton/scripts/rsl_rl/train.py"
    if script not in source_command:
        raise ValueError("Source must use the AMP training entry point")
    original = source_command[source_command.index(script) + 1:]
    controlled = {"--run_name", "--load_run", "--checkpoint", "--max_iterations"}
    preserved = []
    index = 0
    while index < len(original):
        argument = original[index]
        if argument in controlled:
            index += 2
            continue
        if argument.split("=", 1)[0] in controlled:
            index += 1
            continue
        preserved.append(argument)
        index += 1
    if "--resume" not in preserved:
        preserved.append("--resume")
    return [
        "bash", str(ROOT / "project_ops/train_amp_proprio.sh"), *preserved,
        "--load_run", source_run, "--checkpoint", checkpoint,
        "--run_name", run_name, "--max_iterations", str(iterations),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pid", type=int, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-run-name", required=True)
    parser.add_argument("--source-final-iteration", type=int, required=True)
    parser.add_argument("--next-run-name", required=True)
    parser.add_argument("--next-session", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--xml", type=Path, required=True)
    args = parser.parse_args()
    if not training_process_alive(args.source_pid, args.source_run_name):
        raise ValueError("Source PID does not match the active AMP run")
    command = Path(f"/proc/{args.source_pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    source = args.source_run.resolve()
    checkpoint = source / f"model_{args.source_final_iteration}.pt"
    training = continuation_command(
        command, source.name, checkpoint.name, args.next_run_name, args.iterations,
    )
    launch_root = source.parents[2] / "launch"
    status_path = launch_root / f"{args.next_run_name}.workflow.json"
    output = source.parents[3] / "sim2sim_checks" / args.next_run_name
    final = args.source_final_iteration + args.iterations - 1
    evaluation = [
        sys.executable, str(ROOT / "project_ops/scripts/watch_amp_checkpoint_evaluations.py"),
        "--run-root", str(source.parent), "--run-name", args.next_run_name,
        "--baseline", str(checkpoint), "--final-iteration", str(final),
        "--xml", str(args.xml.resolve()), "--output-dir", str(output),
        "--turn-tests", "--timeout-hours", "6",
    ]

    def status(phase, **details):
        data = {"phase": phase, "source_checkpoint": str(checkpoint),
                "next_run": args.next_run_name, "iterations": args.iterations,
                "final_iteration": final, **details}
        status_path.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)

    with status_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if subprocess.run(["tmux", "has-session", "-t", args.next_session], capture_output=True).returncode == 0:
                raise FileExistsError("Continuation tmux session already exists")
            status("waiting_for_source_training")
            deadline = time.monotonic() + 6 * 3600
            while training_process_alive(args.source_pid, args.source_run_name):
                if time.monotonic() > deadline:
                    raise TimeoutError("Source run did not finish within six hours")
                time.sleep(10)
            if not checkpoint.exists():
                raise FileNotFoundError("Source stopped without its final checkpoint")
            while not checkpoint_ready(checkpoint):
                if time.monotonic() > deadline:
                    raise TimeoutError("Source final checkpoint is not readable")
                time.sleep(3)
            status("starting_continuation")
            log = launch_root / f"{args.next_run_name}.log"
            subprocess.run([
                "tmux", "new-session", "-d", "-s", args.next_session, "-c", str(ROOT),
                shlex.join(training) + " > " + shlex.quote(str(log)) + " 2>&1",
            ], check=True)
            evaluation_log = launch_root / f"{args.next_run_name}_eval.log"
            subprocess.run([
                "tmux", "new-window", "-d", "-t", args.next_session, "-n", "checkpoint_eval",
                "-c", str(ROOT), shlex.join(evaluation) + " > " + shlex.quote(str(evaluation_log)) + " 2>&1",
            ], check=True)
            status("continuation_launched", session=args.next_session, evaluation_output=str(output))
        except Exception as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
