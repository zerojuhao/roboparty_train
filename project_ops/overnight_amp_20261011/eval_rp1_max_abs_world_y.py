#!/usr/bin/env python3
"""CPU MuJoCo fixed-speed eval producing max_abs_world_y_m (same stack as sim2sim_rp1_amp)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

REPO = Path("/home/zym/roboparty_train")
sys.path.insert(0, str(REPO / "robo_newton" / "scripts" / "mujoco"))

from sim2sim_rp1_amp import AMPPolicy, DEFAULT_MJCF, Simulation, read_config  # noqa: E402


def run_fixed_speed(policy, config, mjcf, vx: float, duration: float = 20.0):
    environment = Simulation(mjcf, config, camera_distance=2.15)
    try:
        records = []
        states = []
        fallen_at = None
        for step in range(round(duration * 50)):
            seconds = step / 50.0
            command = np.array([vx, 0.0, 0.0], dtype=np.float32)
            observation, _ = environment.observe(command)
            action, _ = policy(observation)
            environment.set_action(action)
            for _ in range(4):
                environment.advance()
            state = environment.measure()
            states.append(state)
            if fallen_at is None and state["root_position_w"][2] < 0.45:
                fallen_at = seconds
            records.append((seconds, vx, float(state["body_velocity"][0])))
        values = np.asarray(records)
        positions = np.asarray([s["root_position_w"] for s in states])
        heading = np.unwrap(np.asarray([s["heading_rad"] for s in states]))
        selected = values[:, 0] >= 1.0
        return {
            "vx": float(vx),
            "duration_s": float(duration),
            "fallen_at_s": fallen_at,
            "max_abs_world_y_m": float(np.abs(positions[:, 1]).max()),
            "heading_change_deg": float(np.rad2deg(heading[-1] - heading[0])),
            "max_abs_heading_deg": float(np.rad2deg(np.abs(heading - heading[0]).max())),
            "mean_actual_vx_mps": float(values[selected, 2].mean()) if selected.any() else None,
            "vx_tracking_mae_mps": float(np.abs(values[selected, 2] - values[selected, 1]).mean())
            if selected.any()
            else None,
            "source": "mujoco_sim2sim_rp1_amp_fixed_speed_no_render",
        }
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 1.5, 2.0, 4.0])
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve(strict=True)
    config = read_config(checkpoint.parent / "params" / "env.yaml")
    policy = AMPPolicy(checkpoint)
    results = []
    for vx in args.speeds:
        row = run_fixed_speed(policy, config, DEFAULT_MJCF.resolve(), vx, args.duration)
        row["checkpoint"] = str(checkpoint)
        row["iteration"] = int(policy.iteration)
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
        results.append(row)
        print(json.dumps({k: row[k] for k in ("iteration", "vx", "max_abs_world_y_m", "heading_change_deg", "fallen_at_s")}), flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "iteration": int(policy.iteration),
        "metric_confirmed_sim2sim": True,
        "eval_script": "sim2sim_rp1_amp.py / evaluate path (CPU no-render wrapper)",
        "results": results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"WROTE {args.output_json}")


if __name__ == "__main__":
    main()
