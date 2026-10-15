"""Compare deterministic AMP checkpoints over a sweep of abrupt-stop timings."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
from sim2sim_rp1_amp import AMPPolicy, Simulation, MJ_JOINT_NAMES, contact_slip_summary, read_config


def yaw_metrics(telemetry, window, default):
    if len(telemetry["joint_position"][window]) == 0:
        return None
    result = {}
    for index, side in enumerate(("left", "right")):
        joint = MJ_JOINT_NAMES.index(f"{side}_hip_yaw_joint")
        deviation = telemetry["joint_position"][window, joint] - default[joint]
        angle = np.rad2deg(telemetry["foot_heading_relative_base_rad"][window, index])
        result[f"{side}_hip_yaw_abs_error_deg"] = float(np.abs(np.rad2deg(deviation)).mean())
        result[f"{side}_foot_outward_deg"] = float(np.maximum(angle * (1 if index == 0 else -1), 0).mean())
    return result


def elbow_metrics(telemetry, window):
    positions = telemetry["joint_position"][window]
    if len(positions) == 0:
        return None
    velocities = telemetry["joint_velocity"][window]
    result = {}
    for side in ("left", "right"):
        joint = MJ_JOINT_NAMES.index(f"{side}_elbow_joint")
        angles = np.rad2deg(positions[:, joint])
        result[f"{side}_elbow_min_deg"] = float(angles.min())
        result[f"{side}_elbow_max_deg"] = float(angles.max())
        result[f"{side}_elbow_speed_p95_rps"] = float(np.percentile(np.abs(velocities[:, joint]), 95))
        lower, upper = (-30, 45) if side == "left" else (-45, 30)
        excess = np.maximum(lower - angles, 0) + np.maximum(angles - upper, 0)
        result[f"{side}_elbow_outside_fraction"] = float((excess > 1e-4).mean())
        result[f"{side}_elbow_max_excess_deg"] = float(excess.max())
    return result


def braking_metrics(telemetry, stop_step, valid):
    end = min(stop_step + 150, len(valid))
    selected = np.arange(stop_step, end)[valid[stop_step:end]]
    if len(selected) == 0:
        return None
    increments = np.abs(telemetry["joint_target"][selected] - telemetry["joint_target"][selected - 1])
    result = {
        "window_s": 3.0,
        "mechanical_braking_power_peak_w": float(telemetry["braking_power_peak_w"][selected].max()),
        "mechanical_braking_energy_j": float(telemetry["braking_energy_j"][selected].sum()),
        "joint_torque_peak_nm": float(telemetry["joint_torque_peak_nm"][selected].max()),
        "target_increment_max_rad": float(increments.max()),
        "target_increment_p95_rad": float(np.percentile(increments, 95)),
    }
    full_stop = np.arange(stop_step, len(valid))[valid[stop_step:]]
    result["full_stop_power_peak_w"] = float(telemetry["braking_power_peak_w"][full_stop].max())
    result["full_stop_joint_torque_peak_nm"] = float(telemetry["joint_torque_peak_nm"][full_stop].max())
    if "joint_torque_delta_ratio_peak" in telemetry:
        result["torque_delta_ratio_peak"] = float(telemetry["joint_torque_delta_ratio_peak"][selected].max())
    return result


def stop_metrics(telemetry, stop_step, default, body_weight, fallen_at, cruise_start_step=125):
    positions = telemetry["root_position_w"]
    velocity = telemetry["body_velocity"]
    time = np.arange(len(positions)) / 50.0
    origin = positions[stop_step - 1]
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    delays = {}
    for threshold in (0.2, 0.1):
        candidates = [i for i in range(stop_step, len(time) - 24)
                      if np.all(speed[i:i + 25] < threshold)
                      and (fallen_at is None or time[i + 24] < fallen_at)]
        index = candidates[0] if candidates else None
        delays[str(threshold)] = {
            "seconds": None if index is None else float(time[index] - time[stop_step]),
            "distance_m": None if index is None else float(np.linalg.norm(positions[index, :2] - origin[:2])),
        }
    cruise_end = min(cruise_start_step + 250, stop_step)
    cruise = (np.arange(len(time)) >= cruise_start_step) & (np.arange(len(time)) < cruise_end)
    first_second = (time >= time[stop_step]) & (time < time[stop_step] + 1)
    valid = np.ones(len(time), dtype=bool) if fallen_at is None else time < fallen_at
    last = slice(-50, None)

    def pose(window, endpoint):
        if fallen_at is not None and fallen_at <= endpoint:
            return None
        joints = telemetry["joint_position"][window]
        deviation = np.abs(joints - default).sum(axis=1)
        gate = (speed[window] < 0.1) & (np.abs(telemetry["body_angular_velocity"][window, 2]) < 0.1)
        indices = np.arange(len(speed))[window]
        return {
            "l1_rad": float(deviation.mean()),
            "left_knee_abs_deg": float(np.abs(np.rad2deg(joints[:, MJ_JOINT_NAMES.index("left_knee_joint")])).mean()),
            "right_knee_abs_deg": float(np.abs(np.rad2deg(joints[:, MJ_JOINT_NAMES.index("right_knee_joint")])).mean()),
            "mean_planar_speed_mps": float(speed[window].mean()),
            "standing_gate_fraction": float(gate.mean()),
            **yaw_metrics(telemetry, window, default),
            "stopped_yaw": yaw_metrics(telemetry, indices[gate], default),
        }

    final_pose = pose(last, time[-1])
    pose_at_five_seconds = pose(slice(stop_step + 200, stop_step + 250), time[stop_step] + 5)
    # Anchor each segment to its own starting pose, including raw step starts.
    start = cruise_start_step - 1
    heading = telemetry["heading_rad"][start]
    displacement = positions[cruise_start_step:cruise_end, :2] - positions[start, :2]
    lateral = -np.sin(heading) * displacement[:, 0] + np.cos(heading) * displacement[:, 1]
    relative_heading = telemetry["heading_rad"][cruise_end - 1] - heading
    force = telemetry["foot_normal_force"][stop_step - 1]
    return {
        "stop_s": float(time[stop_step]), "actual_speed_before_stop_mps": float(speed[stop_step - 1]),
        "loaded_feet_before_stop": (force > body_weight * 0.05).tolist(),
        "fallen_at_s": fallen_at, "stopping": delays,
        "cruise_mean_vx_mps": float(velocity[cruise & valid, 0].mean()) if (cruise & valid).any() else None,
        "cruise_max_abs_y_m": float(np.abs(positions[cruise, 1]).max()),
        "max_abs_world_y_m": float(np.abs(positions[:, 1]).max()),
        "max_extra_stop_y_m": float(np.abs(positions[stop_step:, 1] - origin[1]).max()),
        "max_abs_heading_deg": float(np.abs(np.rad2deg(telemetry["heading_rad"])).max()),
        "cruise_segment_max_abs_lateral_m": float(np.abs(lateral).max()),
        "cruise_segment_end_lateral_m": float(lateral[-1]),
        "cruise_segment_end_heading_deg": float(np.rad2deg(np.arctan2(np.sin(relative_heading), np.cos(relative_heading)))),
        "first_second_slip": contact_slip_summary(telemetry, first_second & valid, body_weight),
        "braking": braking_metrics(telemetry, stop_step, valid),
        "cruise_yaw": yaw_metrics(telemetry, cruise & valid, default),
        "cruise_elbows": elbow_metrics(telemetry, cruise & valid),
        "pose_at_five_seconds": pose_at_five_seconds, "final_pose": final_pose,
    }


def step_command(step, start_step, stop_step):
    return 4.0 if start_step <= step < stop_step else 0.0


def acceleration_metrics(telemetry, start_step, stop_step, fallen_at):
    time = np.arange(len(telemetry["body_velocity"])) / 50
    valid = np.ones(len(time), dtype=bool) if fallen_at is None else time < fallen_at
    speed = telemetry["body_velocity"][:, 0]
    candidates = [i for i in range(start_step, stop_step - 24)
                  if np.all((speed[i:i + 25] >= 3.6) & (speed[i:i + 25] <= 4.4))
                  and valid[i:i + 25].all()]
    selected = np.arange(start_step, min(start_step + 100, stop_step))
    selected = selected[valid[selected]]
    return {
        "start_s": start_step / 50,
        "fallen_before_stop": fallen_at is not None and fallen_at < stop_step / 50,
        "time_to_sustained_3_6_to_4_4_mps_s": (candidates[0] - start_step) / 50 if candidates else None,
        "joint_torque_peak_nm": float(telemetry["joint_torque_peak_nm"][selected].max()) if len(selected) else None,
        "target_increment_max_rad": float(np.abs(telemetry["joint_target"][selected]
                                      - telemetry["joint_target"][selected - 1]).max()) if len(selected) else None,
    }


def summarize(trials):
    def stats(values):
        valid = [v for v in values if v is not None]
        return None if not valid else {"mean": float(np.mean(valid)), "min": float(np.min(valid)),
                                       "max": float(np.max(valid))}

    summary = {
        "trials": len(trials), "falls": sum(t["fallen_at_s"] is not None for t in trials),
        "falls_before_stop": sum(t["fallen_at_s"] is not None and t["fallen_at_s"] < t["stop_s"] for t in trials),
        "stops_over_3s_or_failed": sum(t["stopping"]["0.1"]["seconds"] is None
                                     or t["stopping"]["0.1"]["seconds"] > 3 for t in trials),
        "failed_stops_below_0_1": sum(t["stopping"]["0.1"]["seconds"] is None for t in trials),
        "cruise_mean_vx_mps": stats([t["cruise_mean_vx_mps"] for t in trials]),
        "cruise_segment_max_abs_lateral_m": stats([t["cruise_segment_max_abs_lateral_m"] for t in trials]),
        "cruise_segment_abs_end_lateral_m": stats([abs(t["cruise_segment_end_lateral_m"]) for t in trials]),
        "max_abs_world_y_m": stats([t["max_abs_world_y_m"] for t in trials]),
        "max_extra_stop_y_m": stats([t["max_extra_stop_y_m"] for t in trials]),
        "stop_below_0_1_seconds": stats([t["stopping"]["0.1"]["seconds"] for t in trials]),
        "stop_below_0_1_distance_m": stats([t["stopping"]["0.1"]["distance_m"] for t in trials]),
        "first_second_slip_mean_mps": stats([t["first_second_slip"]["mean_loaded_foot_speed_mps"] for t in trials]),
        "first_second_slip_p95_mps": stats([t["first_second_slip"]["p95_loaded_foot_speed_mps"] for t in trials]),
        "first_second_slip_peak_mps": stats([t["first_second_slip"]["max_loaded_foot_speed_mps"] for t in trials]),
        "standing_pose_l1_rad": stats([t["final_pose"]["l1_rad"] if t["final_pose"] else None for t in trials]),
        "standing_pose_at_five_seconds_l1_rad": stats([t["pose_at_five_seconds"]["l1_rad"] if t["pose_at_five_seconds"] else None for t in trials]),
        "standing_left_knee_abs_deg": stats([t["final_pose"]["left_knee_abs_deg"] if t["final_pose"] else None for t in trials]),
        "standing_right_knee_abs_deg": stats([t["final_pose"]["right_knee_abs_deg"] if t["final_pose"] else None for t in trials]),
        "final_planar_speed_mps": stats([t["final_pose"]["mean_planar_speed_mps"] if t["final_pose"] else None for t in trials]),
        "final_standing_gate_fraction": stats([t["final_pose"]["standing_gate_fraction"] if t["final_pose"] else None for t in trials]),
    }
    for metric in ("mechanical_braking_power_peak_w", "mechanical_braking_energy_j",
                   "joint_torque_peak_nm", "target_increment_max_rad", "target_increment_p95_rad"):
        summary[metric] = stats([t["braking"][metric] if t.get("braking") else None for t in trials])
    summary["torque_delta_ratio_peak"] = stats([
        t["braking"].get("torque_delta_ratio_peak") if t.get("braking") else None for t in trials
    ])
    for metric in ("full_stop_power_peak_w", "full_stop_joint_torque_peak_nm"):
        summary[metric] = stats([t["braking"].get(metric) if t.get("braking") else None for t in trials])
    for side in ("left", "right"):
        for metric in (f"{side}_hip_yaw_abs_error_deg", f"{side}_foot_outward_deg"):
            summary[f"cruise_{metric}"] = stats([
                t["cruise_yaw"][metric] if t["cruise_yaw"] else None for t in trials
            ])
            summary[f"standing_{metric}"] = stats([
                t["final_pose"]["stopped_yaw"][metric]
                if t["final_pose"] and t["final_pose"]["stopped_yaw"] else None for t in trials
            ])
    for side in ("left", "right"):
        for metric in ("min_deg", "max_deg", "speed_p95_rps", "outside_fraction", "max_excess_deg"):
            key = f"{side}_elbow_{metric}"
            summary[f"cruise_{key}"] = stats([
                t["cruise_elbows"][key] if t.get("cruise_elbows") else None for t in trials
            ])
    if any("acceleration" in t for t in trials):
        for metric in ("time_to_sustained_3_6_to_4_4_mps_s", "joint_torque_peak_nm", "target_increment_max_rad"):
            summary[f"acceleration_{metric}"] = stats([t["acceleration"][metric] for t in trials])
        summary["failed_accelerations_to_3_6_mps"] = sum(
            t["acceleration"]["time_to_sustained_3_6_to_4_4_mps_s"] is None for t in trials
        )
    return summary


def turn_metrics(telemetry, yaw_rate, default, fallen_at):
    time = np.arange(len(telemetry["heading_rad"])) / 50.0
    valid = time >= 1.5
    if fallen_at is not None:
        valid &= time < fallen_at
    velocity = telemetry["body_angular_velocity"][valid, 2]
    heading = np.unwrap(telemetry["heading_rad"])
    displacement = telemetry["root_position_w"][25:, :2] - telemetry["root_position_w"][24, :2]
    return {
        "command_yaw_rate_rps": yaw_rate, "fallen_at_s": fallen_at,
        "mean_yaw_rate_rps": float(velocity.mean()) if len(velocity) else None,
        "yaw_rate_mae_rps": float(np.abs(velocity - yaw_rate).mean()) if len(velocity) else None,
        "turned_angle_deg": float(np.rad2deg(heading[-1] - heading[24])),
        "max_planar_drift_m": float(np.linalg.norm(displacement, axis=1).max()),
        "hip_yaw_and_feet": yaw_metrics(telemetry, valid, default),
    }


def evaluate_turn(policy, config, xml, yaw_rate):
    env = Simulation(xml, config, camera_distance=2.15, initial_height=0.85)
    states = []
    fallen_at = None
    try:
        for step in range(525):
            command = np.array([0, 0, yaw_rate if step >= 25 else 0], dtype=np.float32)
            observation, _ = env.observe(command)
            action, _ = policy(observation)
            env.set_action(action)
            for _ in range(4):
                env.advance()
            state = env.measure()
            states.append(state)
            if fallen_at is None and state["root_position_w"][2] < 0.45:
                fallen_at = step / 50
        telemetry = {k: np.asarray([s[k] for s in states]) for k in states[0]}
        return turn_metrics(telemetry, yaw_rate, env.default, fallen_at)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-tests", action="store_true")
    parser.add_argument("--step-start", action="store_true", help="Raw 0-to-4, hold 5 s, then raw 4-to-0.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    timings = [(i, i + 250) for i in range(25, 56)] if args.step_start else [(25, i) for i in range(375, 406)]
    output = {"protocol": {
        "max_command_mps": 4, "settle_s": 0.5, "acceleration_s": 0 if args.step_start else 2,
        "raw_step_start": args.step_start,
        "start_times_s": [start / 50 for start, _ in timings],
        "stop_times_s": [stop / 50 for _, stop in timings], "hold_after_stop_s": 10,
        "control_hz": 50, "physics_hz": 200, "initial_height_m": 0.85,
        "ground_xml": str(args.xml.resolve()), "automatic_resets": False,
        "randomization": False, "slip_sample_hz": 50,
        "turn_yaw_rates_rps": [-1.0, 1.0] if args.turn_tests else [],
        "turn_command_duration_s": 10,
        "turn_tracking_excludes_first_command_second": True,
        "note": "Same deterministic initial state; switching timing changes, not independent random trials.",
        "braking_metrics": "First 3 s after raw command step; torque/power peaks sampled at 200 Hz."
                           " Mechanical power is not DC bus voltage or electrical regeneration power.",
    }, "checkpoints": []}
    for checkpoint in args.checkpoint:
        policy = AMPPolicy(checkpoint)
        config = read_config(checkpoint.parent / "params/env.yaml")
        trials = []
        for start_step, stop_step in timings:
            env = Simulation(args.xml, config, camera_distance=2.15, initial_height=0.85)
            states = []
            fallen_at = None
            try:
                for step in range(stop_step + 500):
                    if args.step_start:
                        cmd = step_command(step, start_step, stop_step)
                    else:
                        cmd = min(4.0, max(0.0, (step / 50 - 0.5) * 2.0)) if step < stop_step else 0.0
                    observation, _ = env.observe(np.array([cmd, 0, 0], dtype=np.float32))
                    action, _ = policy(observation)
                    env.set_action(action)
                    for _ in range(4):
                        env.advance()
                    state = env.measure()
                    states.append(state)
                    if fallen_at is None and state["root_position_w"][2] < 0.45:
                        fallen_at = step / 50
                telemetry = {k: np.asarray([s[k] for s in states]) for k in states[0]}
                body_weight = float(env.model.body_mass.sum() * np.linalg.norm(env.model.opt.gravity))
                trial = stop_metrics(telemetry, stop_step, env.default, body_weight, fallen_at,
                                     cruise_start_step=start_step if args.step_start else 125)
                if args.step_start:
                    trial["acceleration"] = acceleration_metrics(telemetry, start_step, stop_step, fallen_at)
                trials.append(trial)
                print(json.dumps({"iteration": policy.iteration, "stop_s": trial["stop_s"],
                                  "fall": fallen_at, "max_y_m": trial["max_abs_world_y_m"],
                                  "slip_peak": trial["first_second_slip"]["max_loaded_foot_speed_mps"]}), flush=True)
            finally:
                env.close()
        result = {"checkpoint": str(checkpoint.resolve()), "iteration": policy.iteration,
                  "summary": summarize(trials), "trials": trials}
        if args.turn_tests:
            result["turn_trials"] = [evaluate_turn(policy, config, args.xml, rate) for rate in (-1.0, 1.0)]
        output["checkpoints"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"iteration": policy.iteration, "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
