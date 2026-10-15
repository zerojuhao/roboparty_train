#!/usr/bin/env python3
"""Multi-trial MuJoCo fall+lateral eval: step vs ramp; trial 0 nominal, trials>0 joint-pos noise +-0.05 rad.
Adds vx 3.5/4.0 by default and lateral drift: max|y|, final y, mean|vy| after 2s, yaw drift.
Also reports mean |shoulder_roll| and shoulder_pitch range after 2s (raised-arms metric).
"""
import argparse, json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, torch, mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/zym/roboparty_train/robo_newton/scripts/mujoco")
from sim2sim_rp1_amp import AMPPolicy, DEFAULT_MJCF, Simulation, read_config

def _joint_qpos_addr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[jid])


def _flight_stats_from_forces(forces, dt=0.02, force_thr=5.0, after_t=2.0):
    """forces: (T,2) foot normal forces. Return double-flight fraction + max bout (s) after after_t."""
    import numpy as _np
    forces = _np.asarray(forces, dtype=float)
    if forces.ndim != 2 or forces.shape[0] == 0:
        return None, None
    T = forces.shape[0]
    t = _np.arange(T) * float(dt)
    mask = t >= float(after_t)
    if not mask.any():
        return None, None
    both_air = (forces[mask] <= float(force_thr)).all(axis=1)
    frac = float(both_air.mean()) if both_air.size else 0.0
    # max consecutive True run
    max_run = 0
    cur = 0
    for v in both_air.tolist():
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return frac, float(max_run) * float(dt)


def run(policy, config, vx, mode, duration, trial, ramp_s=2.0):
    env = Simulation(DEFAULT_MJCF.resolve(), config, camera_distance=2.15)
    if trial > 0:
        rng = np.random.default_rng(1000 + trial)
        env.data.qpos[env.qids] += rng.uniform(-0.05, 0.05, len(env.qids))
        mujoco.mj_forward(env.model, env.data)
    fallen = None
    fwd = 0.0
    back = 0.0
    vxs = []
    vys = []
    ys = []
    yaws = []
    yaw0 = None
    rolls_abs = []
    pitch_vals = []
    foot_forces = []
    try:
        q_roll_l = _joint_qpos_addr(env.model, "left_shoulder_roll_joint")
        q_roll_r = _joint_qpos_addr(env.model, "right_shoulder_roll_joint")
        q_pitch_l = _joint_qpos_addr(env.model, "left_shoulder_pitch_joint")
        q_pitch_r = _joint_qpos_addr(env.model, "right_shoulder_pitch_joint")
        track_arms = True
    except Exception:
        track_arms = False
    try:
        for step in range(round(duration * 50)):
            t = step / 50.0
            c = vx if mode == "step" else vx * float(np.clip((t - 0.5) / ramp_s, 0, 1))
            obs, _ = env.observe(np.array([c, 0, 0], dtype=np.float32))
            a, _ = policy(obs)
            env.set_action(a)
            for _ in range(4):
                env.advance()
            s = env.measure()
            q = s["root_quaternion_wxyz"]
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            g = rot.inv().apply([0, 0, -1.0])
            lean = np.degrees(np.arctan2(g[0], -g[2]))  # + = forward (nose down)
            yaw = float(rot.as_euler("xyz", degrees=True)[2])
            if yaw0 is None:
                yaw0 = yaw
            y = float(s["root_position_w"][1])
            ys.append(y)
            yaws.append(yaw - yaw0)
            try:
                foot_forces.append(np.asarray(s["foot_normal_force"], dtype=float).tolist())
            except Exception:
                pass
            if s["root_position_w"][2] < 0.45:
                fallen = t
                break
            fwd = max(fwd, lean)
            back = max(back, -lean)
            if t >= 2.0:
                bv = s["body_velocity"]
                vxs.append(float(bv[0]))
                vys.append(abs(float(bv[1])))
                if track_arms:
                    qp = env.data.qpos
                    rl = float(qp[q_roll_l]); rr = float(qp[q_roll_r])
                    pl = float(qp[q_pitch_l]); pr = float(qp[q_pitch_r])
                    rolls_abs.append(0.5 * (abs(rl) + abs(rr)))
                    pitch_vals.append(pl)
                    pitch_vals.append(pr)
    finally:
        env.close()
    max_abs_y = float(np.max(np.abs(ys))) if ys else 0.0
    final_y = float(ys[-1]) if ys else 0.0
    mean_abs_vy = float(np.mean(vys)) if vys else None
    max_abs_yaw = float(np.max(np.abs(yaws))) if yaws else 0.0
    final_yaw = float(yaws[-1]) if yaws else 0.0
    mean_abs_roll = float(np.mean(rolls_abs)) if rolls_abs else None
    pitch_range = float(np.max(pitch_vals) - np.min(pitch_vals)) if pitch_vals else None
    dfrac, dmax = _flight_stats_from_forces(foot_forces, dt=0.02, force_thr=5.0, after_t=2.0) if foot_forces else (None, None)
    return {
        "fallen": fallen,
        "fwd": fwd,
        "back": back,
        "mean_vx": (float(np.mean(vxs)) if vxs else None),
        "max_abs_world_y": max_abs_y,
        "final_world_y": final_y,
        "mean_abs_vy_after2s": mean_abs_vy,
        "max_abs_yaw_deg": max_abs_yaw,
        "final_yaw_deg": final_yaw,
        "mean_abs_shoulder_roll_after2s": mean_abs_roll,
        "shoulder_pitch_range_after2s": pitch_range,
        "double_flight_fraction_after2s": dfrac,
        "max_double_flight_s_after2s": dmax,
    }

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--speeds", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
p.add_argument("--modes", nargs="+", default=["step", "ramp"])
p.add_argument("--duration", type=float, default=10.0)
p.add_argument("--trials", type=int, default=5)
p.add_argument("--out", type=Path)
a = p.parse_args()
torch.set_num_threads(1)
ck = a.checkpoint.resolve(strict=True)
cfg = read_config(ck.parent / "params" / "env.yaml")
pol = AMPPolicy(ck)
res = []
for vx in a.speeds:
    for m in a.modes:
        trials = []
        for tr in range(a.trials):
            trials.append(run(pol, cfg, vx, m, a.duration, tr))
        surv = sum(t["fallen"] is None for t in trials)
        falls = [t["fallen"] for t in trials]
        roll_vals = [t["mean_abs_shoulder_roll_after2s"] for t in trials if t.get("mean_abs_shoulder_roll_after2s") is not None]
        pitch_ranges = [t["shoulder_pitch_range_after2s"] for t in trials if t.get("shoulder_pitch_range_after2s") is not None]
        r = {
            "iteration": int(pol.iteration),
            "vx": vx,
            "mode": m,
            "trials": a.trials,
            "survived": surv,
            "fall_times": falls,
            "mean_time_alive": float(np.mean([a.duration if f is None else f for f in falls])),
            "max_fwd_lean_deg_pre_fall": float(np.median([t["fwd"] for t in trials])),
            "max_back_lean_deg": float(np.median([t["back"] for t in trials])),
            "mean_vx_after2s": [t["mean_vx"] for t in trials],
            "max_abs_world_y_med": float(np.median([t["max_abs_world_y"] for t in trials])),
            "max_abs_world_y_max": float(np.max([t["max_abs_world_y"] for t in trials])),
            "final_world_y_med": float(np.median([t["final_world_y"] for t in trials])),
            "mean_abs_vy_after2s_med": float(np.median([t["mean_abs_vy_after2s"] or 0.0 for t in trials])),
            "max_abs_yaw_deg_med": float(np.median([t["max_abs_yaw_deg"] for t in trials])),
            "final_yaw_deg_med": float(np.median([t["final_yaw_deg"] for t in trials])),
            "mean_abs_shoulder_roll_after2s_med": float(np.median(roll_vals)) if roll_vals else float("nan"),
            "shoulder_pitch_range_after2s_med": float(np.median(pitch_ranges)) if pitch_ranges else float("nan"),
            "double_flight_fraction_after2s_med": float(np.median([t["double_flight_fraction_after2s"] for t in trials if t.get("double_flight_fraction_after2s") is not None] or [float("nan")])),
            "max_double_flight_s_after2s_med": float(np.median([t["max_double_flight_s_after2s"] for t in trials if t.get("max_double_flight_s_after2s") is not None] or [float("nan")])),
            "max_double_flight_s_after2s_max": float(np.max([t["max_double_flight_s_after2s"] for t in trials if t.get("max_double_flight_s_after2s") is not None] or [float("nan")])),
            "per_trial_lateral": [
                {
                    "max_abs_world_y": t["max_abs_world_y"],
                    "final_world_y": t["final_world_y"],
                    "mean_abs_vy_after2s": t["mean_abs_vy_after2s"],
                    "max_abs_yaw_deg": t["max_abs_yaw_deg"],
                    "final_yaw_deg": t["final_yaw_deg"],
                    "mean_abs_shoulder_roll_after2s": t.get("mean_abs_shoulder_roll_after2s"),
                    "shoulder_pitch_range_after2s": t.get("shoulder_pitch_range_after2s"),
                }
                for t in trials
            ],
        }
        res.append(r)
        slim = {k: r[k] for k in (
            "iteration", "vx", "mode", "survived", "mean_time_alive",
            "max_fwd_lean_deg_pre_fall", "max_back_lean_deg",
            "max_abs_world_y_med", "max_abs_world_y_max", "final_world_y_med",
            "mean_abs_vy_after2s_med", "max_abs_yaw_deg_med", "final_yaw_deg_med",
            "mean_abs_shoulder_roll_after2s_med", "shoulder_pitch_range_after2s_med",
            "double_flight_fraction_after2s_med", "max_double_flight_s_after2s_med", "max_double_flight_s_after2s_max",
        )}
        print(json.dumps(slim), flush=True)
if a.out:
    a.out.write_text(json.dumps(res, indent=1))
