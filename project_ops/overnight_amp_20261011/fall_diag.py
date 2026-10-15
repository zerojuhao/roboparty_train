#!/usr/bin/env python3
"""Fall diagnostic: step vs ramp command in MuJoCo sim2sim; trace pitch/roll/height/vx."""
import argparse, json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/zym/roboparty_train/robo_newton/scripts/mujoco")
from sim2sim_rp1_amp import AMPPolicy, DEFAULT_MJCF, Simulation, read_config

def run(policy, config, vx, mode, duration, ramp_s=2.0, trace=False):
    env = Simulation(DEFAULT_MJCF.resolve(), config, camera_distance=2.15)
    rows = []; fallen = None
    try:
        for step in range(round(duration * 50)):
            t = step / 50.0
            c = vx if mode == "step" else vx * float(np.clip((t - 0.5) / ramp_s, 0, 1))
            obs, _ = env.observe(np.array([c, 0, 0], dtype=np.float32))
            a, _ = policy(obs)
            env.set_action(a)
            for _ in range(4): env.advance()
            s = env.measure()
            q = s["root_quaternion_wxyz"]
            g = Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().apply([0, 0, -1.0])
            pitch = np.degrees(np.arctan2(-g[0], -g[2]))  # + = nose forward/down
            roll = np.degrees(np.arctan2(g[1], -g[2]))
            rows.append((t, c, float(s["root_position_w"][2]), pitch, roll, float(s["body_velocity"][0]), float(np.abs(a).max())))
            if fallen is None and s["root_position_w"][2] < 0.45:
                fallen = t
            if fallen is not None and t > fallen + 0.3: break
    finally:
        env.close()
    r = np.array(rows)
    out = {"vx": vx, "mode": mode, "fallen_at_s": fallen,
           "max_pitch": float(r[:, 3].max()), "min_pitch": float(r[:, 3].min()), "max_abs_roll": float(np.abs(r[:, 4]).max()),
           "mean_vx_after2s": float(r[r[:, 0] >= 2.0, 5].mean()) if (r[:, 0] >= 2.0).any() else None,
           "max_abs_action": float(r[:, 6].max())}
    if trace:
        out["trace"] = [[round(x, 3) for x in row] for row in r[::5]]
    return out

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--speeds", type=float, nargs="+", default=[1.5, 2.0, 2.5])
p.add_argument("--modes", nargs="+", default=["step", "ramp"])
p.add_argument("--duration", type=float, default=10.0)
p.add_argument("--trace", action="store_true")
p.add_argument("--out", type=Path)
a = p.parse_args()
torch.set_num_threads(1)
ck = a.checkpoint.resolve(strict=True)
cfg = read_config(ck.parent / "params" / "env.yaml")
pol = AMPPolicy(ck)
res = []
for vx in a.speeds:
    for m in a.modes:
        r = run(pol, cfg, vx, m, a.duration, trace=a.trace)
        r["iteration"] = int(pol.iteration)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "trace"}), flush=True)
        if a.trace and "trace" in r:
            print("  t, cmd, z, pitch(+fwd), roll, vx, max|a|")
            for row in r["trace"]: print("  ", row)
if a.out:
    a.out.write_text(json.dumps(res, indent=1))