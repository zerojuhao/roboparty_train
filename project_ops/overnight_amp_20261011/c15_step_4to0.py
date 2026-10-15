#!/usr/bin/env python3
"""True STEP cmd: hold vx=4 for cruise_s, then abruptly 0 for the rest. Records video+json+npz."""
import os, sys, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path[:0] = [
    "/home/zym/roboparty_train/rsl_rl",
    "/home/zym/roboparty_train/robo_newton",
    "/home/zym/roboparty_train/robo_newton/scripts/mujoco",
]
import sim2sim_rp1_amp as S

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/"
    "2026-10-11_11-51-20_amp_path_c15_shoulder_roll_from29000_20261011/model_30500.pt"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else (
    "/data/zym/roboparty_train/videos/amp_path_4mps_upperfix_20261011/c15_30500_step_4to0_10s"
)
CRUISE_S = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5  # then step to 0
DURATION = 10.0
MAX_SPEED = 4.0

def step_cmd(seconds, duration, maximum, ramp_seconds=None, deceleration_seconds=None, standing_seconds=0.0):
    # ignore ramp/decel API; true step schedule
    if seconds < CRUISE_S:
        return float(maximum)
    return 0.0

S.ramp_command = step_cmd
# drive through CLI: max-speed path uses ramp_command
argv = [
    "sim2sim_rp1_amp.py",
    "--checkpoint", CKPT,
    "--max-speed", str(MAX_SPEED),
    "--ramp-seconds", "0.01",  # unused by our patched ramp_command but required by CLI when using decel paths; still needed for max-speed branch
    "--duration", str(DURATION),
    "--fps", "50",
    "--width", "1920",
    "--height", "1080",
    "--output", OUT,
]
# avoid FileExistsErro
from pathlib import Path
for ext in ("", ".mp4", ".json", ".npz"):
    p = Path(OUT + ("" if ext == "" else ""))
# output path without suffix; script may write as given
outp = Path(OUT)
if outp.exists():
    outp.unlink()
for s in (".mp4", ".json", ".npz"):
    p = Path(str(OUT) + s) if not str(OUT).endswith(".mp4") else Path(str(OUT).replace(".mp4", s) if s != ".mp4" else OUT)
    # OUT has no extension typically
    p2 = Path(OUT + s)
    if p2.exists():
        p2.unlink()
if Path(OUT).exists():
    Path(OUT).unlink()

sys.argv = argv
S.main()

# post: rename bare file to .mp4 if needed, print extras
import glob
base = OUT
cand = [base, base + ".mp4"]
mp4 = next((c for c in cand if Path(c).exists() and Path(c).stat().st_size > 1000), None)
if mp4 and not mp4.endswith(".mp4"):
    Path(mp4).rename(mp4 + ".mp4")
    mp4 = mp4 + ".mp4"
    if Path(base + ".json").exists() is False and Path(base).with_suffix(".json").exists():
        pass
jf = base + ".json"
if not Path(jf).exists() and Path(base).with_suffix(".json").exists():
    jf = str(Path(base).with_suffix(".json"))
# compute pitch extremes + mean |shoulder_roll| after step
npz = base + ".npz"
z = np.load(npz, allow_pickle=True)
names = [str(n) for n in z["joint_names"]]
t = z["time"]; q = z["joint_position"]
# lean from root quat: approximate pitch from body
# use fall_diag style: root height already in json; pitch from quaternion
rq = z["root_quaternion_wxyz"]
# pitch from wxyz: asin(2*(w*y - z*x)) roughly for pitch; better use rotation matrix
def quat_to_rpy(q):
    w,x,y,z_ = q
    # roll (x), pitch (y), yaw (z)
    sinr = 2*(w*x + y*z_); cosr = 1 - 2*(x*x + y*y); roll = np.arctan2(sinr, cosr)
    sinp = 2*(w*y - z_*x); sinp = np.clip(sinp, -1, 1); pitch = np.arcsin(sinp)
    siny = 2*(w*z_ + x*y); cosy = 1 - 2*(y*y + z_*z_); yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw
rpy = np.array([quat_to_rpy(q) for q in rq])
pitch = np.rad2deg(rpy[:,1]); rollb = np.rad2deg(rpy[:,0])
ri = [names.index(n) for n in names if "shoulder_roll" in n]
mask_all = t >= 0
mask_after = t >= CRUISE_S
mean_abs_sh = float(np.abs(q[:, ri]).mean())
mean_abs_sh_after = float(np.abs(q[mask_after][:, ri]).mean()) if mask_after.any() else float("nan")
mean_abs_sh_cruise = float(np.abs(q[t < CRUISE_S][:, ri]).mean())
rep = json.load(open(jf))
summary = {
    "checkpoint": CKPT,
    "schedule": f"STEP: vx={MAX_SPEED} for t<[0,{CRUISE_S}), then vx=0 until {DURATION}",
    "survived": rep.get("fallen_at_s") is None,
    "fallen_at_s": rep.get("fallen_at_s"),
    "max_abs_world_y_m": rep.get("max_abs_world_y_m"),
    "minimum_root_height_m": rep.get("minimum_root_height_m"),
    "pitch_min_deg": float(pitch.min()),
    "pitch_max_deg": float(pitch.max()),
    "base_roll_min_deg": float(rollb.min()),
    "base_roll_max_deg": float(rollb.max()),
    "mean_abs_shoulder_roll_all": mean_abs_sh,
    "mean_abs_shoulder_roll_cruise": mean_abs_sh_cruise,
    "mean_abs_shoulder_roll_after_step": mean_abs_sh_after,
    "final_actual_vx_mps": rep.get("final_actual_vx_mps"),
    "max_actual_vx_mps": rep.get("max_actual_vx_mps"),
    "video": mp4,
}
print("SUMMARY=" + json.dumps(summary, indent=2))
open(base + "_4to0_summary.json", "w").write(json.dumps(summary, indent=2) + "\n")
