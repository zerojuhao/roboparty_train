#!/usr/bin/env python3
import json, glob, os, re, subprocess
RUN = "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_11-51-20_amp_path_c15_shoulder_roll_from29000_20261011"
LOG = "/data/zym/roboparty_train/logs/launch/amp_path_c15_shoulder_roll_from29000_20261011.log"
EVAL = os.path.join(RUN, "eval_fall")

def last_iter():
    try:
        out = subprocess.check_output(
            ["grep", "-aaoE", r"Learning iteration [0-9]+/[0-9]+", LOG], text=True
        )
        lines = [l for l in out.strip().splitlines() if l.strip()]
        m = re.search(r"(\d+)/(\d+)", lines[-1] if lines else "")
        return m.groups() if m else ("?", "?")
    except Exception as e:
        return ("err", str(e))

def roll_reward():
    try:
        out = subprocess.check_output(
            ["grep", "-a", "Episode_Reward/shoulder_roll_excess", LOG], text=True
        )
        lines = [l for l in out.strip().splitlines() if l.strip()]
        m = re.search(r"(-?[0-9.]+)\s*$", lines[-1].strip() if lines else "")
        return m.group(1) if m else "?"
    except Exception:
        return "?"

def summarize():
    files = sorted(
        glob.glob(os.path.join(EVAL, "diag5_*.json")),
        key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]),
    )
    print("ITER " + "/".join(last_iter()) + "  shoulder_roll_excess=" + str(roll_reward()))
    print("ckpts: " + str([os.path.basename(p) for p in sorted(glob.glob(os.path.join(RUN, "model_*.pt")))]))
    print("iter | TOT | s4 | r4 | |y|max | step4_roll | step4_pitch | step2_roll | step2_pitch")
    for f in files:
        rows = json.load(open(f))
        d = {(r["mode"], float(r["vx"])): r for r in rows}
        it = rows[0]["iteration"]
        tot = sum(r["survived"] for r in rows)
        n = sum(r["trials"] for r in rows)
        s4 = d.get(("step", 4.0))
        r4 = d.get(("ramp", 4.0))
        s2 = d.get(("step", 2.0))
        ymax = max((r.get("max_abs_world_y_max", 0) for r in rows), default=0)

        def arm(r):
            if not r:
                return ("-", "-")
            return (
                "{:.3f}".format(r.get("mean_abs_shoulder_roll_after2s_med", float("nan"))),
                "{:.3f}".format(r.get("shoulder_pitch_range_after2s_med", float("nan"))),
            )

        s4r, s4p = arm(s4)
        s2r, s2p = arm(s2)
        s4s = "{}/5".format(s4["survived"]) if s4 else "-"
        r4s = "{}/5".format(r4["survived"]) if r4 else "-"
        print(
            "{:5d} | {:2d}/{} | {:3s} | {:3s} | {:5.2f} | {:>10s} | {:>11s} | {:>10s} | {:>11s}".format(
                it, tot, n, s4s, r4s, ymax, s4r, s4p, s2r, s2p
            )
        )

if __name__ == "__main__":
    summarize()
