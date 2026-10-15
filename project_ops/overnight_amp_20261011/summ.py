#!/usr/bin/env python3
"""Summarize eval_fall/diag_*.json for one or more run dirs (substring match)."""
import glob, json, sys, os
root = "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path"
for sub in sys.argv[1:]:
    for run in sorted(glob.glob(f"{root}/*{sub}*")):
        files = sorted(glob.glob(f"{run}/eval_fall/diag_*.json"), key=lambda f: int(f.split("_")[-1][:-5]))
        if not files: continue
        print("##", os.path.basename(run))
        print("  iter | " + " | ".join(f"{m[:4]}{v:g}" for v in (1.5, 2.0, 2.5, 3.0) for m in ("step", "ramp")))
        for f in files:
            d = {(r["mode"], r["vx"]): r for r in json.load(open(f))}
            it = list(d.values())[0]["iteration"]
            cells = []
            for v in (1.5, 2.0, 2.5, 3.0):
                for m in ("step", "ramp"):
                    r = d.get((m, v))
                    if not r: cells.append("  -  "); continue
                    fa = r["fallen_at_s"]
                    cells.append(" OK  " if fa is None else f"{fa:5.2f}")
            back = max(d[("step", v)]["max_pitch"] for v in (2.0, 2.5) if ("step", v) in d)
            print(f"  {it:5d}| " + " | ".join(cells) + f"  | step back-rock max {back:.1f}deg")