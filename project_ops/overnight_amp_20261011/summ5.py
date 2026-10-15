#!/usr/bin/env python3
import glob, json, sys, os
root = "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path"
SP = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
YLIM = 1.0
for sub in sys.argv[1:]:
    for run in sorted(glob.glob(f"{root}/*{sub}*")):
        files = sorted(glob.glob(f"{run}/eval_fall/diag5_*.json"), key=lambda f: int(f.split("_")[-1].replace("v40lat.json","json")[:-5]) if "v40lat" not in f else 18500)
        # also include v40lat named files
        files = sorted(set(glob.glob(f"{run}/eval_fall/diag5_*.json")), key=lambda f: (0 if "v40lat" in f else 1, int("".join(ch for ch in f.split("diag5_")[-1] if ch.isdigit()) or "0")))
        if not files:
            continue
        print("##", os.path.basename(run), f" cell=surv/5; yOK= max|y|_max<={YLIM}m")
        hdr = "|".join(f"{m[:1]}{v:g}".center(9) for v in SP for m in ("step", "ramp"))
        print("  iter |" + hdr + "|TOT |fwd |back| |y|med| |y|max| yPASS |yawmed")
        for f in files:
            rows = json.load(open(f))
            # skip old 8-cell-only if mixed? still print
            d = {(r["mode"], float(r["vx"])): r for r in rows}
            it = rows[0]["iteration"]
            tag = "v40" if "v40lat" in f else ""
            cells = []
            tot = n = 0
            ypass = ytot = 0
            for v in SP:
                for m in ("step", "ramp"):
                    r = d.get((m, float(v)))
                    if not r:
                        cells.append("-".center(9)); continue
                    cells.append(f"{r['survived']}/{r['trials']}".center(9))
                    tot += r["survived"]; n += r["trials"]
                    if "max_abs_world_y_max" in r:
                        ytot += 1
                        if r["max_abs_world_y_max"] <= YLIM:
                            ypass += 1
            fl = sorted(r["max_fwd_lean_deg_pre_fall"] for r in rows if r["vx"] >= 2.0)
            bl = sorted(r["max_back_lean_deg"] for r in rows if r["vx"] >= 2.0)
            ymeds = [r["max_abs_world_y_med"] for r in rows if "max_abs_world_y_med" in r and r["vx"] >= 2.0]
            ymaxs = [r["max_abs_world_y_max"] for r in rows if "max_abs_world_y_max" in r and r["vx"] >= 2.0]
            yaws = [r["max_abs_yaw_deg_med"] for r in rows if "max_abs_yaw_deg_med" in r and r["vx"] >= 2.0]
            def med(xs):
                if not xs: return None
                xs = sorted(xs); return xs[len(xs)//2]
            def fmt(x,p=2):
                return f"{x:5.{p}f}" if x is not None else "  n/a"
            yp = f"{ypass}/{ytot}" if ytot else "n/a"
            print(f"  {it:5d}{tag:3s}|" + "|".join(cells)
                  + f"|{tot:2d}/{n}|{fmt(med(fl),1)}|{fmt(med(bl),1)}|{fmt(med(ymeds))}|{fmt(max(ymaxs) if ymaxs else None)}|{yp:^7}|{fmt(med(yaws),1)}")
            # per speed/mode y pass/fail for high speeds of interest
            bits = []
            for v in (3.0, 3.5, 4.0):
                for m in ("step", "ramp"):
                    r = d.get((m, float(v)))
                    if not r or "max_abs_world_y_max" not in r: continue
                    ok = "PASS" if r["max_abs_world_y_max"] <= YLIM else "FAIL"
                    bits.append(f"{m[0]}{v:g}:{r['survived']}/5 y{r['max_abs_world_y_max']:.2f}{ok[0]}")
            if bits:
                print("         " + "  ".join(bits))
