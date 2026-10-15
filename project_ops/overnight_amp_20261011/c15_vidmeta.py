#!/usr/bin/env python3
import json, glob, sys, os, numpy as np
outdir = sys.argv[1] if len(sys.argv) > 1 else "/data/zym/roboparty_train/videos/amp_path_4mps_upperfix_20261011"
for jf in sorted(glob.glob(os.path.join(outdir, "*.json"))):
    d = json.load(open(jf))
    base = jf[:-5]
    mp4 = base + ".mp4" if os.path.exists(base + ".mp4") else (base if os.path.exists(base) else None)
    print("==", os.path.basename(jf))
    print("  command:", d.get("command"))
    print("  fallen_at_s:", d.get("fallen_at_s"), "minH:", round(d.get("minimum_root_height_m", -1), 3) if d.get("minimum_root_height_m") is not None else None)
    print("  max|y|:", round(d.get("max_abs_world_y_m", -1), 3) if d.get("max_abs_world_y_m") is not None else None)
    print("  mean_vx:", round(d.get("mean_actual_vx_mps", -1), 3) if d.get("mean_actual_vx_mps") is not None else None,
          "final_vx:", round(d.get("final_actual_vx_mps", -1), 3) if d.get("final_actual_vx_mps") is not None else None)
    print("  video:", mp4, "size:", os.path.getsize(mp4) if mp4 and os.path.exists(mp4) else None)
    # shoulder roll from npz if present
    npz = base + ".npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        keys = list(z.keys())
        print("  npz keys:", keys[:30])
        # try common names
        for k in keys:
            if "shoulder" in k.lower() or "qpos" in k.lower() or "dof" in k.lower() or "joint" in k.lower():
                arr = z[k]
                print("   ", k, "shape", getattr(arr, "shape", None), "dtype", getattr(arr, "dtype", None))
