#!/usr/bin/env python3
import numpy as np, glob, os, sys, json
outdir = sys.argv[1] if len(sys.argv)>1 else "/data/zym/roboparty_train/videos/amp_path_4mps_upperfix_20261011"
for npz_path in sorted(glob.glob(os.path.join(outdir, "*.npz"))):
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["joint_names"]]
    t = z["time"]
    q = z["joint_position"]  # (T, 24)
    # find shoulder roll indices
    roll_idxs = [i for i,n in enumerate(names) if "shoulder_roll" in n.lower()]
    pitch_idxs = [i for i,n in enumerate(names) if "shoulder_pitch" in n.lower()]
    print("==", os.path.basename(npz_path))
    print("  joints:", names)
    print("  roll_idxs", roll_idxs, [names[i] for i in roll_idxs])
    print("  pitch_idxs", pitch_idxs, [names[i] for i in pitch_idxs])
    # after 2s window (match fall_diag)
    mask = t >= 2.0
    if not mask.any():
        mask = np.ones_like(t, dtype=bool)
    if roll_idxs:
        rolls = np.abs(q[mask][:, roll_idxs])
        mean_abs = float(rolls.mean())
        print("  mean|shoulder_roll| after2s:", round(mean_abs, 4),
              "per_joint:", [round(float(np.abs(q[mask][:,i]).mean()),4) for i in roll_idxs])
    if pitch_idxs:
        pitches = q[mask][:, pitch_idxs]
        prange = float(pitches.max() - pitches.min()) if pitches.size else float("nan")
        print("  shoulder_pitch_range after2s:", round(prange, 4))
    # also load json for survived/y
    jf = npz_path[:-4] + ".json"
    if os.path.exists(jf):
        d = json.load(open(jf))
        survived = d.get("fallen_at_s") is None
        print("  survived:", survived, "max|y|:", round(d.get("max_abs_world_y_m", -1), 4),
              "minH:", round(d.get("minimum_root_height_m", -1), 4))
