#!/usr/bin/env python3
"""Poll C15 until train finishes; print status on each new diag."""
import json, glob, os, re, subprocess, time, sys
RUN = "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_11-51-20_amp_path_c15_shoulder_roll_from29000_20261011"
LOG = "/data/zym/roboparty_train/logs/launch/amp_path_c15_shoulder_roll_from29000_20261011.log"
TRAIN_PID = 3835989
EVAL = os.path.join(RUN, "eval_fall")
STATUS = "/home/zym/roboparty_train/project_ops/overnight_amp_20261011/c15_status.txt"
MON = "/home/zym/roboparty_train/project_ops/overnight_amp_20261011/c15_mon.py"

def alive(pid):
    return os.path.exists(f"/proc/{pid}")

def seen():
    return set(glob.glob(os.path.join(EVAL, "diag5_*.json")))

def dump(msg=""):
    out = subprocess.check_output([sys.executable, MON], text=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S CST", time.localtime())
    block = f"=== {stamp} {msg} ===\n{out}\n"
    with open(STATUS, "a") as f:
        f.write(block)
    print(block, flush=True)
    return out

known = seen()
dump("START")
while True:
    if not alive(TRAIN_PID):
        # wait for final watch eval
        time.sleep(90)
        cur = seen()
        if cur - known:
            known = cur
            dump("NEW_DIAG_AFTER_TRAIN_EXIT")
        else:
            # one more chance
            time.sleep(120)
            cur = seen()
            if cur - known:
                known = cur
                dump("NEW_DIAG_AFTER_TRAIN_EXIT2")
        dump("TRAIN_DEAD")
        with open(STATUS, "a") as f:
            f.write("DONE\n")
        print("DONE", flush=True)
        break
    cur = seen()
    if cur - known:
        known = cur
        dump("NEW_DIAG")
    time.sleep(30)
