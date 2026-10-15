#!/bin/bash
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
NAME=amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
C22=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_19-11-34_amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
echo "=== weight line ==="
grep -A25 'high_speed_double_flight:' "$C22/params/env.yaml" | grep -E 'weight:|max_forward_lean|max_flight|speed_gate'
echo "=== status ==="
tail -20 "$OPS/c22_status.txt" 2>/dev/null
echo "=== done? ==="
cat "$OPS/c22_done.txt" 2>/dev/null || echo NOT_DONE
echo "=== train alive ==="
pgrep -af 'rsl_rl/train.py' | grep -c "$NAME" || true
echo "=== iter ==="
strings /data/zym/roboparty_train/logs/launch/${NAME}.log | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -2
echo "=== evals ==="
ls "$C22/eval_fall/" 2>/dev/null | tr '\n' ' '; echo
echo "=== gpu0 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
