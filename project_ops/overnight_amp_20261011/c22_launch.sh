#!/bin/bash
# Launch C22: lean gate 15°, resume C18 model_36500, 2500 iters, watch_fall5 + monitor
set -euo pipefail
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
LOGMD=/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md
cp -n "$LOGMD" "${LOGMD}.bak_pre_c22_start" 2>/dev/null || cp "$LOGMD" "${LOGMD}.bak_pre_c22_start"
R=amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
# Kill any leftover c21 monitor/watch (should already be dead)
pkill -f 'c21_monitor.py' 2>/dev/null || true
pkill -f 'watch_fall5.sh' 2>/dev/null || true
sleep 1
bash "$OPS/go_cycle.sh" "$R" \
  2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011 \
  model_36500.pt 2500 \
  env.curriculum.velocity_tracking.params.max_velocity=4.0 \
  env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295 \
  env.rewards.high_speed_double_flight.params.max_forward_lean=0.2617993877991494
sleep 3
cd "$OPS"
nohup /data/zym/conda/envs/isaaclab_zym/bin/python "$OPS/c22_monitor.py" > "$OPS/c22_monitor.out" 2>&1 &
echo MON_PID=$!
echo RUN_NAME=$R
