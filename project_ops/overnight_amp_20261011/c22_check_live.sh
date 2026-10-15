#!/bin/bash
C18=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011
C22=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_19-11-34_amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
echo "=== flight term C22 ==="
grep -n -A15 'high_speed_double_flight:' "$C22/params/env.yaml" | head -20
echo "=== key params ==="
grep -n -E 'max_forward_lean|max_flight_time|speed_gate:|max_velocity:|forward_tolerance:|step_size:' "$C22/params/env.yaml" | head -40
echo "=== agent load ==="
grep -E 'load_run|checkpoint|max_iterations|run_name' "$C22/params/agent.yaml" | head -20
echo "=== DIFF C18 vs C22 env ==="
diff "$C18/params/env.yaml" "$C22/params/env.yaml"
echo "=== DIFF agent ==="
diff "$C18/params/agent.yaml" "$C22/params/agent.yaml" | head -40
echo "=== iter ==="
strings /data/zym/roboparty_train/logs/launch/amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012.log | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -3
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo
