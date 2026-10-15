#!/bin/bash
set -e
B=/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp
echo "=== code gate value ==="
grep -n 'max_forward_lean\|C22_LEAN_GATE_15\|C21_LEAN' $B/rp1_amp_env_cfg.py $B/mdp/rewards.py | head -30
echo "=== math check ==="
/data/zym/conda/envs/isaaclab_zym/bin/python -c "import math; v=0.2617993877991494; print(v, math.degrees(v))"
echo "=== backups ==="
ls -lt $B/mdp/rewards.py.bak_c22_* $B/rp1_amp_env_cfg.py.bak_c22_* 2>/dev/null | head -4
ls -lt /data/zym/roboparty_train/backups/*bak_c22_* 2>/dev/null | head -4
echo "=== C18 ckpt ==="
ls -la /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011/model_36500.pt
