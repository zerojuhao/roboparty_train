#!/bin/bash
set -euo pipefail
cd /home/zym/roboparty_train

exec /data/zym/conda/envs/isaaclab_zym/bin/python \
    project_ops/scripts/watch_amp_checkpoint_evaluations.py \
    --run-root /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path \
    --run-name amp_clean_v31_braking_power01_from48000_5000_resume_20261015 \
    --baseline /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-15_02-31-41_amp_clean_v29_physics_peak_braking_power_from47500_5000_resume_20261015/model_48000.pt \
    --final-iteration 52999 \
    --xml /home/zym/roboparty_train/robo_newton/data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml \
    --output-dir /data/zym/roboparty_train/evaluations/amp_v31_braking_power01_20261015 \
    --turn-tests --step-tests --timeout-hours 12 "$@"
