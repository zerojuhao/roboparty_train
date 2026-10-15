#!/bin/bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MUJOCO_GL=egl
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton"
cd /home/zym/roboparty_train/project_ops/overnight_amp_20261011
RUN=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-10_22-51-08_amp_path_lean_orient_test_20261011
/data/zym/conda/envs/isaaclab_zym/bin/python fall_diag.py --checkpoint $RUN/model_12500.pt --speeds 2.0 --modes step --duration 3 --trace 2>&1 | grep -v Warn | tail -40
/data/zym/conda/envs/isaaclab_zym/bin/python fall_diag.py --checkpoint $RUN/model_12500.pt --speeds 1.5 2.0 2.5 --modes step ramp --duration 10 --out diag_12500.json 2>&1 | grep '^{'