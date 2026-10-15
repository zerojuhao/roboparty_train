#!/bin/bash
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
LOGMD=/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md
cp $LOGMD ${LOGMD}.bak_pre_c21_start
R=amp_path_c21_flight_leangate10_clamp030_w4_from36500_20261012
bash $OPS/go_cycle.sh $R 2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011 model_36500.pt 2500 \
  env.curriculum.velocity_tracking.params.max_velocity=4.0 \
  env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295
sleep 2
cd $OPS && nohup /data/zym/conda/envs/isaaclab_zym/bin/python $OPS/c21_monitor.py > $OPS/c21_monitor.out 2>&1 &
echo MON_PID=$!
