#!/bin/bash
set -e
OUT=/data/zym/roboparty_train/videos/amp_path_4mps_upperfix_20261011
mkdir -p "$OUT"
CKPT=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_11-51-20_amp_path_c15_shoulder_roll_from29000_20261011/model_30000.pt
PY=/data/zym/conda/envs/isaaclab_zym/bin/python
SCRIPT=/home/zym/roboparty_train/robo_newton/scripts/mujoco/sim2sim_rp1_amp.py
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MUJOCO_GL=egl
export PYTHONPATH=/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton
cd /home/zym/roboparty_train/robo_newton
echo "START_STEP4 $(TZ=Asia/Shanghai date '+%F %T CST')"
$PY $SCRIPT --checkpoint "$CKPT" --forward-speed 4 --duration 10 --fps 50 --width 1920 --height 1080 --output "$OUT/c15_30000_step_vx4p0_10s" > "$OUT/step4.log" 2>&1
echo "DONE_STEP4"
echo "START_RAMP4 $(TZ=Asia/Shanghai date '+%F %T CST')"
$PY $SCRIPT --checkpoint "$CKPT" --max-speed 4 --ramp-seconds 2 --duration 10 --fps 50 --width 1920 --height 1080 --output "$OUT/c15_30000_ramp_0to4p0_2s_10s" > "$OUT/ramp4.log" 2>&1
echo "DONE_RAMP4"
echo "START_STEP2 $(TZ=Asia/Shanghai date '+%F %T CST')"
$PY $SCRIPT --checkpoint "$CKPT" --forward-speed 2 --duration 10 --fps 50 --width 1920 --height 1080 --output "$OUT/c15_30000_step_vx2p0_10s" > "$OUT/step2.log" 2>&1
echo "DONE_STEP2"
ls -la "$OUT"/*.mp4
echo ALL_VIDEOS_DONE
