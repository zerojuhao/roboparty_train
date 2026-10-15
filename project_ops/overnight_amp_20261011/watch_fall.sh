#!/bin/bash
# CPU-only sim2sim fall watcher. usage: watch.sh RUN_DIR_SUBSTR
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MUJOCO_GL=egl
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton"
PY=/data/zym/conda/envs/isaaclab_zym/bin/python
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
ROOT=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path
SUB=$1
RUN=$(ls -dt $ROOT/*${SUB}* | head -1)
echo "$(date -d '+8 hour' '+%F %T') CST WATCHING $RUN"
mkdir -p $RUN/eval_fall
while true; do
  for ck in $RUN/model_*.pt; do
    [ -f "$ck" ] || continue
    it=$(basename $ck .pt); it=${it#model_}
    [ $((it % 500)) -eq 0 ] || [ $((it % 500)) -eq 499 ] || continue
    out=$RUN/eval_fall/diag_${it}.json
    [ -f $out ] && continue
    s1=$(stat -c%s $ck); sleep 3; s2=$(stat -c%s $ck); [ "$s1" = "$s2" ] || continue
    echo "$(date -d '+8 hour' '+%F %T') CST EVAL $it"
    nice -n 10 $PY $OPS/fall_diag.py --checkpoint $ck --speeds 1.5 2.0 2.5 3.0 --modes step ramp --duration 10 --out $out 2>/dev/null | grep '^{'
  done
  sleep 60
done