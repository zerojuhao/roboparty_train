#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MUJOCO_GL=egl
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton"
PY=/data/zym/conda/envs/isaaclab_zym/bin/python
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
RUN=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_02-49-49_amp_path_c5_vmax3p0_from18500_20261011
CK=$RUN/model_21499.pt
OUT=$RUN/eval_fall/diag5_21499.json
if [ -f "$OUT" ]; then
  echo ALREADY_HAVE_21499
else
  echo "EVAL 21499 start $(TZ=Asia/Shanghai date '+%F %T') CST"
  tmpd=$(mktemp -d)
  for v in 1.5 2.0 2.5 3.0 3.5 4.0; do
    for m in step ramp; do
      nice -n 10 $PY $OPS/fall_diag5.py --checkpoint $CK --speeds $v --modes $m --out $tmpd/${v}_${m}.json >/dev/null 2>&1 &
    done
  done
  wait
  $PY -c "import json,glob; json.dump(sum([json.load(open(f)) for f in sorted(glob.glob('$tmpd/*.json'))],[]), open('$OUT','w'), indent=1)"
  rm -rf $tmpd
  echo "DONE 21499 $(TZ=Asia/Shanghai date '+%F %T') CST"
fi
cd $OPS && $PY summ5.py amp_path_c5_vmax3p0_from18500_20261011
