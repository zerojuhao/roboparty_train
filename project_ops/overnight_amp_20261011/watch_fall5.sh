#!/bin/bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MUJOCO_GL=egl
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton"
PY=/data/zym/conda/envs/isaaclab_zym/bin/python
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
ROOT=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path
SUB=$1
RUN=""
for i in $(seq 1 120); do RUN=$(ls -dt $ROOT/*${SUB}* 2>/dev/null | head -1); [ -n "$RUN" ] && break; sleep 10; done
echo "$(TZ=Asia/Shanghai date '+%F %T') CST WATCHING $RUN"
mkdir -p $RUN/eval_fall
while true; do
  for ck in $(ls $RUN/model_*.pt 2>/dev/null | sort -V); do
    it=$(basename $ck .pt); it=${it#model_}
    r=$((it % 500)); [ $r -eq 0 ] || [ $r -ge 498 ] || continue
    out=$RUN/eval_fall/diag5_${it}.json
    [ -f $out ] && continue
    s1=$(stat -c%s $ck); sleep 3; s2=$(stat -c%s $ck); [ "$s1" = "$s2" ] || continue
    echo "$(TZ=Asia/Shanghai date '+%F %T') CST EVAL $it"
    tmpd=$(mktemp -d)
    for v in 1.5 2.0 2.5 3.0 3.5 4.0; do for m in step ramp; do
      nice -n 10 $PY $OPS/fall_diag5.py --checkpoint $ck --speeds $v --modes $m --out $tmpd/${v}_${m}.json >/dev/null 2>&1 &
    done; done
    wait
    $PY -c "import json,glob; json.dump(sum([json.load(open(f)) for f in sorted(glob.glob('$tmpd/*.json'))],[]), open('$out','w'), indent=1)"
    rm -rf $tmpd
    echo "$(TZ=Asia/Shanghai date '+%F %T') CST DONE $it"
  done
  sleep 60
done
