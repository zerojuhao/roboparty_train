#!/bin/bash
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
DONE=$OPS/c22_done.txt
STATUS=$OPS/c22_status.txt
NAME=amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
LOG=/data/zym/roboparty_train/logs/launch/${NAME}.log
for i in $(seq 1 240); do
  if [ -f "$DONE" ]; then
    echo "DONE_FILE"
    cat "$DONE"
    echo "==== STATUS TAIL ===="
    tail -30 "$STATUS"
    echo "==== MONITOR OUT TAIL ===="
    tail -40 "$OPS/c22_monitor.out"
    exit 0
  fi
  ITER=$(strings "$LOG" 2>/dev/null | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -1)
  EVALS=$(ls /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_19-11-34_${NAME}/eval_fall/ 2>/dev/null | tr '\n' ',')
  LAST=$(tail -1 "$STATUS" 2>/dev/null)
  ALIVE=$(pgrep -af 'rsl_rl/train.py' | grep -c "$NAME" || true)
  echo "$(TZ=Asia/Shanghai date '+%F %T') CST poll=$i alive=$ALIVE $ITER evals=$EVALS last=$LAST"
  sleep 60
done
echo TIMEOUT
tail -50 "$STATUS"
exit 1
