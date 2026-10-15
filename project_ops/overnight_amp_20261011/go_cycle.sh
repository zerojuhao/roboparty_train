#!/bin/bash
# usage: go_cycle.sh RUN_NAME LOAD_RUN CKPT MAX_ITERS [overrides...]  -> nohup launch + start watcher
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
RUN_NAME=$1
LOG=/data/zym/roboparty_train/logs/launch/${RUN_NAME}.log
[ -f $LOG ] && mv $LOG ${LOG}.prev_$(date +%s)
nohup bash $OPS/launch_cycle.sh "$@" > $LOG 2>&1 &
echo "LAUNCH_PID=$!"
pkill -f "watch_fall5.sh" 2>/dev/null; sleep 1
nohup bash $OPS/watch_fall5.sh "$RUN_NAME" > /data/zym/roboparty_train/logs/launch/watch_fall5_${RUN_NAME}.log 2>&1 &
echo "WATCH_PID=$!"