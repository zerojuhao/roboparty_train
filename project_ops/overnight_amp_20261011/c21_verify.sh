#!/bin/bash
ROOT=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path
C18=$ROOT/2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011
C21=$(ls -dt $ROOT/*amp_path_c21_flight_leangate10* | head -1)
echo RUN=$C21
echo "--- diff env.yaml C18 vs C21"; diff $C18/params/env.yaml $C21/params/env.yaml
echo "--- diff agent.yaml C18 vs C21"; diff $C18/params/agent.yaml $C21/params/agent.yaml
echo "--- flight term"; grep -n -A3 -E 'max_flight_time|max_forward_lean|speed_gate' $C21/params/env.yaml | head -20
grep -n -B1 -A1 'func: .*high_speed_double_flight' $C21/params/env.yaml; awk '/high_speed_double_flight:/{f=1} f&&/weight:/{print "flight weight line:",$0; exit}' $C21/params/env.yaml
echo "--- training progress"; grep -a -o -E 'Learning iteration [0-9]+/[0-9]+' /data/zym/roboparty_train/logs/launch/amp_path_c21_flight_leangate10_clamp030_w4_from36500_20261012.log | tail -2
grep -a -E 'Error|Traceback' /data/zym/roboparty_train/logs/launch/amp_path_c21_flight_leangate10_clamp030_w4_from36500_20261012.log | grep -v -E 'descriptor_database|health.proto' | tail -5
ps -eo pid,pgid,etime,cmd | grep -E 'torch.distributed.run|watch_fall5|c21_monitor' | grep -v grep | cut -c1-160
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cat /home/zym/roboparty_train/project_ops/overnight_amp_20261011/c21_status.txt 2>/dev/null; cat /home/zym/roboparty_train/project_ops/overnight_amp_20261011/c21_monitor.out
