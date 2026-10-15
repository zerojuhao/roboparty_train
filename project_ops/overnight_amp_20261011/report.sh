#!/bin/bash
/data/zym/conda/envs/isaaclab_zym/bin/python /home/zym/roboparty_train/project_ops/overnight_amp_20261011/summ.py "$@"
for s in "$@"; do
  L=$(ls -t /data/zym/roboparty_train/logs/launch/*${s}*.log 2>/dev/null | grep -v watch | head -1)
  [ -n "$L" ] && { echo "## trend $L"; bash /home/zym/roboparty_train/project_ops/amp_trend.sh $L 250 | tail -6; }
done