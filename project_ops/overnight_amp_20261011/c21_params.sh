#!/bin/bash
cd /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/
for d in *c1[5-9]_* *c2[0-9]_*; do
  echo "== $d"
  grep -h -E 'forward_tolerance|max_velocity:|max_flight_time|max_forward_lean|style_reward_scale|task_style_lerp|style_reward_weight' $d/params/env.yaml $d/params/agent.yaml | tr -s ' ' | tr '\n' ' '
  echo
done
# also find hydra / cmdline recorded
ls /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/*c18_*/ | head; ls /data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/*c18_*/params
grep -n -E 'max_velocity|forward_tolerance' /home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/rp1_amp_env_cfg.py
