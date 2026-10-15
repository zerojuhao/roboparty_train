#!/bin/bash
LOGMD=/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md
B=/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
ROOT=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path
NAME=amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012
RUN=$(ls -dt $ROOT/*${NAME}* 2>/dev/null | head -1)
BAK=$(ls -t $B/mdp/rewards.py.bak_c22_* 2>/dev/null | head -1)
BAKCFG=$(ls -t $B/rp1_amp_env_cfg.py.bak_c22_* 2>/dev/null | head -1)
TS=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M CST')
TRAIN_INFO=$(ps -eo pid,pgid,etime,cmd | grep -E 'torch.distributed.run|rsl_rl/train.py' | grep -v grep | head -3)
WATCH_INFO=$(ps -eo pid,etime,cmd | grep 'watch_fall5' | grep -v grep | head -2)
MON_INFO=$(ps -eo pid,etime,cmd | grep 'c22_monitor' | grep -v grep | head -2)
GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
{
echo "## CYCLE 22 START — $TS — softer lean gate 15° on double-flight"
echo "- From C18 BEST **model_36500** (60/60, s4 5/5, |y|~0.2–0.3, max flight ~0.18s). NOT from C19/C20/C21."
echo "- **Only change vs C21:** \`high_speed_double_flight.max_forward_lean\` **10° → 15°** (\`0.1745 → 0.2617993877991494\` rad). Marker \`C22_LEAN_GATE_15\`."
echo "- Kept: weight **4.0**, clamp **0.30**, speed_gate **1.5**, feet_air_time both-air, vmax **4.0**, step_size **0.3**, lean_orientation fwdtol **10°**, shoulder/AMP settings as C21."
echo "- Resume: \`2026-10-11_15-36-14_amp_path_c18_flight_weight_and_airtime_from34500_20261011\` / \`model_36500.pt\` → max_iterations=2500 (target ~39000)."
echo "- Run name: \`$NAME\`"
echo "- Run dir: \`$RUN\`"
echo "- Procs: train/dist: \`$TRAIN_INFO\` | watch: \`$WATCH_INFO\` | monitor: \`$MON_INFO\`"
echo "- GPUs MiB: $GPU"
echo "- Backups: \`$BAK\`, \`$BAKCFG\` (+ /data/zym/roboparty_train/backups/), log \`.bak_pre_c22_start\`."
echo "- Abort rule: 2 consecutive evals with s4<=2/5 or TOT<55/60 → kill, log ABORT, keep best as C18 36500. **No auto C23.**"
echo "- Code diff (cfg lean gate):"
echo '```diff'
diff -u "$BAKCFG" $B/rp1_amp_env_cfg.py || true
echo "--- rewards.py markers"
diff -u "$BAK" $B/mdp/rewards.py || true
echo '```'
echo
} >> $LOGMD
echo "LOGGED START; RUN=$RUN"
tail -c 2000 $LOGMD
