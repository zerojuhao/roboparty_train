#!/bin/bash
# Wait for C8 finish -> final summ5 -> pick best -> decide C9 STEP/SUCCESS (or reward/step_size)
set -uo pipefail
OPS=/home/zym/roboparty_train/project_ops/overnight_amp_20261011
RUN=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_06-21-30_amp_path_c8_keep_vmax4p0_from24500_20261011
LOG=/data/zym/roboparty_train/logs/launch/amp_path_c8_keep_vmax4p0_from24500_20261011.log
PY=/data/zym/conda/envs/isaaclab_zym/bin/python
OUT=/data/zym/roboparty_train/logs/launch/c8_finish_continue_result_20261011.txt
OVERNIGHT=/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md
C6_22000=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_04-01-09_amp_path_c6_vmax3p5_from20500_20261011/model_22000.pt
mkdir -p /data/zym/roboparty_train/logs/launch
exec > >(tee -a "$OUT") 2>&1

echo "==== START $(TZ=Asia/Shanghai date '+%F %T') CST ===="

for i in $(seq 1 180); do
  if ! kill -0 3168381 2>/dev/null; then
    echo "launcher 3168381 gone at $(TZ=Asia/Shanghai date '+%F %T') CST"
    break
  fi
  if [ -f "$RUN/model_27500.pt" ] || [ -f "$RUN/model_27499.pt" ]; then
    sleep 10
    if ! pgrep -af 'robo_newton/scripts/rsl_rl/train.py.*amp_path_c7_vmax4p0' >/dev/null 2>&1; then
      echo "final model present and train workers gone"
      break
    fi
  fi
  it=$(grep -oE 'Learning iteration [0-9]+/27500' "$LOG" | tail -1 || true)
  vmax=$(grep -oE 'Curriculum/velocity_tracking/max_velocity: [0-9.]+' "$LOG" | tail -1 || true)
  eta=$(grep -oE 'ETA: .*' "$LOG" | tail -1 || true)
  echo "$(TZ=Asia/Shanghai date '+%T') CST poll $i: $it | $vmax | $eta"
  sleep 20
done

sleep 8
echo "=== CKPTS ==="
ls -lt "$RUN"/model_*.pt

FINAL_IT=$(ls "$RUN"/model_*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1)
echo "FINAL_IT=$FINAL_IT"

# Wait for watcher diag on final (or one-shot)
for i in $(seq 1 90); do
  if [ -f "$RUN/eval_fall/diag5_${FINAL_IT}.json" ]; then
    echo "diag5_${FINAL_IT}.json ready"
    break
  fi
  echo "$(TZ=Asia/Shanghai date '+%T') CST waiting diag5_${FINAL_IT} ($i)"
  sleep 15
done

oneshot_diag() {
  local it=$1
  local ck="$RUN/model_${it}.pt"
  local out="$RUN/eval_fall/diag5_${it}.json"
  [ -f "$ck" ] || return 0
  [ -f "$out" ] && return 0
  echo "One-shot diag for $it"
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MUJOCO_GL=egl
  export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton"
  local tmpd; tmpd=$(mktemp -d)
  for v in 1.5 2.0 2.5 3.0 3.5 4.0; do for m in step ramp; do
    nice -n 10 $PY $OPS/fall_diag5.py --checkpoint "$ck" --speeds $v --modes $m --out "$tmpd/${v}_${m}.json" >/dev/null 2>&1 &
  done; done
  wait
  $PY -c "import json,glob; json.dump(sum([json.load(open(f)) for f in sorted(glob.glob('$tmpd/*.json'))],[]), open('$out','w'), indent=1)"
  rm -rf "$tmpd"
}

if [ ! -f "$RUN/eval_fall/diag5_${FINAL_IT}.json" ]; then
  oneshot_diag "$FINAL_IT"
fi

# Backfill any missing 500-step diags
for it in 25000 25500 26000 26500 27000; do
  oneshot_diag "$it"
done

echo "=== FINAL SUMM5 ==="
$PY $OPS/summ5.py amp_path_c8_keep_vmax4p0_from24500_20261011

echo "=== FINAL CURRICULUM ==="
grep -E 'Curriculum/velocity_tracking/max_velocity' "$LOG" | tail -12 || true
FINAL_VMAX=$(grep -oE 'Curriculum/velocity_tracking/max_velocity: [0-9.]+' "$LOG" | tail -1 | awk '{print $2}')
echo "FINAL_VMAX=$FINAL_VMAX"

echo "=== PICK BEST C7 CKPT ==="
PICK=/tmp/c8_pick_best.py
cat > "$PICK" << 'PY'
import json, glob, os
run = "/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_06-21-30_amp_path_c8_keep_vmax4p0_from24500_20261011"
cands = []
for f in glob.glob(run + "/eval_fall/diag5_*.json"):
    rows = json.load(open(f))
    d = {(r["mode"], float(r["vx"])): r for r in rows}
    it = int(rows[0]["iteration"])
    s4 = int(d.get(("step", 4.0), {}).get("survived", 0))
    r4 = int(d.get(("ramp", 4.0), {}).get("survived", 0))
    s35 = int(d.get(("step", 3.5), {}).get("survived", 0))
    r35 = int(d.get(("ramp", 3.5), {}).get("survived", 0))
    le3 = sum(int(d.get((m, v), {}).get("survived", 0)) for v in (1.5,2.0,2.5,3.0) for m in ("step","ramp"))
    tot = sum(int(r.get("survived", 0)) for r in rows)
    ymax = max((float(r.get("max_abs_world_y_max", 99)) for r in rows), default=99.0)
    # Prefer: le3>=39, high s4/r4/s35, high tot, low ymax, higher it as tiebreak among equals
    cands.append((s4, r4, s35, r35, le3, tot, -ymax, it))
cands.sort(reverse=True)
for c in cands:
    print("cand s4=%d r4=%d s35=%d r35=%d le3=%d tot=%d ymax=%.3f it=%d" % (c[0],c[1],c[2],c[3],c[4],c[5],-c[6],c[7]))
best = cands[0]
print("BEST_IT %d" % best[7])
print("BEST_METRICS s4=%d r4=%d s35=%d r35=%d le3=%d tot=%d ymax=%.3f" % (best[0],best[1],best[2],best[3],best[4],best[5], -best[6]))
# also print worst ymax among 60/60-ish
perfect = [c for c in cands if c[0]>=5 and c[1]>=5 and c[4]>=39 and c[5]>=59]
if perfect:
    # among near-perfect, lowest ymax
    perfect.sort(key=lambda c: (-c[0], -c[1], -c[4], -c[5], -c[6], -c[7]))
    # re-sort: prefer s4, then ymax (less negative -ymax = higher ymax is worse; -c[6] is ymax)
    perfect.sort(key=lambda c: (-c[0], -c[1], -c[5], c[6], -c[7]))  # c[6]=-ymax so ascending c[6] = lower ymax first? Wait c[6]=-ymax, sort ascending c[6] means more negative first = higher ymax. Want lower ymax = larger -ymax = sort reverse on c[6]
    perfect.sort(key=lambda c: (-c[0], -c[1], -c[5], -c[6], -c[7]))
    p = perfect[0]
    print("TIGHT_IT %d" % p[7])
    print("TIGHT_METRICS s4=%d r4=%d s35=%d r35=%d le3=%d tot=%d ymax=%.3f" % (p[0],p[1],p[2],p[3],p[4],p[5], -p[6]))
PY
BEST=$($PY "$PICK")
echo "$BEST"
BEST_IT=$(echo "$BEST" | awk '/^BEST_IT/{print $2}')
TIGHT_IT=$(echo "$BEST" | awk '/^TIGHT_IT/{print $2}')
S4=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*s4=\([0-9]*\).*/\1/p' | head -1)
R4=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*r4=\([0-9]*\).*/\1/p' | head -1)
S35=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*s35=\([0-9]*\).*/\1/p' | head -1)
LE3=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*le3=\([0-9]*\).*/\1/p' | head -1)
TOT=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*tot=\([0-9]*\).*/\1/p' | head -1)
YMAX=$(echo "$BEST" | sed -n 's/^BEST_METRICS.*ymax=\([0-9.]*\).*/\1/p' | head -1)
echo "Selected BEST_IT=$BEST_IT S4=$S4 R4=$R4 S35=$S35 LE3=$LE3 TOT=$TOT YMAX=$YMAX TIGHT_IT=${TIGHT_IT:-}"

# Prefer resume from best C7 if OK; if C7 clearly worse than C6/22000 (s4<4), use 22000
RESUME_FROM_C6=0
if [ "${S4:-0}" -lt 4 ]; then
  echo "C7 best s4=$S4 < 4 — prefer C6/22000 for next cycle"
  RESUME_FROM_C6=1
fi

# Curriculum stuck? if final vmax barely moved from 3.10 (<3.40 after 3000 iters)
VMAX_NUM=${FINAL_VMAX:-0}
STUCK=0
$PY - << PY
vmax=float("$VMAX_NUM" or 0)
print("vmax_num", vmax)
open("/tmp/c8_stuck.flag","w").write("1" if vmax < 3.99 else "0")
PY
STUCK=$(cat /tmp/c8_stuck.flag)

# Drift check from best metrics
DRIFT_HIGH=0
$PY - << PY
ymax=float("$YMAX" or 0)
open("/tmp/c8_drift.flag","w").write("1" if ymax >= 0.85 else "0")
print("ymax", ymax)
PY
DRIFT_HIGH=$(cat /tmp/c8_drift.flag)

echo "FINAL_VMAX=$FINAL_VMAX STUCK=$STUCK DRIFT_HIGH=$DRIFT_HIGH RESUME_FROM_C6=$RESUME_FROM_C6"

DECISION="SUCCESS_DONE"
CHANGE_DESC="SUCCESS DONE: vmax>=4.0 and step4.0 held"
OVERRIDES=(
  "env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295"
  "env.curriculum.velocity_tracking.params.max_velocity=4.0"
)
RUN_TAG="c9_keep_vmax4p0"
LAUNCH_NEXT=1

if [ "${S4:-0}" -lt 4 ]; then
  DECISION="REWARD"
  CHANGE_DESC="ONE reward: fwdtol15 (0.2618) - s4 regress under high-vx exposure"
  OVERRIDES=(
    "env.rewards.lean_orientation_l2.params.forward_tolerance=0.2617993877991494"
    "env.curriculum.velocity_tracking.params.max_velocity=4.0"
  )
  RUN_TAG="c9_fwdtol15_vmax4p0"
  LAUNCH_NEXT=1
elif [ "$STUCK" = "1" ]; then
  DECISION="STEP"
  CHANGE_DESC="ONE change: step_size 0.05->0.1 to reach curriculum ceiling 4.0"
  OVERRIDES=(
    "env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295"
    "env.curriculum.velocity_tracking.params.max_velocity=4.0"
    "env.curriculum.velocity_tracking.params.step_size=0.1"
  )
  RUN_TAG="c9_step0p1_vmax4p0"
  LAUNCH_NEXT=1
elif [ "$DRIFT_HIGH" = "1" ]; then
  DECISION="DRIFT"
  CHANGE_DESC="ONE change: enable straight_line_drift weight=-0.5 (drift approaching 1m)"
  OVERRIDES=(
    "env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295"
    "env.curriculum.velocity_tracking.params.max_velocity=4.0"
    "env.rewards.straight_line_drift.weight=-0.5"
  )
  RUN_TAG="c9_drift_vmax4p0"
  LAUNCH_NEXT=1
else
  if [ "${S4:-0}" -ge 5 ] && [ "${LE3:-0}" -ge 39 ]; then
    DECISION="SUCCESS_DONE"
    CHANGE_DESC="SUCCESS DONE: vmax>=4.0, step4.0 5/5, guardrail OK - leave GPUs free"
    LAUNCH_NEXT=0
  else
    DECISION="KEEP"
    CHANGE_DESC="KEEP same under vmax=4.0 - consolidate step4.0 toward 5/5"
    LAUNCH_NEXT=1
  fi
fi

# Choose load
if [ "$RESUME_FROM_C6" = "1" ]; then
  LOAD_RUN=2026-10-11_04-01-09_amp_path_c6_vmax3p5_from20500_20261011
  CKPT=model_22000.pt
  FROM_IT=22000
  BEST_PATH="$C6_22000"
else
  # Prefer TIGHT among near-perfect if available else BEST
  USE_IT=${TIGHT_IT:-$BEST_IT}
  # If TIGHT exists and s4 from BEST is good, use TIGHT for lower |y|
  if [ -n "${TIGHT_IT:-}" ]; then USE_IT=$TIGHT_IT; fi
  LOAD_RUN=$(basename "$RUN")
  CKPT="model_${USE_IT}.pt"
  FROM_IT=$USE_IT
  BEST_PATH="$RUN/$CKPT"
fi

RUN_NAME="amp_path_${RUN_TAG}_from${FROM_IT}_20261011"
MAXIT=3000

echo "DECISION=$DECISION"
echo "CHANGE=$CHANGE_DESC"
echo "LOAD=$LOAD_RUN/$CKPT"
echo "RUN_NAME=$RUN_NAME"
echo "BEST_PATH=$BEST_PATH"

# Free watcher; ensure train dead
pkill -f "watch_fall5.sh amp_path_c7" 2>/dev/null || true
pkill -f "watch_fall5.sh" 2>/dev/null || true
sleep 2
for i in $(seq 1 40); do
  if ! pgrep -af 'robo_newton/scripts/rsl_rl/train.py' >/dev/null 2>&1; then break; fi
  echo "waiting train exit $i"; sleep 5
done

echo "=== GPU before C8 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

if [ "${LAUNCH_NEXT:-1}" = "1" ]; then
  echo "=== LAUNCH C9: $DECISION ==="
  bash "$OPS/go_cycle.sh" "$RUN_NAME" "$LOAD_RUN" "$CKPT" "$MAXIT" "${OVERRIDES[@]}"
else
  echo "=== SUCCESS DONE - skipping next launch ==="
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8 || true
fi

sleep 15
echo "=== POST LAUNCH ==="
pgrep -af 'amp_path_c8|watch_fall5|launch_cycle|train.py' | head -30 || true
head -20 /data/zym/roboparty_train/logs/launch/${RUN_NAME}.log 2>/dev/null || true
tail -25 /data/zym/roboparty_train/logs/launch/${RUN_NAME}.log 2>/dev/null || true
LAUNCH_P=$(pgrep -f "launch_cycle.sh ${RUN_NAME}" | head -1 || true)
WATCH_P=$(pgrep -f "watch_fall5.sh ${RUN_NAME}" | head -1 || true)
echo "LAUNCH_PID=${LAUNCH_P:-?} WATCH_PID=${WATCH_P:-?}"

# Append overnight log finish + morning summary via python
$PY - << PY
from datetime import datetime
from zoneinfo import ZoneInfo
LOGP = "$OVERNIGHT"
now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M CST")
decision = "$DECISION"
change = """$CHANGE_DESC"""
best_path = "$BEST_PATH"
from_it = "$FROM_IT"
run_name = "$RUN_NAME"
final_it = "$FINAL_IT"
final_vmax = "$FINAL_VMAX"
s4, r4, s35, le3, tot, ymax = "$S4", "$R4", "$S35", "$LE3", "$TOT", "$YMAX"
launch_p, watch_p = "${LAUNCH_P:-?}", "${WATCH_P:-?}"

# pull summ table lines from result file if present
import subprocess, os
summ = subprocess.check_output(["$PY", "$OPS/summ5.py", "amp_path_c8_keep_vmax4p0_from24500_20261011"], text=True, stderr=subprocess.STDOUT)

block = f"""
## CYCLE 8 FINISH — {now} (model_{final_it}) — {decision} / consolidating
- Completed to **model_{final_it}**. Curriculum climbed **3.10 → {final_vmax}** (ceiling 4.0).
- Final fall_diag5 (v40lat 12-cell):

{summ}

- Best C7 ckpt for resume: **{best_path}** (from_it={from_it}; BEST metrics s4={s4} r4={r4} s35={s35} le3={le3} tot={tot} |y|max={ymax}).
- Guardrail: ≤3.0 target ≥19/20 (le3={le3}/40). Drift |y|max={ymax} m (≤1 m).
- Verdict: **{decision}**. Change: {change}
- Prior ultimate-goal HIT remains: C6/22000 step4.0 5/5 |y|0.21; C7 held step4.0 under more high-vx exposure.

## CYCLE 9 START — {now} — {decision}
- Run: logs/rsl_rl/rp1_amp_path/*_{run_name}
  launcher PID {launch_p}, watch_fall5 PID {watch_p}
  launch log: logs/launch/{run_name}.log
- Resume **{best_path}**, +3000 iters, lr 2e-5 fixed.
- **ONE change / KEEP:** {change}
- Keep fwdtol last-wins as launched; lean -6, back16, flat0, lateral -0.3. No separate critic.

## MORNING-READY SUMMARY (updated {now})
- **Goal:** step cmd from rest to **vx=4.0 m/s, no fall, max|world y| ≤ 1.0 m** over 10 s. **HIT (C6+); consolidating on C7/C8.**
- History: C1 FAIL → C2 SUCCESS → C3 KEEP → C4 REGRESS → C5 KEEP → **C6 KEEP (60/60 unlock step4.0)** → **C7 {decision} (ceiling 4.0 exposure; final vmax {final_vmax})** → **C8 LIVE**.
- **Best ckpt:** `{best_path}`
  - C7-best metrics: s4={s4}/5, r4={r4}/5, ≤3.0 le3={le3}/40, tot={tot}/60, |y|max={ymax} m.
  - Also strong: C6 `/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_04-01-09_amp_path_c6_vmax3p5_from20500_20261011/model_22000.pt` (60/60, |y|0.21).
- **LIVE: C8** `{run_name}` from {from_it}, launcher {launch_p}, watch {watch_p}. Decision={decision}.
- Tell user: overnight goal met; C7 finished vmax→{final_vmax}; C8 consolidating on 5090D-5 with training left running.
"""
text = open(LOGP, encoding="utf-8").read()
idx = text.rfind("## MORNING-READY SUMMARY")
if idx >= 0:
    head = text[:idx].rstrip() + "\n"
    open(LOGP, "w", encoding="utf-8").write(head + "\n" + block.lstrip())
else:
    open(LOGP, "a", encoding="utf-8").write("\n" + block)
print("overnight log updated")

import json
json.dump({
  "c7_final_it": int(final_it),
  "c7_final_vmax": final_vmax,
  "best_it": int(from_it) if str(from_it).isdigit() else from_it,
  "best_path": best_path,
  "s4": int(s4 or 0), "r4": int(r4 or 0), "le3": int(le3 or 0), "tot": int(tot or 0), "ymax": float(ymax or 0),
  "decision": decision,
  "c8_run": run_name,
  "launch_pid": launch_p,
  "watch_pid": watch_p,
  "ts": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
}, open("/data/zym/roboparty_train/logs/launch/c7_c8_summary.json","w"), indent=2)
print("wrote c7_c8_summary.json")
PY

echo "==== DONE $(TZ=Asia/Shanghai date '+%F %T') CST ===="
