#!/usr/bin/env python3
# C21 monitor: per-eval log append, abort rule, finish table + BEST + video.
import glob, json, os, subprocess, time, datetime
NAME = 'amp_path_c21_flight_' + 'leangate10_clamp030_w4_from36500_20261012'
ROOT = '/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path'
LOG = '/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md'
STATUS = '/home/zym/roboparty_train/project_ops/overnight_amp_20261011/c21_status.txt'
VID = '/data/zym/roboparty_train/videos/amp_path_4mps_flight_20261011'
PY = '/data/zym/conda/envs/isaaclab_zym/bin/python'
BASE = dict(tot=60, s4=5, r4=5, y=0.29, roll=0.359, df=0.4725, ms=0.16, msx=0.18, lean=30.6)
BASE_IT = 36500
def now(): return (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M CST')
def st(msg):
    with open(STATUS, 'a') as f: f.write(f'{now()} {msg}\n')
def app(txt):
    with open(LOG, 'a') as f: f.write(txt.rstrip('\n') + '\n\n')
def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
def train_alive(): return bool(sh(f"pgrep -f 'train.py.*{NAME}'").strip())
def f(x, p=3):
    return 'nan' if x is None or (isinstance(x, float) and x != x) else f'{x:.{p}f}'
def metrics(path):
    rows = json.load(open(path)); d = {(r['mode'], float(r['vx'])): r for r in rows}
    s4 = d.get(('step', 4.0), {}); r4 = d.get(('ramp', 4.0), {})
    tot = sum(r['survived'] for r in rows); n = sum(r['trials'] for r in rows)
    ys = [r['max_abs_world_y_max'] for r in rows if 'max_abs_world_y_max' in r and r['vx'] >= 2.0]
    bad_cells = [f"{m[0]}{v:g}:{r['survived']}/5" for (m, v), r in sorted(d.items()) if r['survived'] < r['trials']]
    g = lambda k: s4.get(k)
    return dict(it=rows[0]['iteration'], tot=tot, n=n, s4=s4.get('survived', 0), r4=r4.get('survived', 0),
                y=max(ys) if ys else None, roll=g('mean_abs_shoulder_roll_after2s_med'),
                pitch=g('shoulder_pitch_range_after2s_med'), df=g('double_flight_fraction_after2s_med'),
                ms=g('max_double_flight_s_after2s_med'), msx=g('max_double_flight_s_after2s_max'),
                lean=g('max_fwd_lean_deg_pre_fall'), bad=bad_cells)
def row(m):
    return (f"| {m['it']} | {m['tot']}/{m['n']} | {m['s4']}/5 | {m['r4']}/5 | {f(m['y'],2)} | {f(m['roll'])} | "
            f"{f(m['df'],4)} | {f(m['ms'])} | {f(m['msx'])} | {f(m['lean'],1)} |")
HDR = ("| iter | TOT | s4 | r4 | |y|max | step4 |roll| | dflight_s4 | max_s_s4 | max_smax_s4 | s4 fwdlean_max° |\n"
       "|------|-----|----|----|-------|-------------|-------------|----------|-------------|----------------|\n"
       f"| 36500 (C18 base) | 60/60 | 5/5 | 5/5 | 0.29 | 0.359 | 0.4725 | 0.160 | 0.180 | 30.6 |")
def table(ms): return HDR + '\n' + '\n'.join(row(m) for m in ms)
def kill_all():
    sh(f"pkill -TERM -f '{NAME}'"); time.sleep(25); sh(f"pkill -KILL -f '{NAME}'"); time.sleep(10)
    for _ in range(30):
        used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()
        if used and all(int(u) < 100 for u in used): return 'FREE (0 MiB x8)', used
        time.sleep(10)
    return 'NOT FREE', used
def pick_best(ms):
    c = [m for m in ms if m['it'] != BASE_IT and m['s4'] == 5 and (m['y'] or 9) <= 1.0]
    if not c: return None
    top = max(m['tot'] for m in c); c = [m for m in c if m['tot'] == top]
    k = lambda m: ((m['msx'] or 0), (m['ms'] or 0), (m['df'] or 0), -(m['y'] or 9))
    return max(c, key=k)
def main():
    st('monitor start')
    run = None
    while not run:
        g = sorted(glob.glob(f'{ROOT}/*{NAME}'));
        run = g[-1] if g else None
        if not run: time.sleep(20)
    st(f'run {run}')
    seen = {}; ms = []; consec = 0; dead_since = None
    while True:
        files = sorted(glob.glob(f'{run}/eval_fall/diag5_*.json'), key=lambda p: int(p.split('_')[-1][:-5]))
        for p in files:
            if p in seen: continue
            try: m = metrics(p)
            except Exception as e: st(f'parse fail {p} {e}'); continue
            seen[p] = 1; ms.append(m)
            isbase = m['it'] == BASE_IT
            badf = (m['s4'] <= 2 or m['tot'] < 55) and not isbase
            consec = consec + 1 if badf else (consec if isbase else 0)
            flags = []
            if m['msx'] is not None and m['msx'] > BASE['msx']: flags.append('FLIGHT_UP')
            if m['s4'] == 5: flags.append('S4_HOLD')
            if m['y'] is not None and m['y'] <= 1.0: flags.append('Y_OK')
            if badf: flags.append(f'ABORT_CAND({consec})')
            if isbase: flags.append('BASE_REEVAL')
            app(f"## CYCLE 21 @{m['it']} — {now()}\n"
                f"- fall_diag5: **{m['tot']}/{m['n']}**, s4 **{m['s4']}/5**, r4 {m['r4']}/5, |y|max **{f(m['y'],2)}**, step4 |roll| **{f(m['roll'])}**, pitch_rng **{f(m['pitch'])}**, s4 fwd lean max **{f(m['lean'],1)}°** (C18 base 30.6°)\n"
                f"- double_flight step4: frac **{f(m['df'],4)}** (base 0.4725), max_s **{f(m['ms'])}** (base 0.160), max_smax **{f(m['msx'])}** (base 0.180)\n"
                f"- failing cells: {', '.join(m['bad']) or 'none'}\n- flags: {', '.join(flags)}")
            st(f"EVAL {m['it']} tot={m['tot']} s4={m['s4']} y={f(m['y'],2)} df={f(m['df'],4)} ms={f(m['ms'])} msx={f(m['msx'])} lean={f(m['lean'],1)} consec_bad={consec}")
            if consec >= 2:
                st('ABORT triggered; killing'); gpu, used = kill_all()
                app(f"## CYCLE 21 ABORT — {now()}\n- Run: `{os.path.basename(run)}` — STOPPED early.\n"
                    f"- Reason: 2 consecutive bad evals (s4<=2/5 or TOT<55/60); last @{m['it']} tot={m['tot']}/60 s4={m['s4']}/5.\n"
                    f"- Killed train+watch (pkill -f run name). GPUs: **{gpu}** {used}.\n- Best remains C18 **model_36500**. No auto C22.\n\n" + table(ms))
                st(f'ABORT_DONE gpu={gpu}'); return
        alive = train_alive()
        if not alive:
            dead_since = dead_since or time.time()
            ck = sorted(glob.glob(f'{run}/model_*.pt'), key=lambda p: int(p.split('_')[-1][:-3]))
            need = [c for c in ck if (lambda it: it % 500 == 0 or it % 500 >= 498)(int(c.split('_')[-1][:-3]))
                    and not os.path.exists(f"{run}/eval_fall/diag5_{c.split('_')[-1][:-3]}.json")]
            if not need or time.time() - dead_since > 3600: break
        time.sleep(30)
    st('train finished; finalizing')
    sh("pkill -TERM -f 'watch_fall5.sh " + NAME + "'"); time.sleep(5)
    gpu, used = kill_all()
    best = pick_best(ms)
    vid = None
    if best:
        it = best['it']; out = f'{VID}/c21_{it}_step_vx4p0_10s'
        os.makedirs(VID, exist_ok=True)
        st(f'BEST {it}; recording video')
        cmd = (f"cd /home/zym/roboparty_train/robo_newton && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MUJOCO_GL=egl "
               f"PYTHONPATH=/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton "
               f"{PY} scripts/mujoco/sim2sim_rp1_amp.py --checkpoint {run}/model_{it}.pt --forward-speed 4 --duration 10 "
               f"--fps 50 --width 1920 --height 1080 --output {out} > {VID}/c21_{it}_step4.log 2>&1")
        subprocess.run(cmd, shell=True)
        vid = out + '.mp4' if os.path.exists(out + '.mp4') else None
    btxt = (f"**BEST C21: model_{best['it']}** — tot {best['tot']}/60, s4 {best['s4']}/5, |y|max {f(best['y'],2)}, df {f(best['df'],4)}, max_s {f(best['ms'])}, max_smax {f(best['msx'])}, roll {f(best['roll'])}, s4 lean max {f(best['lean'],1)}°"
            if best else '**No C21 ckpt qualifies** (s4 5/5 and |y|<=1)')
    parts = [f"## CYCLE 21 FINISH — {now()}",
             f"- Run: `{os.path.basename(run)}` (36500→end). Change: lean gate 10° on high_speed_double_flight (w4, clamp 0.30), vmax4.0 + fwdtol10 restored.",
             f"- {btxt}"]
    if best: parts.append(f"- BEST path: `{run}/model_{best['it']}.pt` (vs C18 base model_36500: 60/60, df 0.4725, max_s 0.16, max_smax 0.18, |y| 0.29)")
    parts.append(f"- Video (remote): `{vid}`")
    parts.append(f"- Watch+train stopped. GPUs: **{gpu}** {used}. No auto C22.")
    app('\n'.join(parts) + '\n\n' + table(ms))
    st(f"FINISH_DONE best={best['it'] if best else None} vid={vid} gpu={gpu}")
main()
