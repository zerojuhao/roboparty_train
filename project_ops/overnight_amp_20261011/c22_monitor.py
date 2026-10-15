#!/usr/bin/env python3
"""C22 monitor: per-eval log, abort (2 consec s4<=2 or TOT<55), finish summ5+flight+BEST+video. No auto C23."""
import glob, json, os, subprocess, time, datetime, shutil

NAME = 'amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012'
ROOT = '/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path'
LOG = '/home/zym/roboparty_train/project_ops/OVERNIGHT_AMP_FALL_LOG_20261011.md'
STATUS = '/home/zym/roboparty_train/project_ops/overnight_amp_20261011/c22_status.txt'
OPS = '/home/zym/roboparty_train/project_ops/overnight_amp_20261011'
VID = '/data/zym/roboparty_train/videos/amp_path_4mps_flight_20261011'
PYBIN = '/data/zym/conda/envs/isaaclab_zym/bin/python'
BASE = dict(tot=60, s4=5, r4=5, y=0.29, roll=0.359, df=0.4725, ms=0.16, msx=0.18, lean=30.6)
BASE_IT = 36500
C18_MSX = 0.180

def now():
    return (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M CST')

def st(msg):
    with open(STATUS, 'a') as f:
        f.write(f'{now()} {msg}\n')
    print(msg, flush=True)

def app(txt):
    with open(LOG, 'a') as f:
        f.write(txt.rstrip('\n') + '\n\n')

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def train_alive():
    # Avoid self-match: require train.py AND the run name in the cmdline
    out = sh("pgrep -af 'rsl_rl/train.py' || true")
    return any(NAME in line for line in out.splitlines())

def f(x, p=3):
    return 'nan' if x is None or (isinstance(x, float) and x != x) else f'{x:.{p}f}'

def metrics(path):
    rows = json.load(open(path))
    d = {(r['mode'], float(r['vx'])): r for r in rows}
    s4 = d.get(('step', 4.0), {})
    r4 = d.get(('ramp', 4.0), {})
    tot = sum(r['survived'] for r in rows)
    n = sum(r['trials'] for r in rows)
    ys = [r['max_abs_world_y_max'] for r in rows if 'max_abs_world_y_max' in r and r['vx'] >= 2.0]
    bad_cells = [f"{m[0]}{v:g}:{r['survived']}/5" for (m, v), r in sorted(d.items()) if r['survived'] < r['trials']]
    g = lambda k: s4.get(k)
    return dict(
        it=rows[0]['iteration'], tot=tot, n=n,
        s4=s4.get('survived', 0), r4=r4.get('survived', 0),
        y=max(ys) if ys else None,
        roll=g('mean_abs_shoulder_roll_after2s_med'),
        pitch=g('shoulder_pitch_range_after2s_med'),
        df=g('double_flight_fraction_after2s_med'),
        ms=g('max_double_flight_s_after2s_med'),
        msx=g('max_double_flight_s_after2s_max'),
        lean=g('max_fwd_lean_deg_pre_fall'),
        bad=bad_cells,
    )

def row(m):
    return (f"| {m['it']} | {m['tot']}/{m['n']} | {m['s4']}/5 | {m['r4']}/5 | {f(m['y'],2)} | {f(m['roll'])} | "
            f"{f(m['df'],4)} | {f(m['ms'])} | {f(m['msx'])} | {f(m['lean'],1)} |")

HDR = ("| iter | TOT | s4 | r4 | |y|max | step4 |roll| | dflight_s4 | max_s_s4 | max_smax_s4 | s4 fwdlean_max° |\n"
       "|------|-----|----|----|-------|-------------|-------------|----------|-------------|----------------|\n"
       "| 36500 (C18 base) | 60/60 | 5/5 | 5/5 | 0.29 | 0.359 | 0.4725 | 0.160 | 0.180 | 30.6 |")

def table(ms):
    return HDR + '\n' + '\n'.join(row(m) for m in ms)

def kill_all():
    sh(f"pkill -TERM -f 'watch_fall5.sh {NAME}' || true")
    sh(f"pkill -TERM -f 'rsl_rl/train.py' || true")
    sh(f"pkill -TERM -f 'torch.distributed.run' || true")
    time.sleep(25)
    sh(f"pkill -KILL -f 'watch_fall5.sh {NAME}' || true")
    sh(f"pkill -KILL -f 'rsl_rl/train.py' || true")
    sh(f"pkill -KILL -f 'torch.distributed.run' || true")
    time.sleep(10)
    for _ in range(30):
        used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()
        if used and all(int(u) < 100 for u in used):
            return 'FREE (0 MiB x8)', used
        time.sleep(10)
    return 'NOT FREE', used

def pick_best(ms):
    c = [m for m in ms if m['it'] != BASE_IT and m['s4'] == 5 and m['tot'] == 60 and (m['y'] or 9) <= 1.0]
    if not c:
        # relax tot==60
        c = [m for m in ms if m['it'] != BASE_IT and m['s4'] == 5 and (m['y'] or 9) <= 1.0]
    if not c:
        return None
    top = max(m['tot'] for m in c)
    c = [m for m in c if m['tot'] == top]
    k = lambda m: ((m['msx'] or 0), (m['ms'] or 0), (m['df'] or 0), -(m['y'] or 9))
    return max(c, key=k)

def ensure_video(run, it):
    os.makedirs(VID, exist_ok=True)
    out = f'{VID}/c22_{it}_step_vx4p0_10s'
    mp4 = out + '.mp4'
    if os.path.isfile(mp4) and os.path.getsize(mp4) > 1000:
        st(f'video exists {mp4}')
        return mp4
    logp = f'{VID}/c22_{it}_step4.log'
    cmd = (
        f"cd /home/zym/roboparty_train/robo_newton && "
        f"CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MUJOCO_GL=egl "
        f"PYTHONPATH=/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton "
        f"{PYBIN} scripts/mujoco/sim2sim_rp1_amp.py "
        f"--checkpoint {run}/model_{it}.pt --forward-speed 4 --duration 10 "
        f"--fps 50 --width 1920 --height 1080 --output {out} > {logp} 2>&1"
    )
    st(f'recording video it={it}')
    subprocess.run(cmd, shell=True)
    if os.path.isfile(mp4) and os.path.getsize(mp4) > 1000:
        return mp4
    # sometimes written without .mp4
    if os.path.isfile(out) and os.path.getsize(out) > 1000:
        os.rename(out, mp4)
        return mp4
    return None

def write_scp_hint(vid):
    """Drop a small marker so box/ZERO can scp; also copy into a known pull path."""
    hint = f'{OPS}/c22_video_ready.txt'
    with open(hint, 'w') as fh:
        fh.write(f'{vid}\n')
    return hint

def finalize(run, ms, aborted=False, abort_reason=''):
    st('finalizing' + (' ABORT' if aborted else ' FINISH'))
    sh(f"pkill -TERM -f 'watch_fall5.sh {NAME}' || true")
    time.sleep(5)
    gpu, used = kill_all()
    best = pick_best(ms)
    any_gain = any((m['msx'] or 0) > C18_MSX for m in ms if m['it'] != BASE_IT)
    best_msx = max(((m['msx'] or 0) for m in ms if m['it'] != BASE_IT), default=0.0)

    summ = subprocess.run([PYBIN, f'{OPS}/summ5.py', NAME], capture_output=True, text=True)
    summ_txt = summ.stdout or summ.stderr or '(summ5 empty)'

    vid = None
    if best and not aborted:
        vid = ensure_video(run, best['it'])
        if vid:
            write_scp_hint(vid)

    if aborted:
        app(f"## CYCLE 22 ABORT — {now()}\n"
            f"- Run: `{os.path.basename(run)}` — STOPPED early.\n"
            f"- Reason: {abort_reason}\n"
            f"- Killed train+watch. GPUs: **{gpu}** {used}.\n"
            f"- Best remains C18 **model_36500**. **No auto C23.**\n\n"
            + table(ms) + "\n\n### summ5\n```\n" + summ_txt.rstrip() + "\n```\n")
        st(f'ABORT_DONE gpu={gpu}')
        with open(f'{OPS}/c22_done.txt', 'w') as fh:
            fh.write(f'ABORT\nGPU={gpu}\nBEST=C18_36500\nANY_FLIGHT_GAIN=False\n')
        return

    btxt = (
        f"**BEST C22: model_{best['it']}** — tot {best['tot']}/60, s4 {best['s4']}/5, |y|max {f(best['y'],2)}, "
        f"df {f(best['df'],4)}, max_s {f(best['ms'])}, max_smax {f(best['msx'])}, roll {f(best['roll'])}, "
        f"s4 lean max {f(best['lean'],1)}°"
        if best else '**No C22 ckpt qualifies** (s4 5/5 and |y|<=1)'
    )
    flight_beat = any_gain
    overall = (
        f"C22 BEST improves flight beyond C18 max_smax=0.18s (peak {best_msx:.3f}s)"
        if flight_beat else
        f"C18 model_36500 remains overall best on flight (C22 peak max_smax={best_msx:.3f}s; C18=0.180s)"
    )
    verdict = 'FINISH' if flight_beat else 'KEEP-no-gain'
    parts = [
        f"## CYCLE 22 {verdict} — {now()}",
        f"- Run: `{os.path.basename(run)}` (36500→~39000). Change vs C21: lean gate **10°→15°** (`max_forward_lean=0.2618`). Weight 4.0, clamp 0.30, speed_gate 1.5, vmax4.0, fwdtol10 kept. Marker `C22_LEAN_GATE_15`.",
        f"- {btxt}",
        f"- **Honest vs C18 36500**: C18 max_smax=**0.180s**. C22 peak max_smax=**{best_msx:.3f}s**. **Flight beat 0.18s? {'YES' if flight_beat else 'NO'}**. {overall}",
    ]
    if best:
        parts.append(f"- BEST path: `{run}/model_{best['it']}.pt`")
    parts.append(f"- Video (remote): `{vid}`")
    parts.append(f"- Watch+train stopped. GPUs: **{gpu}** {used}. **No auto C23.**")
    parts.append("")
    parts.append("### Step-4 flight table")
    parts.append(table(ms))
    parts.append("")
    parts.append("### summ5")
    parts.append("```")
    parts.append(summ_txt.rstrip())
    parts.append("```")
    app('\n'.join(parts))
    st(f"FINISH_DONE best={best['it'] if best else None} vid={vid} gpu={gpu} flight_beat={flight_beat}")
    with open(f'{OPS}/c22_done.txt', 'w') as fh:
        fh.write(
            f"{verdict}\nBEST_IT={best['it'] if best else None}\nVID={vid}\nGPU={gpu}\n"
            f"ANY_FLIGHT_GAIN={flight_beat}\nBEST_MSX={best_msx}\n"
        )
    with open('/tmp/c22_wrap_report.txt', 'w') as fh:
        fh.write('\n'.join(parts) + '\n')

def main():
    open(STATUS, 'w').write(f'{now()} monitor start\n')
    st('monitor start')
    run = None
    while not run:
        g = sorted(glob.glob(f'{ROOT}/*{NAME}'))
        run = g[-1] if g else None
        if not run:
            time.sleep(20)
    st(f'run {run}')
    seen = {}
    ms = []
    consec = 0
    dead_since = None
    while True:
        files = sorted(
            glob.glob(f'{run}/eval_fall/diag5_*.json'),
            key=lambda p: int(p.split('_')[-1][:-5]),
        )
        for p in files:
            if p in seen:
                continue
            try:
                m = metrics(p)
            except Exception as e:
                st(f'parse fail {p} {e}')
                continue
            seen[p] = 1
            ms.append(m)
            isbase = m['it'] == BASE_IT
            badf = (m['s4'] <= 2 or m['tot'] < 55) and not isbase
            consec = consec + 1 if badf else (consec if isbase else 0)
            flags = []
            if m['msx'] is not None and m['msx'] > BASE['msx']:
                flags.append('FLIGHT_UP')
            if m['s4'] == 5:
                flags.append('S4_HOLD')
            if m['y'] is not None and m['y'] <= 1.0:
                flags.append('Y_OK')
            if badf:
                flags.append(f'ABORT_CAND({consec})')
            if isbase:
                flags.append('BASE_REEVAL')
            app(
                f"## CYCLE 22 @{m['it']} — {now()}\n"
                f"- fall_diag5: **{m['tot']}/{m['n']}**, s4 **{m['s4']}/5**, r4 {m['r4']}/5, |y|max **{f(m['y'],2)}**, "
                f"step4 |roll| **{f(m['roll'])}**, pitch_rng **{f(m['pitch'])}**, s4 fwd lean max **{f(m['lean'],1)}°** (C18 base 30.6°)\n"
                f"- double_flight step4: frac **{f(m['df'],4)}** (base 0.4725), max_s **{f(m['ms'])}** (base 0.160), "
                f"max_smax **{f(m['msx'])}** (base 0.180)\n"
                f"- failing cells: {', '.join(m['bad']) or 'none'}\n- flags: {', '.join(flags)}"
            )
            st(
                f"EVAL {m['it']} tot={m['tot']} s4={m['s4']} y={f(m['y'],2)} "
                f"df={f(m['df'],4)} ms={f(m['ms'])} msx={f(m['msx'])} lean={f(m['lean'],1)} consec_bad={consec}"
            )
            if consec >= 2:
                reason = (
                    f"2 consecutive bad evals (s4<=2/5 or TOT<55/60); last @{m['it']} "
                    f"tot={m['tot']}/60 s4={m['s4']}/5"
                )
                st('ABORT triggered; killing')
                finalize(run, ms, aborted=True, abort_reason=reason)
                return
        alive = train_alive()
        if not alive:
            dead_since = dead_since or time.time()
            ck = sorted(glob.glob(f'{run}/model_*.pt'), key=lambda p: int(p.split('_')[-1][:-3]))
            need = []
            for c in ck:
                it = int(c.split('_')[-1][:-3])
                if it % 500 == 0 or it % 500 >= 498:
                    if not os.path.exists(f'{run}/eval_fall/diag5_{it}.json'):
                        need.append(c)
            # Wait up to 45min for pending evals after train ends
            if not need or time.time() - dead_since > 2700:
                break
        else:
            dead_since = None
        time.sleep(30)
    finalize(run, ms, aborted=False)

if __name__ == '__main__':
    main()
