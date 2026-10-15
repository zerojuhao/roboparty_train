"""Summarize saved fixed-context push trials without re-running physics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_pufo_stair_push import segment_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory/'report.json').read_text())
    details = {}
    for force, result in report['trials'].items():
        folder = args.directory / f'force_{force}'
        fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
        details[force] = {}
        for label, color in [('before','#4277a8'), ('after','#c45542')]:
            with np.load(folder / f'{label}_traces.npz') as archive:
                trace = {key: archive[key] for key in archive.files}
            t = np.arange(len(trace['root'])) / 50
            tp = np.arange(len(trace['pd'])) / 200
            speed = trace['pd'][:, 0, :12]
            rms = np.sqrt(np.mean(speed.reshape(-1,20,12)**2,axis=(1,2)))
            axes[0].plot(t,trace['root'][:,0],color=color,label=label)
            axes[1].plot(t,trace['root'][:,2],color=color,label=label)
            axes[2].plot(tp[::20]+.05,rms,color=color,label=label)
            axes[3].plot(t,np.max(abs(trace['foot_force']),axis=1),color=color,label=label,alpha=.85)
            axes[4].step(t,trace['treads'].max(axis=1),where='post',color=color,label=label)
            axes[5].plot(t,trace['force'],color=color,label=label)
            end = min(12.,result['common_prefall_end_s'])
            landing = np.flatnonzero((trace['treads'] > 0).any(axis=1) & (trace['up'] > .7))
            landing = landing[(landing/50 >=3) & (landing/50 < end)]
            first = float(landing[0]/50) if len(landing) else None
            events = []
            # Touchdown requires at least 60ms without meaningful load and 40ms contact.
            for foot in range(2):
                contact = trace['foot_force'][:,foot] > 20
                for index in range(3,len(contact)-5):
                    if contact[index:index+2].all() and not contact[index-3:index].any() and 3 <= index/50 < end:
                        events.append(dict(time_s=index/50,foot=['left','right'][foot],
                            tread=int(trace['treads'][index:index+5,foot].max()),
                            previous_vertical_speed_m_s=float(trace['foot_v'][index-1,foot]),
                            peak_next_100ms_n=float(trace['foot_force'][index:index+5,foot].max())))
            own_end = min(16.,result[label]['first_instability_s'] or 16.)
            upright_treads = np.where((trace['up'] > .7)[:,None], trace['treads'], 0)
            both_visited_top = (np.maximum.accumulate(upright_treads,axis=0) >=6).all(axis=1)
            completion = np.flatnonzero(both_visited_top & (t < own_end))
            completed = float(completion[0]/50) if len(completion) else None
            ascent_end = completed if completed is not None else min(12.,own_end)
            tail = (trace['root'][-100:,0] >=1.5) & (trace['root'][-100:,0] < 1.5+report['setup']['terrain']['landing_length_m'])
            tail &= (trace['root'][-100:,2] > 1.3) & (trace['up'][-100:] > .7)
            details[force][label] = dict(first_supported_step_s=first, both_feet_visited_top_s=completed,
                upright_on_landing_last_2s=bool(tail.all()), touchdown_events=events,
                ascent_only=segment_metrics(trace,first,ascent_end) if first is not None else None,
                after_first_step=segment_metrics(trace,first,end) if first is not None else None)
        axes[0].axhline(0,color='gray',linestyle=':',label='first riser')
        labels = ['Root forward X (m)','Root height (m)','Leg speed RMS (rad/s, 100ms)','Max foot net Fz (N, 50Hz)','Supported tread (either foot)','Applied force +X (N)']
        for ax, label in zip(axes, labels):
            ax.set_ylabel(label)
            ax.axvspan(3,12,color='gray',alpha=.08)
            ax.axvline(result['common_prefall_end_s'],color='black',linestyle=':',alpha=.6)
            ax.grid(alpha=.2)
            ax.legend(loc='upper left')
        axes[-1].set_xlabel('Time (s); shaded = push schedule; dotted black = common pre-instability cutoff')
        fig.suptitle(f'Held context row 0 | 15 cm stairs | horizontal torso push {force}')
        fig.tight_layout()
        fig.savefig(folder/'metrics.png',dpi=150)
        plt.close(fig)
    (args.directory/'details.json').write_text(json.dumps(details,indent=2)+'\n')
    lines = ['# Held-context passive stair push: 15 cm', '',
             f"Checkpoints: before {report['setup']['steps'][0]:,} / after {report['setup']['steps'][1]:,} training steps; exact paths in setup.json.",
             'Context row 0 is source frame 50, encoded from next-frame observations; window=5, gamma=0.8.',
             'Heading and velocity actor inputs are zero, matching the supplied deployment hold branch.',
             'Six straight steps, 15 cm rise, 30 cm tread, 2 m width. Start x=-0.45 m, first riser x=0.',
             'Only horizontal +X force at waist_yaw_link COM. Settle 3 s, ramp/hold/ramp until 12 s, release until 16 s.',
             f"Top landing length {report['setup']['terrain']['landing_length_m']} m. Stop-on-landing: {report['setup'].get('stop_on_landing',False)}; root X >=2.1 m removes force permanently.",
             'Peak force and approach schedule are matched, but landing-dependent stopping produces different durations and impulses.',
             'Force is removed on detected instability. No upward force, position correction, playback, or automatic reset.', '',
             '## Matched push interval results', '',
             'Numbers are before -> after, using [3 s, min(12 s, first instability of either model)].', '',
             '| Push | Max supported tread left/right | Instability time (s) | Leg speed RMS (rad/s) | Leg speed P95 (rad/s) | Foot force P95 (N) | Sampled max foot force (N) |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for force, result in report['trials'].items():
        b,a = result['before'],result['after']
        bm,am = b['push_common_prefall'],a['push_common_prefall']
        fmt = lambda key: f"{bm[key]:.2f} -> {am[key]:.2f}" if bm and am else 'N/A'
        lines.append(f"| {force} | {b['max_supported_tread_per_foot']} -> {a['max_supported_tread_per_foot']} | {b['first_instability_s']} -> {a['first_instability_s']} | {fmt('leg_speed_rms_rad_s')} | {fmt('leg_speed_p95_rad_s')} | {fmt('contact_force_p95_n')} | {fmt('contact_force_max_n')} |")
    lines.extend(['', '## Actual ascent intervals', '',
                  'From first upright supported stair contact until both feet have visited tread 6, or until instability/push end if incomplete.',
                  'Intervals differ between policies because climbing speed differs. They exclude later quiet standing on the landing.', '',
                  '| Push / policy | Interval (s) | Both feet visited top (s) | Upright on landing last 2 s | Leg speed RMS / P95 / max (rad/s) | Foot force P95 / max (N) |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for force, pair in details.items():
        for label, values in pair.items():
            m = values['ascent_only']
            if m is None:
                continue
            lines.append(f"| {force} / {label} | {m['interval_s']} | {values['both_feet_visited_top_s']} | {values['upright_on_landing_last_2s']} | {m['leg_speed_rms_rad_s']:.2f} / {m['leg_speed_p95_rad_s']:.2f} / {m['leg_speed_max_rad_s']:.2f} | {m['contact_force_p95_n']:.0f} / {m['contact_force_max_n']:.0f} |")
    lines.extend(['', 'Tread numbers require upward support >10 N and contact within 2.5 cm of the tread top while upright.',
                  'A transient step touch is not proof of completing the staircase. See videos and traces for actual progress.', '',
                  '## Limitations', '',
                  '- CPU Torch/MuJoCo evaluation, not ONNX/hardware or a full reproduction of deployment action filtering and latency.',
                  '- Joint speed / clipped torque sampled at 200 Hz. Foot COM velocity and net vertical contact force sampled at 50 Hz; brief peaks may be missed.',
                  '- This is a before/after training comparison, not a matched-step reward ablation. No acoustic data.',
                  '- Reward differences must be checked in the saved checkpoint configurations; torque is measured, not assumed to be a changed reward.',
                  '- Initial stance comes from source frame 50 with velocities reset to zero. Only the startup pose is assigned.',
                  '- No training jobs or reward settings were changed.', '',
                  '## Artifacts', '',
                  'Each force directory contains comparison.mp4 (before left / after right), metrics.png, report.json and full traces.',
                  'setup.json specifies checkpoints and protocol. details.json contains touchdown diagnostics and post-first-step intervals.', ''])
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
