"""Plot physical push/recovery traces and keep failed recovery metrics distinct."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_pufo_stair_push import segment_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=json.loads((args.directory/'report.json').read_text())
    setup=report['setup']
    detail={}
    lines=['# PUFO physical push and recovery','',
        f"Baseline {setup['steps'][0]:,} steps (joint velocity weight {setup['joint_vel_weights'][0]}); new {setup['steps'][1]:,} steps (weight {setup['joint_vel_weights'][1]}).",
        'Deployment context row 0, source frame 50, window 5 / gamma 0.8; heading and velocity inputs zero.',
        setup['force_protocol'],
        'No pose/velocity assignment after initial stance. No reset, lifting, or reference playback.',
        f"Down criterion: {setup['down_rule']}",f"Standing criterion: {setup['stand_rule']}",'',
        '| Direction / model | Knockdown (s) | Force off (s) | Stable stand begins / confirmed (s) | Recovery to confirmed stand (s) | Upright at end | All-joint speed RMS / P95 (rad/s) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for direction,pair in report['trials'].items():
        folder=args.directory/direction
        fig,axes=plt.subplots(6,1,figsize=(12,15),sharex=True)
        detail[direction]={}
        for label,color in [('before','#4277a8'),('after','#c45542')]:
            with np.load(folder/f'{label}_traces.npz') as a:
                trace={key:a[key] for key in a.files}
            result=pair[label]
            t=np.arange(len(trace['root']))/50
            tp=np.arange(len(trace['pd']))/200
            axes[0].plot(t,trace['force'],label=label,color=color)
            axes[1].plot(t,trace['root'][:,2],label=label,color=color)
            axes[2].plot(t,trace['up'],label=label,color=color)
            axes[3].plot(tp[::20]+.05,np.sqrt(np.mean(trace['pd'][:,0].reshape(-1,20,24)**2,axis=(1,2))),label=label,color=color)
            axes[4].plot(t,abs(trace['foot_force']).max(axis=1),label=label,color=color)
            foot_ids=[list(trace['body_names']).index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
            loads=trace['ground_load'].copy()
            loads[:,foot_ids]=0
            axes[5].plot(t,loads.sum(axis=1),label=label,color=color)
            for ax in axes:
                if result['knocked_down_s'] is not None:ax.axvline(result['knocked_down_s'],color=color,linestyle=':',alpha=.5)
                if result['standing_confirmed_s'] is not None:ax.axvline(result['standing_confirmed_s'],color=color,linestyle='--',alpha=.5)
            metrics=result['recovery']
            metrics_text='N/A' if metrics is None else f"{metrics['joint_speed_rms_rad_s']:.2f} / {metrics['joint_speed_p95_rad_s']:.2f}"
            lines.append(f"| {direction} / {label} | {result['knocked_down_s']} | {result['push_off_s']} | {result['standing_started_s']} / {result['standing_confirmed_s']} | {result['recovery_duration_s']} | {result['remained_standing_last_2s']} | {metrics_text} |")
            start=max(result['knocked_down_s'],result['push_off_s']) if result['knocked_down_s'] is not None else None
            detail[direction][label]=dict(fixed_5s_after_down=segment_metrics(trace,start,min(start+5,t[-1])) if start is not None else None,
                interpretation='Successful interval ends after 1s confirmed standing; failed interval extends to rollout end. Unequal intervals are not a like-for-like smoothness comparison.')
        labels=['Horizontal torso force (N)','Root height (m)','Root up-dot','All-joint speed RMS (rad/s, 100ms)','Maximum foot net Fz (N, 50Hz)','Sum nonfoot ground normal load (N, 50Hz)']
        for ax,label in zip(axes,labels):
            ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc='upper right')
        axes[-1].set_xlabel('Time (s); dotted = knockdown; dashed = confirmed standing')
        fig.suptitle(f'{direction} push | baseline vs joint velocity 10x | frozen context 0')
        fig.tight_layout();fig.savefig(folder/'metrics.png',dpi=150);plt.close(fig)
    lines.extend(['','## Interpretation limits','',
        '- Reported recovery interval includes remaining fall/impact after force removal and the 1s standing confirmation period; it is not pure powered get-up.',
        '- A failed rollout has a longer metric interval. Lower RMS in a failed rollout is not an improvement.',
        '- Nonfoot ground load includes hands, knees and torso; initial fall impact must not be mislabeled as a get-up strike.',
        '- Foot force is sampled at 50Hz and may miss narrow peaks. No hardware/audio measurements.',
        '- Same peak/world-direction force does not guarantee identical fallen states. One deterministic trial per direction is not a success-rate estimate.',
        '- Additional training, state differences, and learned context differences prevent attributing all changes solely to the reward weight.',
        '- Left/before = baseline; right/after = 10x joint-velocity model. Training was not stopped or modified for evaluation.',''])
    (args.directory/'details.json').write_text(json.dumps(detail,indent=2)+'\n')
    (args.directory/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
