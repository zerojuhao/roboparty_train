"""Summarize paired PUFO reward evaluation traces and plot their time series."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_pufo_rewards import ROOT, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / 'report.json').read_text())
    manifest_path = ROOT / 'robo_newton/data/motions/rp1_lafan/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = next(c for c in manifest['clips'] if c['name'] == report['motion'])
    source = manifest_path.parent / record['npz']
    if hashlib.sha256(source.read_bytes()).hexdigest() != report['source_sha256']:
        raise ValueError('Reference motion differs from the evaluated data')
    with np.load(source) as archive:
        reference_xy = archive['body_pos_w'][report['start_frame']:, 0, :2].copy()
    reference_xy -= reference_xy[0].copy()
    traces = {}
    for label in ('before', 'after'):
        with np.load(args.directory / f'{label}_traces.npz') as archive:
            traces[label] = {key: archive[key] for key in archive.files}
    duration = report['duration_s']
    end = report['common_prefall_end_s']
    windows = {'common_prefall_after_2s': (2., end), 'first_30s_prefall': (2., min(30., end))}
    result = {}
    for name, (start, stop) in windows.items():
        pair = {label: summarize(trace, start, stop) for label, trace in traces.items()}
        if any(value is None for value in pair.values()):
            result[name] = pair
            continue
        for label, trace in traces.items():
            names = trace['joint_names'].tolist()
            legs = np.array([any(part in joint for part in ('_hip_', '_knee_', '_ankle_')) for joint in names])
            samples = trace['pd'][round(start * 200):round(stop * 200)]
            acceleration = np.diff(samples[:, 0], axis=0) / .005
            xy_error = trace['root'][:, :2] - trace['root'][0, :2] - reference_xy[:len(trace['root'])]
            xy_error = xy_error[round(start * 50) + 1:round(stop * 50) + 1]
            pair[label]['root_xy_tracking_rmse_m'] = float(np.sqrt(np.mean(np.sum(xy_error ** 2, axis=-1))))
            for part, mask in (('legs', legs), ('upper', ~legs)):
                pair[label][f'{part}_speed_rms_rad_s'] = float(np.sqrt(np.mean(samples[:, 0, mask] ** 2)))
                pair[label][f'{part}_torque_rms_nm'] = float(np.sqrt(np.mean(samples[:, 1, mask] ** 2)))
                pair[label][f'{part}_acc_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration[:, mask] ** 2)))
            for joint in ('left_ankle_pitch_joint', 'right_ankle_pitch_joint'):
                pair[label][f'{joint}_acc_rms_rad_s2'] = float(np.sqrt(np.mean(acceleration[:, names.index(joint)] ** 2)))
        pair['change_percent'] = {
            key: 100 * (pair['after'][key] / value - 1)
            for key, value in pair['before'].items()
            if isinstance(value, (int, float)) and value != 0 and pair['after'][key] is not None
        }
        result[name] = pair
    (args.directory / 'metrics_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    for label, color in (('before', '#bf5140'), ('after', '#16847a')):
        trace = traces[label]
        total = len(trace['targets'])
        speed = np.sqrt(np.mean(trace['pd'][:, 0].reshape(total, 4, -1) ** 2, axis=(1, 2)))
        torque = np.sqrt(np.mean(trace['pd'][:, 1].reshape(total, 4, -1) ** 2, axis=(1, 2)))
        foot_acc = np.diff(trace['foot_v'], axis=0) / .02
        sound = np.sum((trace['foot_v'][1:] ** 2 + foot_acc ** 2) * (abs(trace['foot_force'][1:]) > 0), axis=-1)
        error = np.sqrt(np.mean((trace['joint_pos'][1:] - trace['reference'][1:]) ** 2, axis=-1))
        force = abs(trace['foot_force'][1:]).max(axis=-1)
        for axis, values in zip(axes, (speed, torque, sound, force, error)):
            # Nonoverlapping 0.5 s averages; no smoothing across missing/fall boundaries.
            count = len(values) // 25
            block = values[:count * 25].reshape(count, 25).mean(axis=1)
            axis.plot((np.arange(count) + .5) * .5, block, label=label.upper(), color=color, lw=1.2)
            axis.grid(alpha=.2)
        fall = report['results'][label]['first_instability_s']
        if fall is not None:
            for axis in axes:
                axis.axvline(fall, color=color, linestyle='--', alpha=.7)
    for axis, title in zip(axes, ('Joint speed RMS (rad/s)', 'Applied torque RMS (Nm)',
                                  'Contact foot vz/az proxy (50 Hz)', 'Max foot |Fz| (N)',
                                  'Joint tracking RMSE (rad)')):
        axis.set_ylabel(title)
    axes[2].set_yscale('symlog', linthresh=1)
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Playback time (s)')
    axes[-1].set_xlim(0, duration)
    fig.suptitle(f"dance1_subject2 | {report['terrain']} | before {report['steps'][0]:,} vs after {report['steps'][1]:,}\n"
                 'Same physics/initial state; window=5, gamma=0.8; dashed lines mark first instability')
    fig.tight_layout()
    fig.savefig(args.directory / 'metrics.png', dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
