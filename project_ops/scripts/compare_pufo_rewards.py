"""Paired CPU sim2sim rollouts with videos and motion/contact diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))
import sim2sim_rp1_pufo as sim

np, torch, mujoco = sim.np, sim.torch, sim.mujoco


class MeasuredSimulation(sim.Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feet = [self.model.body(name).id for name in
                     ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.pd_samples = []

    def advance(self, action, rescale):
        # Same four 200 Hz PD updates and eight physics steps as Simulation.
        target = self.prepare_action_target(action, rescale)
        for _ in range(4):
            vel = self.data.qvel[self.vids].copy()
            torque = self.kp * (target - self.data.qpos[self.qids]) - self.kd * vel
            drive = self.limit * np.clip((self.speed - abs(vel)) / (self.speed - self.keep), 0, 1)
            torque = np.clip(torque, np.where(vel <= 0, -drive, -self.limit),
                             np.where(vel >= 0, drive, self.limit))
            self.data.ctrl[self.aids] = torque
            self.pd_samples.append((vel, torque.copy()))
            mujoco.mj_step(self.model, self.data, nstep=2)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite MuJoCo state")

    def feet_state(self):
        velocity = np.zeros((2, 6))
        for row, body in enumerate(self.feet):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                    body, velocity[row], 0)
        forces = np.zeros((2, 3))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if not any(body in self.feet for body in bodies):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            world = contact.frame.reshape(3, 3).T @ wrench[:3]
            for row, body in enumerate(self.feet):
                if bodies[0] == body:
                    forces[row] -= world
                if bodies[1] == body:
                    forces[row] += world
        return velocity[:, 5], forces[:, 2]


def summarize(trace, start, end):
    """Use identical time intervals, including only pre-fall samples when requested."""
    lo, hi = round(start * 50), round(end * 50)
    if hi <= lo:
        return None
    vel, torque = trace['pd'][lo * 4:hi * 4].transpose(1, 0, 2)
    joint_acc = np.diff(trace['pd'][:, 0], axis=0) / .005
    joint_acc = joint_acc[lo * 4:max(lo * 4, hi * 4 - 1)]
    foot_v = trace['foot_v'][lo + 1:hi + 1]
    foot_a = (np.diff(trace['foot_v'], axis=0) / .02)[lo:hi]
    force = abs(trace['foot_force'][lo + 1:hi + 1])
    contact = force > 0
    qerr = trace['joint_pos'][lo + 1:hi + 1] - trace['reference'][lo + 1:hi + 1]
    targets = trace['targets'][lo:hi]
    return dict(
        interval_s=[start, end],
        joint_speed_rms_rad_s=float(np.sqrt(np.mean(vel ** 2))),
        joint_speed_p95_rad_s=float(np.percentile(abs(vel), 95)),
        joint_vel_l2_mean=float(np.mean(np.sum(vel ** 2, axis=-1))),
        joint_acc_rms_rad_s2=float(np.sqrt(np.mean(joint_acc ** 2))),
        joint_acc_l2_mean=float(np.mean(np.sum(joint_acc ** 2, axis=-1))),
        torque_rms_nm=float(np.sqrt(np.mean(torque ** 2))),
        torque_p95_nm=float(np.percentile(abs(torque), 95)),
        torque_max_nm=float(np.max(abs(torque))),
        abs_mechanical_power_mean_w=float(np.mean(np.sum(abs(torque * vel), axis=-1))),
        contact_force_p95_n=float(np.percentile(force[contact], 95)) if contact.any() else None,
        contact_force_max_n=float(force.max()),
        contact_fraction=float(contact.mean()),
        sound_suppression_proxy_mean=float(np.mean(np.sum((foot_v ** 2 + foot_a ** 2) * contact, axis=-1))),
        joint_tracking_rmse_rad=float(np.sqrt(np.mean(qerr ** 2))),
        joint_tracking_l2_mean_rad=float(np.mean(np.linalg.norm(qerr, axis=-1))),
        target_increment_rms_rad=float(np.sqrt(np.mean(np.diff(targets, axis=0) ** 2))) if len(targets) > 1 else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terrain', choices=('flat', 'stairs'), default='flat')
    parser.add_argument('--motion', default='dance1_subject2')
    parser.add_argument('--start-frame', type=int, default=50)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=.8)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.start_frame < 0 or args.window_size < 1 or not np.isfinite(args.gamma) or args.gamma <= 0:
        parser.error('Invalid frame or window settings')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('Duration must be finite and positive')
    torch.set_num_threads(args.threads)
    sim.cv2.setNumThreads(1)
    manifest_path = ROOT / 'robo_newton/data/motions/rp1_lafan/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = next(c for c in manifest['clips'] if c['name'] == args.motion)
    source = manifest_path.parent / record['npz']
    with np.load(source) as archive:
        motion = {k: archive[k].copy() for k in archive.files}
    if motion['fps'].item() != 50:
        raise ValueError('Expected 50 Hz reference')
    motion['body_pos_w'][..., :2] -= motion['body_pos_w'][args.start_frame, 0, :2].copy()
    n = len(motion['joint_pos']) - args.start_frame - 1
    if args.duration is not None:
        n = min(n, round(args.duration * 50))
    if n < 1:
        raise ValueError('Empty reference interval')
    checkpoints = [args.before.resolve(), args.after.resolve()]
    configs = [sim.read_config(p.parent / 'params/env.yaml') for p in checkpoints]
    # Fail rather than compare different actuator/observation contracts silently.
    for key in ('sim', 'decimation', 'actions', 'observations'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Saved evaluation contracts differ: {key}')
    for key in ('robot', 'camera'):
        if configs[0]['scene'][key] != configs[1]['scene'][key]:
            raise ValueError(f'Saved scene contracts differ: {key}')
    names = manifest['npz']['joint_order']
    policies = [sim.Policy(p) for p in checkpoints]
    ref = sim.Simulation(sim.make_model(False), names, configs[0])
    robots = [MeasuredSimulation(sim.make_model(args.terrain == 'stairs'), names, c) for c in configs]
    contexts, headings, velocities = [], [], []
    offsets = []
    for policy, robot in zip(policies, robots):
        robot.set_reference(motion, args.start_frame)
        offsets.append(sim.place_on_stairs(robot) if args.terrain == 'stairs' else 0.)
        z, h, v = policy.contexts(motion, robot.default)
        # Encode full future reference before slicing.  Hold/segment boundaries
        # must not accidentally remove available future frames from the window.
        contexts.append(z[args.start_frame + 1:])
        headings.append(h)
        velocities.append(v)
    ref.set_reference(motion, args.start_frame)
    ids = [ref.model.body(name).id for name in manifest['npz']['body_order']]
    fk_error = float(np.max(np.linalg.norm(ref.data.xpos[ids] - motion['body_pos_w'][args.start_frame], axis=-1)))
    if fk_error > .025:
        raise ValueError(f'Reference/MJCF mismatch: {fk_error}')
    args.output.mkdir(parents=True)
    reward_weights = [{k: v['weight'] for k, v in c['rewards'].items() if isinstance(v, dict)} for c in configs]
    metadata = dict(checkpoints=[str(p) for p in checkpoints], steps=[p.step for p in policies],
                    motion=args.motion, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    start_frame=args.start_frame, duration_s=n / 50, terrain=args.terrain,
                    window_size=args.window_size, gamma=args.gamma, reward_weights=reward_weights,
                    clip_joint_targets=True,
                    initial_fk_error_m=fk_error, reset_height_offsets_m=offsets,
                    context='Each checkpoint encodes B(s[t+1]); future gamma-weighted window, first-row norm',
                    initialization='Same reference pose/velocity; zero proprioceptive history; no noise/resets',
                    scope='Torch CPU policy, MuJoCo sim2sim; not ONNX or a real-robot trial',
                    sampling='Joint velocity and commanded clipped torque at 200 Hz; foot COM velocity/net force at 50 Hz',
                    sound_proxy='sum((foot_vz^2 + finite_difference_50Hz(foot_vz)^2) * (abs(net_Fz)>0)); not exact training reward',
                    limitation='Before/after training comparison, not matched-step ablation; further training is confounded')
    (args.output / 'setup.json').write_text(json.dumps(metadata, indent=2) + '\n')
    traces = [dict(joint_pos=[], root=[], up=[], foot_v=[], foot_force=[], targets=[]) for _ in robots]
    video = args.output / 'comparison.mp4'
    command = [sim.imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '1920x720', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
               '-threads', '2', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(n + 1):
                frame_idx = args.start_frame + step
                ref.set_reference(motion, frame_idx)
                for robot, trace in zip(robots, traces):
                    foot_v, force = robot.feet_state()
                    trace['joint_pos'].append(robot.data.qpos[robot.qids].copy())
                    trace['root'].append(robot.data.qpos[:3].copy())
                    trace['up'].append(robot.data.xmat[robot.base].reshape(3, 3)[2, 2])
                    trace['foot_v'].append(foot_v)
                    trace['foot_force'].append(force)
                if step % 2 == 0:
                    image = np.full((720, 1920, 3), 24, dtype=np.uint8)
                    labels = ['REFERENCE', f'BEFORE | step {policies[0].step:,}', f'AFTER | step {policies[1].step:,}']
                    for col, (robot, label) in enumerate(zip([ref, *robots], labels)):
                        x = col * 640
                        image[50:690, x:x + 640] = robot.render()
                        sim.cv2.putText(image, label, (x + 14, 31), sim.cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 2)
                        text = f'{args.terrain}  t={step/50:.2f}/{n/50:.2f}s  window={args.window_size}'
                        if col:
                            error = np.linalg.norm(traces[col - 1]['joint_pos'][-1] - motion['joint_pos'][frame_idx])
                            text += f'  joint L2={error:.2f}'
                        sim.cv2.putText(image, text, (x + 12, 711), sim.cv2.FONT_HERSHEY_SIMPLEX, .46, (230, 230, 230), 1)
                    process.stdin.write(image.tobytes())
                if step == n:
                    break
                for index, (policy, robot, trace) in enumerate(zip(policies, robots, traces)):
                    obs, values = robot.observe(headings[index][frame_idx], velocities[index][frame_idx])
                    z = sim.weighted_context_window(contexts[index], step, args.gamma, args.window_size)[None]
                    action = policy.act(obs, z).numpy()[0]
                    if not np.isfinite(action).all():
                        raise FloatingPointError('Nonfinite policy action')
                    robot.advance(action, policy.action_rescale)
                    robot.history.add(values)
                    trace['targets'].append(robot.joint_target.copy())
                if step % 250 == 0:
                    print(f'{args.terrain}: {step/50:.1f}/{n/50:.1f}s, elapsed={time.monotonic()-started:.1f}s', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoder failed')
    finally:
        for robot in [ref, *robots]:
            robot.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    falls = []
    for label, robot, trace, offset in zip(('before', 'after'), robots, traces, offsets):
        for key, values in trace.items():
            trace[key] = np.asarray(values)
        trace['pd'] = np.asarray(robot.pd_samples)
        trace['reference'] = motion['joint_pos'][args.start_frame:args.start_frame + n + 1]
        # Conservative instability marker, not a success criterion by itself.
        fell = np.flatnonzero((trace['root'][:, 2] - offset < .35) | (trace['up'] < .5))
        falls.append(float(fell[0] / 50) if len(fell) else None)
        np.savez_compressed(args.output / f'{label}_traces.npz', **trace, joint_names=names)
    common_end = min([n / 50] + [t for t in falls if t is not None])
    results = {}
    for index, (label, trace) in enumerate(zip(('before', 'after'), traces)):
        results[label] = dict(first_instability_s=falls[index],
                              full=summarize(trace, 0, n / 50),
                              common_prefall=summarize(trace, 0, common_end),
                              common_prefall_after_2s=summarize(trace, 2, common_end))
    report = dict(**metadata, common_prefall_end_s=common_end, results=results,
                  elapsed_s=time.monotonic() - started)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    subprocess.run([sim.imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-i', str(video),
                    '-f', 'null', '-'], check=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
