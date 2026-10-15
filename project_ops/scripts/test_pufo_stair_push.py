"""Matched held-context stair push evaluation; never edits training state."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from compare_pufo_rewards import ROOT, MeasuredSimulation, sim, np, torch, mujoco, summarize


def stair_model(height=.15, tread=.30, landing_length=1.2):
    xml = sim.ET.parse(sim.MJCF).getroot()
    xml.remove(xml.find('include'))
    robot = sim.ET.parse(sim.MJCF.parent / 'robot_24dof.xml').getroot()
    robot.find('compiler').set('meshdir', str(sim.MJCF.parent.parent))
    xml.find('visual/global').set('offwidth', '640')
    xml.find('visual/global').set('offheight', '640')
    world = xml.find('worldbody')
    world.find('geom').set('friction', '0.9 0.02 0.002')
    # Adjacent solid boxes, first vertical riser at x=0, with a top landing.
    for i in range(6):
        width = tread if i < 5 else landing_length
        z = (i + 1) * height
        sim.ET.SubElement(world, 'geom', name=f'push_stair_{i + 1}', type='box',
                          pos=f'{i*tread + width/2} 0 {z/2}',
                          size=f'{width/2} 1 {z/2}', friction='0.9 0.02 0.002',
                          rgba='0.55 0.60 0.65 1', group='0')
    xml.extend(list(robot))
    model = mujoco.MjModel.from_xml_string(sim.ET.tostring(xml, encoding='unicode'))
    model.opt.timestep = .0025
    model.opt.iterations, model.opt.ls_iterations, model.opt.tolerance = 600, 80, .01
    return model


def push_force(t, peak, start=3., end=12., ramp=.75):
    return peak * max(0., min(1., (t-start)/ramp, (end-t)/ramp))


def hold_observation(robot):
    invalid = torch.zeros(1, 1, dtype=torch.bool)
    return robot.observe(torch.tensor([1., 0.]), torch.zeros(2), invalid, invalid)


def supported_treads(robot):
    """Highest stair touched by each foot with positive upward support force."""
    levels = np.zeros(2, dtype=int)
    for index in range(robot.data.ncon):
        contact = robot.data.contact[index]
        bodies = robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(robot.model, robot.data, index, wrench)
        world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
        for row, body in enumerate(robot.feet):
            for foot_side in (0, 1):
                if bodies[foot_side] != body:
                    continue
                other = [contact.geom1, contact.geom2][1-foot_side]
                name = robot.model.geom(other).name
                sign = -1 if foot_side == 0 else 1
                if name.startswith('push_stair_') and sign * world_force[2] > 10:
                    level = int(name.rsplit('_', 1)[1])
                    # A toe pressed into the vertical wall is not a tread landing.
                    top = robot.model.geom_pos[other, 2] + robot.model.geom_size[other, 2]
                    if abs(contact.pos[2] - top) < .025:
                        levels[row] = max(levels[row], level)
    return levels


def segment_metrics(trace, start, end):
    result = summarize(trace, start, end)
    if result is None:
        return None
    lo, hi = round(start*200), round(end*200)
    speed = abs(trace['pd'][lo:hi, 0, :12])
    result.update(leg_speed_rms_rad_s=float(np.sqrt(np.mean(speed**2))),
                  leg_speed_p95_rad_s=float(np.percentile(speed, 95)),
                  leg_speed_max_rad_s=float(speed.max()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forces', type=float, nargs='+', default=[0., 40., 80., 120.])
    parser.add_argument('--height', type=float, default=.15)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--start-frame', type=int, default=50)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--stop-on-landing', action='store_true')
    parser.add_argument('--landing-length', type=float, default=1.2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.height <= .3 or args.window_size < 1 or args.start_frame < 0:
        parser.error('Invalid geometry or context selection')
    if not np.isfinite(args.landing_length) or args.landing_length < 1.2:
        parser.error('Top landing must be at least 1.2 m long')
    if any(not np.isfinite(f) or f < 0 or f > 300 for f in args.forces):
        parser.error('Forces must be finite and between 0 and 300 N')
    if len(set(args.forces)) != len(args.forces):
        parser.error('Duplicate forces')
    torch.set_num_threads(2)
    sim.cv2.setNumThreads(1)
    manifest_path = ROOT / 'robo_newton/data/motions/rp1_lafan/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = manifest_path.parent / next(c for c in manifest['clips'] if c['name'] == 'dance1_subject2')['npz']
    with np.load(source) as archive:
        motion = {k: archive[k].copy() for k in archive.files}
    names = manifest['npz']['joint_order']
    configs = [sim.read_config(p.parent / 'params/env.yaml') for p in (args.before, args.after)]
    for key in ('sim', 'decimation', 'actions', 'observations'):
        assert configs[0][key] == configs[1][key], key
    for key in ('robot', 'camera'):
        assert configs[0]['scene'][key] == configs[1]['scene'][key], key
    policies = [sim.Policy(p) for p in (args.before, args.after)]
    args.output.mkdir(parents=True)
    setup = dict(checkpoints=[str(args.before), str(args.after)], steps=[p.step for p in policies],
                 motion='dance1_subject2', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 context_row=0, source_start_frame=args.start_frame, window_size=args.window_size, gamma=.8,
                 clip_joint_targets=True,
                 hold='Constant future-window B(s[start+1:start+1+window]); heading=[0,0], velocity=[0,0,0]',
                 terrain=dict(height_m=args.height, tread_m=.3, width_m=2., steps=6, first_riser_x=0., landing_length_m=args.landing_length),
                 forces_n=args.forces, force_schedule='World +X torso force only, ramp 3-3.75s, hold until 11.25s, ramp off at 12s',
                 initialization='Reference source pose, yaw aligned +X, x=-0.45 m, zero initial velocity; no later state resets',
                 sampling='PD speed/commanded torque 200 Hz; net foot Fz/COM velocity/tread support 50 Hz',
                 scope='CPU Torch/MuJoCo held-context test, not full real-controller reproduction or hardware test',
                 limitations=['No hardware action smoothing/motor/sensor latency reproduction',
                              'Foot-force peaks sampled at 50 Hz may miss short impacts; no acoustic measurement',
                              'Before/after also includes additional training, not a causal reward ablation'])
    setup['stop_on_landing'] = args.stop_on_landing
    setup['landing_stop_rule'] = 'Remove force permanently once root X >= 2.1 m; verify tread support independently'
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2)+'\n')
    contexts = None
    reports = {}
    began = time.monotonic()
    for peak in args.forces:
        folder = args.output / f'force_{peak:g}N'
        folder.mkdir()
        robots = [MeasuredSimulation(stair_model(args.height, landing_length=args.landing_length), names, c) for c in configs]
        traces = [dict(joint_pos=[], root=[], up=[], foot_v=[], foot_force=[], targets=[], treads=[], force=[]) for _ in robots]
        if contexts is None:
            contexts = []
            for policy, robot in zip(policies, robots):
                z, _, _ = policy.contexts(motion, robot.default)
                contexts.append(sim.weighted_context_window(z[args.start_frame+1:], 0, .8, args.window_size)[None])
        for robot in robots:
            robot.set_reference(motion, args.start_frame)
            q = sim.Rotation.from_quat(robot.data.qpos[[4, 5, 6, 3]])
            yaw = np.arctan2(q.as_matrix()[1, 0], q.as_matrix()[0, 0])
            q = sim.Rotation.from_euler('z', -yaw) * q
            robot.data.qpos[3:7] = q.as_quat()[[3, 0, 1, 2]]
            robot.data.qpos[:2] = [-.45, 0.]
            robot.data.qvel[:] = 0.
            mujoco.mj_forward(robot.model, robot.data)
            robot.camera_distance, robot.camera_azimuth, robot.camera_elevation = 3.8, 135., -20.
        mass = float(robots[0].model.body_subtreemass[robots[0].base])
        force_body = [r.model.body('waist_yaw_link').id for r in robots]
        writer = None
        video = folder / 'comparison.mp4'
        if not args.no_video:
            writer = subprocess.Popen([sim.imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-v', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', '1280x740', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
                '-threads', '2', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
        falls = [None, None]
        landing_times = [None, None]
        try:
            with torch.inference_mode():
                for step in range(801):
                    t = step / 50
                    force = push_force(t, peak)
                    for i, (robot, trace) in enumerate(zip(robots, traces)):
                        v, f = robot.feet_state()
                        up = robot.data.xmat[robot.base].reshape(3, 3)[2, 2]
                        if falls[i] is None and (up < .5 or robot.data.qpos[2] < .35):
                            falls[i] = t
                        if landing_times[i] is None and robot.data.qpos[0] >= 2.1:
                            landing_times[i] = t
                        for key, value in dict(joint_pos=robot.data.qpos[robot.qids].copy(), root=robot.data.qpos[:3].copy(),
                                               up=up, foot_v=v, foot_force=f, treads=supported_treads(robot),
                                               force=force if falls[i] is None and not (args.stop_on_landing and landing_times[i] is not None) else 0.).items():
                            trace[key].append(value)
                    if writer is not None and step % 2 == 0:
                        frame = np.full((740, 1280, 3), 24, dtype=np.uint8)
                        for i, robot in enumerate(robots):
                            x = i*640
                            frame[50:690, x:x+640] = robot.render()
                            text = f'{["BEFORE", "AFTER"][i]} | stairs {args.height*100:g}cm | context 0 HOLD'
                            sim.cv2.putText(frame, text, (x+12, 32), sim.cv2.FONT_HERSHEY_SIMPLEX, .6, (240,240,240), 2)
                            line = f't={t:.2f}s  push +X={traces[i]["force"][-1]:.0f}N  treads={traces[i]["treads"][-1].tolist()}'
                            sim.cv2.putText(frame, line, (x+12, 713), sim.cv2.FONT_HERSHEY_SIMPLEX, .5, (240,240,240), 1)
                            if falls[i] is not None:
                                sim.cv2.putText(frame, 'INSTABILITY - PUSH REMOVED', (x+12,735), sim.cv2.FONT_HERSHEY_SIMPLEX, .5,(255,100,100),1)
                        writer.stdin.write(frame.tobytes())
                    if step == 800:
                        break
                    for i, (robot, policy, trace, z) in enumerate(zip(robots, policies, traces, contexts)):
                        robot.data.xfrc_applied[:] = 0.
                        robot.data.xfrc_applied[force_body[i], 0] = trace['force'][-1]
                        obs, values = hold_observation(robot)
                        assert not obs['heading'].any() and not obs['velocity_command'].any()
                        action = policy.act(obs, z).numpy()[0]
                        if not np.isfinite(action).all():
                            raise FloatingPointError('Nonfinite action')
                        robot.advance(action, policy.action_rescale)
                        robot.history.add(values)
                        trace['targets'].append(robot.joint_target.copy())
                    if step % 250 == 0:
                        print(f'{peak:g}N t={t:.1f}s elapsed={time.monotonic()-began:.1f}s roots={[r.data.qpos[:3].round(3).tolist() for r in robots]}', flush=True)
            if writer is not None:
                writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError('Video encoder failed')
        finally:
            for robot in robots:
                robot.close()
            if writer is not None and writer.poll() is None:
                writer.kill()
                writer.wait()
        common_end = min([16.] + [f for f in falls if f is not None])
        result = dict(force_n=peak, robot_mass_kg=mass, bodyweight_fraction=peak/(mass*9.81), common_prefall_end_s=common_end)
        for i, (label, robot, trace) in enumerate(zip(('before','after'), robots, traces)):
            trace = {key: np.asarray(value) for key, value in trace.items()}
            trace['pd'] = np.asarray(robot.pd_samples)
            trace['reference'] = np.broadcast_to(motion['joint_pos'][args.start_frame], trace['joint_pos'].shape)
            trace['context'] = contexts[i].numpy()
            np.savez_compressed(folder / f'{label}_traces.npz', **trace, joint_names=names)
            own_end = min(16., falls[i] if falls[i] is not None else 16.)
            valid = trace['treads'][:round(own_end*50)+1]
            upright = trace['up'][:len(valid)] > .7
            max_per_foot = np.where(upright[:, None], valid, 0).max(axis=0)
            both_on_step = (valid > 0).all(axis=1) & upright
            result[label] = dict(first_instability_s=falls[i], max_supported_tread_per_foot=max_per_foot.tolist(),
                reached_landing_x_s=landing_times[i], force_impulse_ns=float(trace['force'][:-1].sum()*.02),
                both_feet_supported_above_floor_s=float(both_on_step.sum()/50),
                final_root_xyz=trace['root'][-1].tolist(),
                settled=segment_metrics(trace, 1., min(3.,common_end)),
                push_common_prefall=segment_metrics(trace, 3., min(12.,common_end)),
                release_common_prefall=segment_metrics(trace, 12., common_end))
        (folder / 'report.json').write_text(json.dumps(result,indent=2)+'\n')
        reports[f'{peak:g}N'] = result
        (args.output / 'report.json').write_text(json.dumps(dict(setup=setup,trials=reports),indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
