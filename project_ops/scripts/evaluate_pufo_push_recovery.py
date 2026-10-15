"""Paired physical knockdown and unassisted recovery with frozen deployment z."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from test_pufo_stair_push import ROOT, MeasuredSimulation, sim, np, torch, mujoco, hold_observation, segment_metrics


def ground_loads(robot):
    loads = np.zeros(robot.model.nbody)
    for index in range(robot.data.ncon):
        contact = robot.data.contact[index]
        a, b = robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
        if (a == 0) == (b == 0):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(robot.model, robot.data, index, wrench)
        loads[b if a == 0 else a] += max(0., wrench[0])
    return loads


def is_down(height, up, nonfoot_load):
    return bool((height < .45 or up < .5) and nonfoot_load > 20.)


def is_standing(height, up, base_speed, foot_force, nonfoot_load):
    return bool(height > .65 and up > .85 and base_speed < .5
                and (np.asarray(foot_force) > 10.).all() and nonfoot_load < 20.)


def first_stable_stand(mask, first_index, duration=50):
    count = 0
    for i in range(first_index, len(mask)):
        count = count + 1 if mask[i] else 0
        if count >= duration:
            return (i-duration+1)/50, i/50
    return None, None


def pulse(t, magnitude, end=4.2):
    if t >= end:
        return 0.
    return magnitude * max(0., min(1., (t-3.)/.1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force', type=float, default=300.)
    parser.add_argument('--directions', choices=['forward','backward'], nargs='+', default=['forward','backward'])
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--push-end', type=float, default=4.2)
    parser.add_argument('--fixed-pulse', action='store_true')
    parser.add_argument('--target-clip-ab', action='store_true',
                        help='Same checkpoint, raw targets on left and clipped targets on right')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.target_clip_ab and (args.before.resolve() != args.after.resolve() or not args.fixed_pulse):
        parser.error('Target clipping A/B requires the same checkpoint and --fixed-pulse')
    if not np.isfinite(args.force) or not 0 < args.force <= 1000 or not np.isfinite(args.duration) or args.duration < 8:
        parser.error('Invalid force/duration')
    if not np.isfinite(args.push_end) or not 3.1 < args.push_end <= 5.:
        parser.error('Push end must be between 3.1 and 5 seconds')
    torch.set_num_threads(2)
    sim.cv2.setNumThreads(1)
    manifest_path = ROOT/'robo_newton/data/motions/rp1_lafan/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = manifest_path.parent / next(c for c in manifest['clips'] if c['name']=='dance1_subject2')['npz']
    with np.load(source) as archive:
        motion = {k:archive[k].copy() for k in archive.files}
    names = manifest['npz']['joint_order']
    checkpoints = [args.before,args.after]
    clip_flags = [False, True] if args.target_clip_ab else [True, True]
    labels = ['NO TARGET CLIP', 'TARGET CLIPPED'] if args.target_clip_ab else ['BASELINE', 'JVEL 10X']
    configs = [sim.read_config(p.parent/'params/env.yaml') for p in checkpoints]
    for key in ['sim','decimation','actions','observations']:
        assert configs[0][key] == configs[1][key], key
    for key in ['robot','camera']:
        assert configs[0]['scene'][key] == configs[1]['scene'][key], key
    policies = [sim.Policy(p) for p in checkpoints]
    args.output.mkdir(parents=True)
    setup = dict(checkpoints=[str(p) for p in checkpoints],steps=[p.step for p in policies],
        joint_vel_weights=[c['rewards']['joint_vel_l2']['weight'] for c in configs],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),motion='dance1_subject2',
        context_row=0,source_frame=50,window=5,gamma=.8,heading=[0,0],velocity=[0,0,0],
        force_n=args.force,push_end_s=args.push_end,fixed_pulse=args.fixed_pulse,
        target_clip_ab=args.target_clip_ab,clip_joint_targets=clip_flags,labels=labels,
        target_clip_rule='Clip final PD position target to model joint ranges; action history and physical state unchanged',
        force_protocol=f'World +/-X at waist COM, start 3s, 0.1s ramp, off at {args.push_end}s' + (' regardless of state (identical pulse)' if args.fixed_pulse else ' or first confirmed knockdown'),
        down_rule='(root height<0.45m OR up-dot<0.5) AND nonfoot ground normal load>20N',
        stand_rule='For 1s: root height>0.65m, up-dot>0.85, base linear speed<0.5m/s, both feet Fz>10N, other ground load<20N',
        initialization='Source frame50 stance, yaw +X, zero velocity; no subsequent pose/velocity assignment or reset',
        scope='MuJoCo CPU Torch policy; no hardware action filtering/latency; not a hardware safety test',
        sampling='Joint velocity/clipped torque 200Hz; foot forces and per-body ground normal loads 50Hz',
        limitations=('One deterministic rollout/direction; same checkpoint and force, potentially different fall states'
                     if args.target_clip_ab else 'One deterministic rollout/direction/model; additional training confound'))
    (args.output/'setup.json').write_text(json.dumps(setup,indent=2)+'\n')
    contexts = None
    reports = {}
    began = time.monotonic()
    n = round(args.duration*50)
    for direction in args.directions:
        folder = args.output/direction
        folder.mkdir()
        robots = [MeasuredSimulation(sim.make_model(False),names,c,clip_joint_targets=clip)
                  for c,clip in zip(configs,clip_flags)]
        if contexts is None:
            contexts = []
            for robot,policy in zip(robots,policies):
                z,_,_ = policy.contexts(motion,robot.default)
                contexts.append(sim.weighted_context_window(z[51:],0,.8,5)[None])
        for robot in robots:
            robot.set_reference(motion,50)
            q=sim.Rotation.from_quat(robot.data.qpos[[4,5,6,3]])
            yaw=np.arctan2(q.as_matrix()[1,0],q.as_matrix()[0,0])
            robot.data.qpos[3:7]=(sim.Rotation.from_euler('z',-yaw)*q).as_quat()[[3,0,1,2]]
            robot.data.qpos[:2]=0.
            robot.data.qvel[:]=0.
            mujoco.mj_forward(robot.model,robot.data)
            robot.camera_distance,robot.camera_azimuth,robot.camera_elevation=3.6,110.,-20.
        traces=[{k:[] for k in ['joint_pos','root','up','foot_v','foot_force','targets','raw_targets','ground_load','force','base_speed','standing']} for _ in robots]
        down=[None,None]
        stop=[None,None]
        body_ids=[r.model.body('waist_yaw_link').id for r in robots]
        sign=1 if direction=='forward' else -1
        video=folder/'comparison.mp4'
        writer=subprocess.Popen([sim.imageio_ffmpeg.get_ffmpeg_exe(),'-n','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x740','-r','25','-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        try:
            with torch.inference_mode():
                for step in range(n+1):
                    t=step/50
                    for i,(robot,trace) in enumerate(zip(robots,traces)):
                        v,f=robot.feet_state()
                        loads=ground_loads(robot)
                        nonfoot=float(loads.sum()-loads[robot.feet].sum())
                        up=float(robot.data.xmat[robot.base].reshape(3,3)[2,2])
                        height=float(robot.data.qpos[2])
                        speed=float(np.linalg.norm(robot.data.qvel[:3]))
                        if down[i] is None and is_down(height,up,nonfoot):
                            down[i]=t
                        if stop[i] is None and ((down[i] is not None and not args.fixed_pulse) or t>=args.push_end):
                            stop[i]=t
                        force=sign*pulse(t,args.force,args.push_end) if stop[i] is None else 0.
                        for key,val in dict(joint_pos=robot.data.qpos[robot.qids].copy(),root=robot.data.qpos[:3].copy(),
                            up=up,foot_v=v,foot_force=f,ground_load=loads,force=force,base_speed=speed,
                            standing=is_standing(height,up,speed,f,nonfoot)).items():
                            trace[key].append(val)
                    if step%2==0:
                        frame=np.full((740,1280,3),24,dtype=np.uint8)
                        for i,robot in enumerate(robots):
                            x=i*640
                            frame[50:690,x:x+640]=robot.render()
                            title=f'{labels[i]} | {direction} {args.force:g}N | HOLD 0'
                            sim.cv2.putText(frame,title,(x+12,32),sim.cv2.FONT_HERSHEY_SIMPLEX,.57,(240,240,240),2)
                            line=f't={t:.2f}s F={traces[i]["force"][-1]:.0f}N height={traces[i]["root"][-1][2]:.2f}m'
                            sim.cv2.putText(frame,line,(x+12,713),sim.cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1)
                            if down[i] is not None:
                                status='STANDING' if traces[i]['standing'][-1] else 'RECOVERING'
                                sim.cv2.putText(frame,f'DOWN at {down[i]:.2f}s | {status}',(x+12,735),sim.cv2.FONT_HERSHEY_SIMPLEX,.5,(255,190,100),1)
                        writer.stdin.write(frame.tobytes())
                    if step==n:
                        break
                    for i,(robot,policy,trace,z) in enumerate(zip(robots,policies,traces,contexts)):
                        robot.data.xfrc_applied[:]=0.
                        robot.data.xfrc_applied[body_ids[i],0]=trace['force'][-1]
                        obs,values=hold_observation(robot)
                        assert not obs['heading'].any() and not obs['velocity_command'].any()
                        action=policy.act(obs,z).numpy()[0]
                        if not np.isfinite(action).all():
                            raise FloatingPointError('Nonfinite action')
                        robot.advance(action,policy.action_rescale)
                        robot.history.add(values)
                        trace['targets'].append(robot.joint_target.copy())
                        trace['raw_targets'].append(robot.raw_joint_target.copy())
                    if step%250==0:
                        print(f'{direction} t={t:.1f}s down={down} elapsed={time.monotonic()-began:.1f}s',flush=True)
            writer.stdin.close()
            if writer.wait()!=0:
                raise RuntimeError('Video encoder failed')
        finally:
            for robot in robots:robot.close()
            if writer.poll() is None:
                writer.kill();writer.wait()
        pair={}
        for i,(label,robot,trace) in enumerate(zip(['before','after'],robots,traces)):
            trace={k:np.asarray(v) for k,v in trace.items()}
            trace['pd']=np.asarray(robot.pd_samples)
            trace['reference']=np.broadcast_to(motion['joint_pos'][50],trace['joint_pos'].shape)
            trace['context']=contexts[i].numpy()
            body_names=[robot.model.body(j).name for j in range(robot.model.nbody)]
            np.savez_compressed(folder/f'{label}_traces.npz',**trace,joint_names=names,body_names=body_names,
                                joint_limits=robot.joint_limits)
            first,confirmed=(None,None) if down[i] is None else first_stable_stand(trace['standing'],round(max(down[i],stop[i])*50)+1)
            start=max(down[i],stop[i]) if down[i] is not None else None
            end=confirmed if confirmed is not None else n/50
            metrics=segment_metrics(trace,start,end) if start is not None else None
            if metrics is not None:
                lo,hi=round(start*50),round(end*50)+1
                loads=trace['ground_load'][lo:hi].copy()
                loads[:,robot.feet]=0
                metrics['nonfoot_ground_load_peak_n']=float(loads.sum(axis=1).max())
                metrics['joint_speed_max_rad_s']=float(abs(trace['pd'][round(start*200):round(end*200),0]).max())
            lower,upper=robot.joint_limits.T
            target_excess=np.maximum(np.maximum(lower-trace['targets'],trace['targets']-upper),0.)
            actual_excess=np.maximum(np.maximum(lower-trace['joint_pos'],trace['joint_pos']-upper),0.)
            raw_excess=np.maximum(np.maximum(lower-trace['raw_targets'],trace['raw_targets']-upper),0.)
            if clip_flags[i] and np.any(target_excess>1e-12):
                raise AssertionError('Clipped PD target outside joint limits')
            pair[label]=dict(label=labels[i],clip_joint_targets=clip_flags[i],
                target_limit_violation_max_rad=float(target_excess.max()),
                raw_target_limit_violation_max_rad=float(raw_excess.max()),
                raw_target_outside_fraction=float((raw_excess>1e-12).mean()),
                actual_limit_violation_max_rad_50hz=float(actual_excess.max()),
                knocked_down_s=down[i],push_off_s=stop[i],standing_started_s=first,
                standing_confirmed_s=confirmed,recovery_duration_s=None if confirmed is None else confirmed-start,
                remained_standing_last_2s=bool(trace['standing'][-100:].all()),
                impulse_ns=float(abs(trace['force'][:-1]).sum()*.02),recovery=metrics,
                final_root_xyz=trace['root'][-1].tolist())
        reports[direction]=pair
        (folder/'report.json').write_text(json.dumps(pair,indent=2)+'\n')
        (args.output/'report.json').write_text(json.dumps(dict(setup=setup,trials=reports),indent=2)+'\n')
        print(json.dumps(pair),flush=True)


if __name__=='__main__':
    main()
