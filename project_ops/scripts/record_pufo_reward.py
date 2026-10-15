"""Record fixed-context FB reward inference for forward locomotion."""


from pathlib import Path
import sys

_MUJOCO_SCRIPTS = Path(__file__).resolve().parents[2] / "robo_newton" / "scripts" / "mujoco"
if str(_MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MUJOCO_SCRIPTS))

import argparse
import json
from pathlib import Path
import subprocess

import sim2sim_rp1_pufo as sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=.7)
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--terrain', choices=('flat', 'low_stairs'), default='flat')
    parser.add_argument('--stair-height', type=float, help='Override stair rise in metres')
    args = parser.parse_args()
    if args.output.exists() or args.duration <= 0:
        raise ValueError('Output must be new and duration positive')
    if args.stair_height is not None and (args.stair_height <= 0 or args.terrain != 'low_stairs'):
        parser.error('--stair-height requires low_stairs and a positive height')
    sim.torch.set_num_threads(2)
    sim.cv2.setNumThreads(1)
    cfg = sim.read_config(args.checkpoint.parent / 'params/env.yaml')
    metadata = json.loads(args.manifest.read_text())
    stairs = args.terrain == 'low_stairs'
    model = sim.make_model(stairs)
    stair_height = json.loads(sim.STAIRS_ASSET.read_text())['step_height'] if stairs else None
    if args.stair_height is not None:
        scale = args.stair_height / stair_height
        for index in range(model.ngeom):
            name = model.geom(index).name
            if name.startswith(('step_', 'landing_')):
                model.geom_pos[index, 2] *= scale
                model.geom_size[index, 2] *= scale
        stair_height = args.stair_height
    env = sim.Simulation(model, metadata['npz']['joint_order'], cfg)
    policy = sim.Policy(args.checkpoint)
    total = sim.torch.zeros(1, policy.cfg['z_dim'])
    reward_sum, count = 0., 0
    selected = set(cfg['commands']['motion']['motion_names'])
    # Uniform temporal subsampling across the training motions, with reward-weighted B(s).
    for record in metadata['clips']:
        if record['name'] not in selected:
            continue
        with sim.np.load(args.manifest.parent / record['npz']) as archive:
            motion = {k: archive[k][::10].copy() if k != 'fps' else archive[k] for k in archive.files}
        contexts, headings, velocities = policy.contexts(motion, env.default)
        forward = (velocities * headings).sum(-1)
        lateral = -velocities[:, 0] * headings[:, 1] + velocities[:, 1] * headings[:, 0]
        reward = sim.torch.exp(-((forward - args.speed)**2 + lateral**2) / .25)
        total += (contexts * reward[:, None]).sum(0, keepdim=True)
        reward_sum += reward.sum().item()
        count += len(reward)
    if count == 0 or reward_sum <= 0:
        raise ValueError('No training motion samples available for inference')
    context = total / count
    if policy.cfg['norm_z']:
        context = context.shape[-1]**.5 * sim.F.normalize(context, dim=-1)
    env.data.qpos[:3] = [0., 0., .85]
    env.data.qpos[3:7] = [1., 0., 0., 0.]
    env.data.qpos[env.qids] = env.default
    env.data.qvel[:] = 0
    sim.mujoco.mj_forward(env.model, env.data)
    reset_height = sim.place_on_stairs(env) if stairs else 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sim.imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-loglevel', 'error',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '640x720', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
               '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    samples = []
    try:
        for step in range(round(args.duration * 50)):
            obs, values = env.observe(sim.torch.tensor([1., 0.]), sim.torch.tensor([args.speed, 0.]))
            samples.append([step / 50, *env.data.qpos[:3], *env.data.qvel[:3]])
            if step % 2 == 0:
                frame = sim.np.full((720, 640, 3), 24, dtype=sim.np.uint8)
                frame[40:680] = env.render()
                label = 'LOW STAIRS' if stairs else 'FLAT'
                sim.cv2.putText(frame, f'REWARD INFERENCE | {label} | model {policy.step}',
                                (12, 26), sim.cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
                sim.cv2.putText(frame, f't={step/50:.2f}s  target +X={args.speed:.2f} m/s  actual={env.data.qvel[0]:.2f}',
                                (12, 707), sim.cv2.FONT_HERSHEY_SIMPLEX, .46, (240, 240, 240), 1)
                process.stdin.write(frame.tobytes())
            action = policy.act(obs, context).numpy()[0]
            env.advance(action, policy.action_rescale)
            env.history.add(values)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Video encoding failed')
    finally:
        env.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    subprocess.run([sim.imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror',
                    '-i', str(args.output), '-f', 'null', '-'], check=True)
    trajectory = sim.np.asarray(samples)
    report = dict(checkpoint=str(args.checkpoint), model_step=policy.step,
                  manifest=str(args.manifest), mode='reward_inference', duration_s=args.duration,
                  target_world_velocity=[args.speed, 0.], inference_samples=count,
                  reward=f'exp(-((v_forward-{args.speed})^2+v_lateral^2)/0.25)',
                  terrain=args.terrain, reset_height_m=reset_height,
                  terrain_asset=str(sim.STAIRS_ASSET) if stairs else None,
                  stair_height_m=stair_height,
                  context_method='project_z(mean(B(s)*reward(s))); fixed throughout rollout',
                  mean_world_vx=float(trajectory[:, 4].mean()),
                  mean_world_vx_after_2s=float(trajectory[100:, 4].mean()),
                  min_root_height=float(trajectory[:, 3].min()),
                  final_root_position=env.data.qpos[:3].tolist(),
                  automatic_resets=False, observation_noise=False, actuator_delay_ticks=0)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    sim.np.savez_compressed(args.output.with_suffix('.npz'), trajectory=trajectory, context=context.numpy())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
