"""Read-only policy rollouts on the native training backend, with physical-limit probes."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['training', 'push'], default='training')
parser.add_argument('--num-envs', type=int, default=128)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--context-trace', type=Path)
parser.add_argument('--force', type=float, default=300.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from scipy.spatial.transform import Rotation

import robolab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_newton.physics.newton_manager import NewtonManager
from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg, RP1PUFOEvalEnvCfg
from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
from robolab.tasks.manager_based.ufo.mdp.rewards import BodyImpactJointPositionAction
from robolab.utils.tensors import as_torch_tensor as tensor
from robolab.utils.physics_memory import check_solver_overflow
from rsl_rl.runners.fbcpr_runner import FbCprRunner


class LimitProbe:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene['robot']
        self.solver = NewtonManager._solver
        self.model = self.solver.mj_model
        self.names = list(self.robot.joint_names)
        self.limits = tensor(self.robot.data.joint_pos_limits).clone()
        shape = (env.num_envs, len(self.names))
        self.maximum = torch.zeros(shape, device=env.device)
        self.target_max = torch.zeros_like(self.maximum)
        self.count = torch.zeros_like(self.maximum, dtype=torch.long)
        self.target_count = torch.zeros_like(self.count)
        self.knees = [self.names.index(s + '_knee_joint') for s in ['left', 'right']]
        self.stall_run = torch.zeros((env.num_envs, 2), device=env.device)
        self.stall_max = torch.zeros_like(self.stall_run)
        self.contact_peak = torch.zeros_like(self.stall_run)
        self.contact_run = torch.zeros_like(self.stall_run)
        self.contact_max = torch.zeros_like(self.stall_run)
        body_names = [self.model.body(int(b)).name for b in self.model.geom_bodyid]
        self.knee_geom = torch.tensor([[name.endswith(side + '_knee_link') for side in ['left', 'right']]
                                       for name in body_names], device=env.device)
        self.hip_geom = torch.tensor([[any(name.endswith(side + suffix) for suffix in
                                           ['_hip_roll_link', '_hip_pitch_link']) for side in ['left', 'right']]
                                      for name in body_names], device=env.device)
        assert self.knee_geom.any(0).all() and self.hip_geom.any(0).all(), body_names
        self.contact_bodies = sorted(set(name for name in body_names if any(
            name.endswith(side + suffix) for side in ['left', 'right'] for suffix in
            ['_knee_link', '_hip_roll_link', '_hip_pitch_link'])))
        self.terrain_levels = None
        if hasattr(env.scene.terrain, 'terrain_levels'):
            self.terrain_levels = env.scene.terrain.terrain_levels.tolist()
        self.ticks = 0
        self.worst = {}
        self.trace = []
        self.resets = 0
        self.original_update = env.scene.update
        env.scene.update = self.update

    def update(self, dt):
        result = self.original_update(dt)
        if dt <= 0:
            return result
        data = self.robot.data
        q, v, tau = (tensor(x) for x in (data.joint_pos, data.joint_vel, data.applied_torque))
        if not all(torch.isfinite(x).all() for x in [q, v, tau]):
            raise FloatingPointError('Non-finite native joint state')
        excess = torch.maximum(self.limits[..., 0] - q, q - self.limits[..., 1]).clamp_min(0)
        target = self.env.action_manager.get_term('joint_pos').processed_actions
        ids = self.env.action_manager.get_term('joint_pos')._joint_ids
        lim = self.limits[:, ids]
        target_excess = torch.maximum(lim[..., 0] - target, target - lim[..., 1]).clamp_min(0)
        assert target.shape == q.shape and (isinstance(ids, slice) or list(ids) == list(range(q.shape[1])))
        self.target_max = torch.maximum(self.target_max, target_excess)
        self.target_count += target_excess > 1e-4
        value = float(excess.max())
        if value > self.worst.get('excess_rad', -1):
            flat = int(excess.argmax())
            e, j = divmod(flat, len(self.names))
            self.worst = dict(tick=self.ticks, time_s=self.ticks * dt, env=e, joint=self.names[j],
                              excess_rad=value, q=float(q[e, j]), qvel=float(v[e, j]),
                              torque=float(tau[e, j]), target=float(target[e, j]),
                              limits=self.limits[e, j].tolist(), episode_age=int(self.env.episode_length_buf[e]))
        self.maximum = torch.maximum(self.maximum, excess)
        self.count += excess > 1e-4
        check_solver_overflow(self.solver.mjw_data)
        contacts = self.solver.mjw_data
        n = int(wp.to_torch(contacts.nacon)[0])
        forces = torch.zeros_like(self.contact_peak)
        if n:
            geoms = wp.to_torch(contacts.contact.geom)[:n].long()
            a, b = geoms[:, 0], geoms[:, 1]
            mask = (self.knee_geom[a] & self.hip_geom[b]) | (self.knee_geom[b] & self.hip_geom[a])
            selected = mask.any(-1).nonzero().flatten().to(torch.int32).contiguous()
            if len(selected):
                output = wp.zeros(len(selected), dtype=wp.spatial_vector, device=self.env.device)
                mjw.contact_force(self.solver.mjw_model, contacts, wp.from_torch(selected), False, output)
                normal = wp.to_torch(output)[:, 0].clamp_min(0)
                worlds = wp.to_torch(contacts.contact.worldid)[selected.long()].long()
                for side in range(2):
                    f = normal * mask[selected.long(), side]
                    forces[:, side].scatter_reduce_(0, worlds, f, reduce='amax', include_self=True)
        self.contact_peak = torch.maximum(self.contact_peak, forces)
        self.contact_run = torch.where(forces > 100, self.contact_run + dt, 0.)
        self.contact_max = torch.maximum(self.contact_max, self.contact_run)
        # A candidate lock requires all three signals, not merely contact or a soft-limit reward.
        stalled = ((v[:, self.knees].abs() < .1) & (tau[:, self.knees].abs() > 100)
                   & (forces > 100))
        self.stall_run = torch.where(stalled, self.stall_run + dt, 0.)
        self.stall_max = torch.maximum(self.stall_max, self.stall_run)
        if self.env.num_envs <= 8:
            self.trace.append(torch.cat([q, v, tau, target, forces,
                                        tensor(data.root_state_w)], -1).detach().cpu().numpy())
        self.ticks += 1
        return result

    def report(self):
        return dict(physics_ticks=self.ticks, samples=self.ticks * self.env.num_envs,
                    resets=self.resets, worst=self.worst,
                    monitored_contact_bodies=self.contact_bodies, initial_terrain_levels=self.terrain_levels,
                    per_env_actual_excess_rad=self.maximum.max(-1).values.tolist(),
                    actual_violation_envs=int((self.maximum > 1e-4).any(-1).sum()),
                    target_violation_envs=int((self.target_max > 1e-4).any(-1).sum()),
                    per_joint={name: dict(max_actual_excess_rad=float(self.maximum[:, j].max()),
                                         max_target_excess_rad=float(self.target_max[:, j].max()),
                                         actual_samples=int(self.count[:, j].sum()),
                                         target_samples=int(self.target_count[:, j].sum()))
                               for j, name in enumerate(self.names)},
                    knee_hip_contact_peak_n=self.contact_peak.tolist(),
                    knee_hip_contact_longest_s=self.contact_max.tolist(),
                    candidate_lock_longest_s=self.stall_max.tolist())


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = RP1PUFOEnvCfg() if args.mode == 'training' else RP1PUFOEvalEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.log_dir = str(args.output)
    cfg.actions.joint_pos.class_type = BodyImpactJointPositionAction
    if args.mode == 'push':
        assert args.num_envs == 2 and args.context_trace
        for key in ['randomize_camera_installation', 'randomize_rigid_body_material',
                    'randomize_rigid_body_com', 'scale_link_mass', 'scale_actuator_gains', 'push_robot']:
            setattr(cfg.events, key, None)
        for group in vars(cfg.observations).values():
            if hasattr(group, 'enable_corruption'):
                group.enable_corruption = False
        for actuator in cfg.scene.robot.actuators.values():
            if hasattr(actuator, 'max_delay'):
                actuator.min_delay = actuator.max_delay = 0
            if hasattr(actuator, 'rfi_lim'):
                actuator.rfi_lim = 0.
        cfg.episode_length_s = 100000.
        cfg.terminations.terrain_boundary = None
    env = gym.make('RP1-PUFO' if args.mode == 'training' else 'RP1-PUFO-EVAL', cfg=cfg).unwrapped
    agent_cfg = RP1PUFORunnerCfg().to_dict()
    agent_cfg.update(training_contract='pufo_h200_alignment_v6', logger='tensorboard',
                     use_mix_rollout=False, torch_compile_mode=None)
    runner = FbCprRunner(RslRlVecEnvWrapper(env), agent_cfg, log_dir=str(args.output),
                        device=env.device, inference_only=True)
    runner.load(args.checkpoint, load_optimizer=False)
    env._ufo_global_step = int(Path(args.checkpoint).stem.split('_')[-1])
    if args.mode == 'training':
        runner._build_expert_buffer()
    obs, _ = env.reset()
    robot = env.scene['robot']
    if args.mode == 'push':
        root = Path(__file__).resolve().parents[2]
        manifest_path = root / 'robo_newton/data/motions/rp1_lafan/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        motion_path = manifest_path.parent / next(c for c in manifest['clips'] if c['name'] == 'dance1_subject2')['npz']
        motion = np.load(motion_path)
        assert list(robot.joint_names) == manifest['npz']['joint_order']
        state = tensor(robot.data.default_root_state).clone()
        state[:, :3] = torch.as_tensor(motion['body_pos_w'][50, 0], device=env.device)
        state[:, :2] = 0.
        state[:, :3] += tensor(env.scene.env_origins)
        rot = Rotation.from_quat(motion['body_quat_w'][50, 0])
        yaw = np.arctan2(rot.as_matrix()[1, 0], rot.as_matrix()[0, 0])
        state[:, 3:7] = torch.as_tensor((Rotation.from_euler('z', -yaw) * rot).as_quat(), device=env.device)
        state[:, 7:] = 0.
        q = torch.as_tensor(motion['joint_pos'][50], device=env.device).repeat(args.num_envs, 1)
        robot.write_root_pose_to_sim_index(root_pose=state[:, :7])
        robot.write_root_velocity_to_sim_index(root_velocity=state[:, 7:])
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        env.sim.forward()
        env.scene.update(dt=0.)
        from robolab.tasks.manager_based.ufo.mdp.pufo_reset import refresh_reset_camera
        refresh_reset_camera(env, torch.arange(args.num_envs, device=env.device))
        runner.history.reset(torch.arange(args.num_envs, device=env.device))
        obs = env.observation_manager.compute()
        z = torch.as_tensor(np.load(args.context_trace)['context'], device=env.device).repeat(args.num_envs, 1)
        waist = robot.body_names.index('waist_yaw_link')
    probe = LimitProbe(env)
    setup = dict(mode=args.mode, checkpoint=args.checkpoint, seed=args.seed, num_envs=args.num_envs,
                 steps=args.steps, sample_hz=200, physics_hz=400, action_class=cfg.actions.joint_pos.class_type.__name__,
                 optimizer_updates=0, contract=agent_cfg['training_contract'],
                 context='expert plus random z; no live mixed z replay buffer' if args.mode == 'training' else 'saved same336M window5 held z, zero heading/velocity',
                 limits='physical joint_pos_limits, tolerance 1e-4rad; not soft reward limits',
                 lock_rule='simultaneous knee speed<0.1rad/s, torque>100Nm, knee-hip contact>100N; report uninterrupted durations')
    setup['force_n'] = args.force if args.mode == 'push' else None
    setup['actuators'] = {name: dict(class_name=type(act).__name__,
                                    effort_limit=tensor(act.effort_limit)[0].tolist(),
                                    stiffness=tensor(act.stiffness)[0].tolist(),
                                    damping=tensor(act.damping)[0].tolist())
                          for name, act in robot.actuators.items()}
    setup['env_config_source'] = __import__('inspect').getfile(type(cfg))
    setup['monitored_contact_bodies'] = probe.contact_bodies
    # Save the converted collision/limit model for read-only inspection of native parameters.
    import mujoco
    mujoco.mj_saveModel(probe.model, str(args.output / 'native_model.mjb'))
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2))
    try:
        with torch.no_grad():
            for step in range(args.steps):
                reset_ids = (env.episode_length_buf == 0).nonzero().flatten()
                runner.history.reset(reset_ids)
                probe.stall_run[reset_ids] = 0
                probe.contact_run[reset_ids] = 0
                parts = runner._obs_dict(obs)
                if args.mode == 'training':
                    heading, _ = runner._root_heading_velocity()
                    runner._context = runner.agent.advance_rollout_context(
                        runner._context, env.episode_length_buf, heading, runner.replay)
                    parts.update(runner.agent.context_observations(runner._context, heading))
                    z = runner._context.z
                else:
                    parts['heading'] = torch.zeros((args.num_envs, 2), device=env.device)
                    parts['velocity_command'] = torch.zeros((args.num_envs, 3), device=env.device)
                    t = step * env.step_dt
                    magnitude = args.force * max(0., min(1., (t - 3.) / .1)) if t < 3.75 else 0.
                    forces = torch.zeros((args.num_envs, 1, 3), device=env.device)
                    forces[:, 0, 0] = forces.new_tensor([magnitude, -magnitude])
                    robot.permanent_wrench_composer.reset()
                    robot.permanent_wrench_composer.add_forces_and_torques_index(
                        forces=forces, body_ids=[waist], is_global=True)
                action = runner._env_action(runner.agent.act(parts, z, mean=args.mode == 'push'))
                runner.history.add(runner._current_obs_values(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                probe.resets += int((terminated | truncated).sum())
                if args.mode == 'push' and (terminated | truncated).any():
                    raise RuntimeError('Unexpected reset in controlled push')
                if not torch.isfinite(reward).all():
                    raise FloatingPointError('Non-finite reward')
                if (step + 1) % 100 == 0:
                    print('[LIMITS]', step + 1, json.dumps(probe.worst), 'lock_s', float(probe.stall_max.max()), flush=True)
                    (args.output / 'partial.json').write_text(json.dumps(probe.report(), indent=2))
        (args.output / 'report.json').write_text(json.dumps(probe.report(), indent=2))
        if probe.trace:
            np.savez_compressed(args.output / 'trace.npz', values=np.stack(probe.trace),
                                joint_names=probe.names, limits=probe.limits.cpu().numpy())
    finally:
        env.scene.update = probe.original_update
        runner.writer.close()
        env.close()


if __name__ == '__main__':
    main()
    launcher.app.close()
