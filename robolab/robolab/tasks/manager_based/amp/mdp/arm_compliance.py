"""Upper-limb compliance under external load and smooth return to rest after release."""

import math

import torch
from isaaclab.managers import ManagerTermBase


def _smoothstep(value, bounds):
    lo, hi = bounds
    x = ((value - lo) / (hi - lo)).clamp(0., 1.)
    return x * x * (3. - 2. * x)


class full_body_return(ManagerTermBase):
    """Smoothly return each unloaded arm to rest when idle and not force-following."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('return_std', 'recovery_time_s', 'posture_weight',
                     'return_max_speed', 'return_speed_threshold'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('arm_velocity_weight', 'arm_fast_return_weight', 'motion_target_threshold'):
            if not math.isfinite(p.get(name, 0.)) or p.get(name, 0.) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        names = env.scene[p['asset_cfg'].name].joint_names
        self.arm_joint_ids = [
            [names.index(f'{side}_{part}_joint')
             for part in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist')]
            for side in ('left', 'right')
        ]
        self._joint_flat = [j for side in self.arm_joint_ids for j in side]
        self.arm_local_load = True
        self.gate = torch.ones(env.num_envs, device=env.device)
        self.arm_gate = torch.ones(env.num_envs, 2, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.gate[ids] = 1.
        self.arm_gate[ids] = 1.

    def __call__(self, env, asset_cfg, command_name, return_std, recovery_time_s, posture_weight,
                 arm_velocity_weight=2.0, return_max_speed=0.4, arm_fast_return_weight=8.0,
                 return_speed_threshold=0.8, motion_target_threshold=0.3):
        from .soft import reward_motion_target
        arm = env.reward_manager.get_term_cfg('arm_compliance').func
        zero = (env.command_manager.get_command(command_name) == 0).all(-1)
        motion = reward_motion_target(env, command_name)
        moving = (motion[:, :2].norm(dim=-1) > motion_target_threshold) | (
            motion[:, 2].abs() > motion_target_threshold)

        loaded = arm.passive_observed_gate.amax(-1)
        idle = (zero & ~moving).to(self.arm_gate.dtype)[:, None]
        target = idle * (1. - loaded)
        recovering = torch.lerp(self.arm_gate, target, -math.expm1(-env.step_dt / recovery_time_s))
        self.arm_gate.copy_(torch.minimum(target, recovering))
        self.gate.copy_(self.arm_gate.mean(-1))

        data = env.scene[asset_cfg.name].data
        q = data.joint_pos.torch[:, self._joint_flat].reshape(-1, 2, 5)
        rest = (arm.rest_positions if arm.rest_positions is not None
                else data.default_joint_pos.torch[:, self._joint_flat].reshape(-1, 2, 5))
        error = q - rest
        velocity = data.joint_vel.torch[:, self._joint_flat].reshape_as(error)

        gate = self.arm_gate[..., None]
        immediate = target[..., None]
        # Bounded posture avoids distance-proportional rush; desired velocity guides smooth convergence.
        posture = posture_weight * (torch.tanh(error.abs() / return_std) * gate).mean(-1)
        desired_vel = -return_max_speed * torch.tanh(error / 0.15)
        vel_track = arm_velocity_weight * (((velocity - desired_vel) / return_max_speed).square() * gate).mean(-1)
        # Immediate overspeed brake stops rebound spikes right at load release before arm_gate ramps up.
        overspeed = arm_fast_return_weight * (
            (velocity.abs() / return_speed_threshold - 1.).clamp_min(0.).square() * immediate).mean(-1)

        log = env.extras.setdefault('log', {})
        log['Interaction/return_gate'] = self.gate.mean()
        log['Interaction/return_error_rad'] = (
            error.abs().mean(-1) * self.arm_gate).sum() / self.arm_gate.sum().clamp_min(1e-6)
        return (posture + vel_track + overspeed).sum(-1) * (5. / 24.)


class arm_compliance(ManagerTermBase):
    """Substep-averaged positive power + excess opposing effort with soft-limit protection."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('force_filter_time_s', 'power_scale_w', 'force_scale_bw', 'torque_scale_nm',
                     'damping_allowance', 'max_joint_speed', 'limit_margin'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('force_deadband_bw', 'resistance_tolerance_nm', 'overspeed_weight'):
            if not math.isfinite(p.get(name, 0.)) or p.get(name, 0.) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        lo, hi = p['passive_joint_torque_range_nm']
        if not all(math.isfinite(x) for x in (lo, hi)) or not 0 <= lo < hi:
            raise ValueError('passive_joint_torque_range_nm must be finite, nonnegative and increasing')
        if not getattr(env.cfg, 'force_estimation_compact', False):
            raise ValueError('Arm compliance requires compact wrench labels')

        # Positive passive_weight enables substep sampling in AmpEnv.step.
        self.passive_weight = 1.0
        self.power_scale_w = p['power_scale_w']
        self.torque_scale_nm = p['torque_scale_nm']
        self.damping_allowance = p['damping_allowance']
        self.resistance_tolerance_nm = p.get('resistance_tolerance_nm', 0.5)
        self.limit_margin = p['limit_margin']
        self.max_joint_speed = p['max_joint_speed']
        self.overspeed_weight = p.get('overspeed_weight', 0.1)
        self.torque_range_nm = p['passive_joint_torque_range_nm']

        self.robot = env.scene[p['asset_cfg'].name]
        self.wrist_ids = p['asset_cfg'].body_ids
        if len(self.wrist_ids) != 2:
            raise ValueError('Arm compliance requires left and right wrists')
        self.joint_ids = [
            self.robot.find_joints(
                [f'{side}_{part}_joint'
                 for part in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist')],
                preserve_order=True)[0]
            for side in ('left', 'right')
        ]
        if any(len(ids) != 5 for ids in self.joint_ids):
            raise ValueError('Arm compliance requires five joints per RP1 arm')
        self._joint_flat = [j for arm in self.joint_ids for j in arm]

        rest = p.get('rest_joint_pos')
        self.rest_positions = None
        if rest is not None:
            ordered = [[f'{side}_{part}_joint'
                        for part in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist')]
                       for side in ('left', 'right')]
            if set(rest) != {name for arm in ordered for name in arm} or not all(
                    math.isfinite(v) for v in rest.values()):
                raise ValueError('rest_joint_pos must contain all ten arm joints with finite targets')
            self.rest_positions = torch.tensor(
                [[rest[name] for name in arm] for arm in ordered],
                device=env.device, dtype=self.robot.data.joint_pos.torch.dtype)

        n, device = env.num_envs, env.device
        self.filtered_force_w = torch.zeros(n, 2, 3, device=device)
        self.filtered_joint_torque = torch.zeros(n, 2, 5, device=device)
        self.activation = torch.zeros(n, 2, device=device)
        self.joint_return_gate = torch.ones(n, 2, 5, device=device)
        self.motion_blend = torch.zeros(n, device=device)
        self.passive_observed_gate = torch.zeros(n, 2, 5, device=device)
        self._power_sum = torch.zeros(n, 2, 5, device=device)
        self._hold_sum = torch.zeros(n, 2, 5, device=device)
        self._speed_sum = torch.zeros(n, 2, 5, device=device)
        self._substeps = 0
        self._snapshot = None
        self._contact_slots = None
        self._contact_forces = []

    def _ensure_contacts(self, push, all_arm_links):
        if self._contact_slots is not None:
            return
        self._contact_slots = []
        for side, wrist_id in enumerate(self.wrist_ids):
            if all_arm_links:
                prefix = 'left_' if side == 0 else 'right_'
                slots = [i for i, b in enumerate(push.body_ids)
                         if self.robot.body_names[b].startswith(prefix)
                         and any(part in self.robot.body_names[b]
                                 for part in ('shoulder_', 'elbow_', 'wrist_'))]
            else:
                slots = [push.body_ids.index(wrist_id)] if wrist_id in push.body_ids else []
            self._contact_slots.append(torch.tensor(slots, device=self._env.device, dtype=torch.long))
            self._contact_forces.append(torch.zeros(
                self._env.num_envs, len(slots), 3, device=self._env.device))

    def _compute_gate(self, zero_command, ext):
        return _smoothstep(ext.abs(), self.torque_range_nm) * zero_command[:, None, None]

    def begin_passive_step(self):
        self._power_sum.zero_()
        self._hold_sum.zero_()
        self._speed_sum.zero_()
        self._substeps = 0
        self._snapshot = None

    def prepare_passive_substep(self, env):
        data = self.robot.data
        push = env.event_manager.get_term_cfg('sustained_standing_push').func
        self._ensure_contacts(push, self.cfg.params.get('all_arm_links', True))
        q = data.joint_pos.torch[:, self._joint_flat].reshape(-1, 2, 5).clone()
        v = data.joint_vel.torch[:, self._joint_flat].reshape_as(q).clone()
        ext = push.current_compact_wrench()[:, 6:16].reshape_as(q)
        zero = (env.command_manager.get_command(self.cfg.params['command_name']) == 0).all(-1)
        gate = self._compute_gate(zero, ext)
        self._snapshot = (q, v, ext, gate)

    def finish_passive_substep(self):
        q, v, ext, gate = self._snapshot
        data = self.robot.data
        limits = data.soft_joint_pos_limits.torch[:, self._joint_flat].reshape(-1, 2, 5, 2)
        motor = data.applied_torque.torch[:, self._joint_flat].reshape_as(q)

        # Fade compliance as an outward load pushes the joint into its soft limit.
        remaining = torch.where(ext >= 0, limits[..., 1] - q, q - limits[..., 0])
        free = (remaining / self.limit_margin).clamp(0., 1.)
        presence = (ext.abs() / self.torque_scale_nm).clamp(0., 1.)
        weight = gate * presence * free

        power = (motor * v).clamp_min(0.)
        opposing = (-ext.sign() * motor).clamp_min(0.)
        allowance = self.resistance_tolerance_nm + self.damping_allowance * v.abs()
        hold = (opposing - allowance).clamp_min(0.)
        speed = (v.abs() / self.max_joint_speed - 1.).clamp_min(0.).square()

        self._power_sum.add_(power * weight)
        self._hold_sum.add_(hold * weight)
        self._speed_sum.add_(speed * gate)
        self.passive_observed_gate.copy_(gate)
        self._substeps += 1
        self._snapshot = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buf in (self.filtered_force_w, self.filtered_joint_torque, self.activation,
                    self.motion_blend, self.passive_observed_gate,
                    self._power_sum, self._hold_sum, self._speed_sum):
            buf[ids] = 0
        for force in self._contact_forces:
            force[ids] = 0
        self.joint_return_gate[ids] = 1

    def __call__(self, env, asset_cfg, force_filter_time_s, force_deadband_bw, force_scale_bw,
                 power_scale_w, torque_scale_nm, damping_allowance, max_joint_speed, limit_margin,
                 passive_joint_torque_range_nm,
                 resistance_tolerance_nm=0.5, overspeed_weight=0.1,
                 command_name='base_velocity', rest_joint_pos=None, all_arm_links=True):
        push = env.event_manager.get_term_cfg('sustained_standing_push').func
        self._ensure_contacts(push, all_arm_links)
        data = self.robot.data
        bw = data.body_mass.torch.sum(-1) * abs(env.cfg.sim.gravity[2])
        command = env.command_manager.get_command(command_name)
        zero_command = (command == 0).all(-1)
        alpha = -math.expm1(-env.step_dt / force_filter_time_s)

        ext = push.current_compact_wrench()[:, 6:16].reshape(-1, 2, 5)
        self.filtered_joint_torque.lerp_(ext, alpha)

        magnitude = torch.zeros_like(self.activation)
        for side, slots in enumerate(self._contact_slots):
            if slots.numel() == 0:
                continue
            filtered = self._contact_forces[side]
            filtered.lerp_(push.applied_force[:, slots], alpha)
            self.filtered_force_w[:, side] = filtered.sum(1)
            magnitude[:, side] = filtered.norm(dim=-1).sum(-1)

        if self._substeps == 0:
            self.prepare_passive_substep(env)
            self.finish_passive_substep()
        gate = self.passive_observed_gate
        steps = float(self._substeps)

        excess = (magnitude / bw[:, None].clamp_min(1e-6) - force_deadband_bw).clamp_min(0.)
        force_act = (excess / force_scale_bw).clamp(0., 1.)
        joint_act = ((self.filtered_joint_torque.abs() - 0.1) / 0.4).clamp(0., 1.).amax(-1)
        self.activation.copy_(torch.where(
            zero_command[:, None], gate.amax(-1), torch.maximum(force_act, joint_act)))
        env.extras['amp_force_activation'] = self.activation.amax(dim=-1).detach()

        target = torch.where(zero_command[:, None, None], 1. - gate, torch.ones_like(gate))
        recovering = torch.lerp(self.joint_return_gate, target, -math.expm1(-env.step_dt / 0.3))
        self.joint_return_gate.copy_(torch.minimum(target, recovering))

        linear = torch.maximum(command[:, :2].norm(dim=-1), data.root_lin_vel_b.torch[:, :2].norm(dim=-1))
        yaw = torch.maximum(command[:, 2].abs(), data.root_ang_vel_b.torch[:, 2].abs())
        moving = torch.maximum((linear - 0.05) / 0.20, (yaw - 0.10) / 0.40).clamp(0., 1.)
        self.motion_blend.lerp_(moving * moving * (3. - 2. * moving), -math.expm1(-env.step_dt / 0.3))

        power = self._power_sum / steps
        hold = self._hold_sum / steps
        speed = self._speed_sum / steps
        cost = (power / self.power_scale_w + hold / self.torque_scale_nm
                + self.overspeed_weight * speed).sum((1, 2))

        log = env.extras.setdefault('log', {})
        log['Interaction/arm_positive_power_w'] = power.sum((1, 2)).mean()
        log['Interaction/arm_hold_nm'] = hold.sum((1, 2)).mean()
        log['Interaction/arm_load_fraction'] = gate.mean()
        return -cost
