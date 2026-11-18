"""Upper-limb compliance under external load and smooth return to rest after release."""

import math

import torch
from isaaclab.managers import ManagerTermBase


def _smoothstep(value, bounds):
    lo, hi = bounds
    x = ((value - lo) / (hi - lo)).clamp(0., 1.)
    return x * x * (3. - 2. * x)


class full_body_return(ManagerTermBase):
    """Hold rest pose when unloaded; penalize leaving rest and returning too fast.

    Posture is released per arm by arm loads. Velocity terms are additionally released by
    whole-body follow (force presence or a nonzero ramped follow target), so stepping under a
    waist/arm pull or torso yaw torque is never charged for arm shake.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('return_std', 'recovery_time_s', 'posture_weight',
                     'return_max_speed', 'return_speed_threshold'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        threshold = p.get('motion_target_threshold', 0.)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('motion_target_threshold must be finite and nonnegative')
        deadband = p.get('away_speed_deadband', 0.)
        if not math.isfinite(deadband) or deadband < 0:
            raise ValueError('away_speed_deadband must be finite and nonnegative')
        for name in ('arm_velocity_weight', 'arm_fast_return_weight'):
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
        n, device = env.num_envs, env.device
        self.gate = torch.ones(n, device=device)
        self.arm_gate = torch.ones(n, 2, device=device)
        self.joint_gate = torch.ones(n, 2, 5, device=device)
        self.vel_gate = torch.ones(n, 2, 5, device=device)
        self.overspeed_gate = torch.ones(n, 2, 5, device=device)
        floor = p.get('return_speed_floor')
        if floor is not None and (not math.isfinite(floor) or not 0 < floor <= p['return_speed_threshold']):
            raise ValueError('return_speed_floor must be positive and at most return_speed_threshold')
        for name in ('motion_speed_range', 'motion_yaw_range'):
            lo, hi = p.get(name, (0.02, 0.10) if name == 'motion_speed_range' else (0.05, 0.20))
            if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
                raise ValueError(f'{name} must be finite, nonnegative and increasing')

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.gate[ids] = 1.
        self.arm_gate[ids] = 1.
        self.joint_gate[ids] = 1.
        self.vel_gate[ids] = 1.
        self.overspeed_gate[ids] = 1.

    def __call__(self, env, asset_cfg, command_name, return_std, recovery_time_s, posture_weight,
                 arm_velocity_weight=2.0, return_max_speed=1.0, arm_fast_return_weight=2.0,
                 return_speed_threshold=1.0, motion_target_threshold=0.0,
                 motion_speed_range=(0.02, 0.10), motion_yaw_range=(0.05, 0.20),
                 whole_body_releases_posture=False, away_speed_deadband=0.0,
                 return_speed_floor=None, overspeed_saturates=True):
        arm = env.reward_manager.get_term_cfg('arm_compliance').func
        body = env.reward_manager.get_term_cfg('track_lin_vel_xy_exp').func
        yaw = env.reward_manager.get_term_cfg('track_ang_vel_z_exp').func
        zero = (env.command_manager.get_command(command_name) == 0).all(-1)
        # A load anywhere on one arm (joint torque or link pull) releases that whole arm: the
        # loaded joint drags its neighbours through gravity and inertia while yielding.
        arm_loaded = torch.maximum(
            arm.passive_observed_gate.amax(-1),
            _smoothstep(arm.filtered_force_w.norm(dim=-1), (0.5, 2.0)),
        )[:, :, None].expand_as(self.joint_gate)
        # Whole-body follow (waist/arm pull or torso yaw torque) and the ramped follow target that
        # outlasts the load while the robot decelerates: stepping shakes the arms, so the velocity
        # terms are released; posture keeps its deadband unless explicitly released too.
        motion = torch.maximum(
            _smoothstep(body.target_velocity_b.norm(dim=-1), motion_speed_range),
            _smoothstep(yaw.target_yaw_rate.abs(), motion_yaw_range),
        )
        whole = torch.maximum(torch.maximum(body.force_gate, yaw.force_gate), motion)[:, None, None]
        idle = zero.to(self.joint_gate.dtype)[:, None, None]
        alpha = -math.expm1(-env.step_dt / recovery_time_s)
        # Overspeed is charged on the instantaneous release (no recovery lag): a snap-back happens
        # in the first 0.3-0.5 s after release, before the lagged velocity gate has reopened.
        whole_release = torch.maximum(arm_loaded, whole)
        self.overspeed_gate.copy_(idle * (1. - whole_release))
        pose_release = whole_release if whole_body_releases_posture else arm_loaded
        for gate, release in ((self.joint_gate, pose_release), (self.vel_gate, whole_release)):
            target = idle * (1. - release)
            gate.copy_(torch.minimum(target, torch.lerp(gate, target, alpha)))
        self.arm_gate.copy_(self.joint_gate.mean(-1))
        self.gate.copy_(self.arm_gate.mean(-1))

        data = env.scene[asset_cfg.name].data
        q = data.joint_pos.torch[:, self._joint_flat].reshape(-1, 2, 5)
        rest = (arm.rest_positions if arm.rest_positions is not None
                else data.default_joint_pos.torch[:, self._joint_flat].reshape(-1, 2, 5))
        error = q - rest
        velocity = data.joint_vel.torch[:, self._joint_flat].reshape_as(error)
        away = (velocity * error.sign()).clamp_min(0.)
        toward = (velocity * -error.sign()).clamp_min(0.)
        # Return speed limit shrinks near rest: far from rest the arm may return at
        # return_speed_threshold, the last stretch is approached slowly instead of snapping in.
        if return_speed_floor is None:
            limit = torch.full_like(error, return_speed_threshold)
        else:
            limit = (return_speed_floor + error.abs()).clamp_max(return_speed_threshold)
        # Worst joint, not the 10-joint sum: one raised arm is enough, small rest noise must not stack.
        excess = (error.abs() - motion_target_threshold).clamp_min(0.)
        # Fade posture as the joint already returns at the speed limit, otherwise snap-back beats overspeed.
        returning = (toward / limit).clamp(0., 1.)
        posture = posture_weight * (
            torch.tanh(excess / return_std) * self.joint_gate * (1. - returning)).amax((1, 2))
        # Small outward drift (sway, PD sag, exploration) is free; only deliberate swings pay.
        swing = (away - away_speed_deadband).clamp_min(0.)
        vel_away = arm_velocity_weight * (torch.tanh(swing / return_max_speed) * self.vel_gate).amax((1, 2))
        over = (toward / limit - 1.).clamp_min(0.)
        if overspeed_saturates:
            over_cost = torch.tanh(over.square())
        else:
            # Huber: quadratic up to twice the limit, then linear, so 10 rad/s costs far more than 2 rad/s.
            over_cost = torch.where(over < 1., over.square(), 2. * over - 1.)
        overspeed = arm_fast_return_weight * (over_cost * self.overspeed_gate).amax((1, 2))

        log = env.extras.setdefault('log', {})
        log['Interaction/return_gate'] = self.gate.mean()
        log['Interaction/return_velocity_gate'] = self.vel_gate.mean()
        log['Interaction/return_motion_release'] = (zero.float() * whole[:, 0, 0]).mean()
        log['Interaction/return_error_rad'] = (
            (error.abs() * self.joint_gate).sum() / self.joint_gate.sum().clamp_min(1e-6))
        log['Interaction/return_away_rad_s'] = (
            (away * self.joint_gate).sum() / self.joint_gate.sum().clamp_min(1e-6))
        log['Interaction/return_toward_rad_s'] = (
            (toward * self.joint_gate).sum() / self.joint_gate.sum().clamp_min(1e-6))
        log['Interaction/return_overspeed_cost'] = overspeed.mean()
        log['Interaction/return_peak_toward_rad_s'] = (toward * self.overspeed_gate).amax((1, 2)).mean()
        return posture + vel_away + overspeed


class arm_compliance(ManagerTermBase):
    """Loaded-arm 蟿v yield minus opposing motor and overspeed; each term is standalone."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('force_filter_time_s', 'power_scale_w', 'force_scale_bw', 'torque_scale_nm',
                     'damping_allowance', 'max_joint_speed', 'limit_margin', 'yield_scale_w'):
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
        self.yield_scale_w = p['yield_scale_w']
        self.damping_allowance = p['damping_allowance']
        self.resistance_tolerance_nm = p.get('resistance_tolerance_nm', 0.5)
        self.limit_margin = p['limit_margin']
        self.max_joint_speed = p['max_joint_speed']
        self.overspeed_weight = p.get('overspeed_weight', 0.3)
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
        self.style_relief = torch.zeros(n, 2, device=device)
        if not math.isfinite(p.get('style_relief_release_s', 1.0)) or p.get('style_relief_release_s', 1.0) <= 0:
            raise ValueError('style_relief_release_s must be finite and positive')
        for name in ('arm_style_scale', 'body_style_scale'):
            if not 0 <= p.get(name, 0.5) <= 1:
                raise ValueError(f'{name} must be within [0, 1]')
        self.joint_return_gate = torch.ones(n, 2, 5, device=device)
        self.motion_blend = torch.zeros(n, device=device)
        self.passive_observed_gate = torch.zeros(n, 2, 5, device=device)
        self._power_sum = torch.zeros(n, 2, 5, device=device)
        self._hold_sum = torch.zeros(n, 2, 5, device=device)
        self._yield_sum = torch.zeros(n, 2, 5, device=device)
        self._speed_sum = torch.zeros(n, 2, 5, device=device)
        self._open_sum = torch.zeros(n, 2, 5, device=device)
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
        self._yield_sum.zero_()
        self._speed_sum.zero_()
        self._open_sum.zero_()
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
        opened = gate * free

        power = (motor * v).clamp_min(0.)
        opposing = (-ext.sign() * motor).clamp_min(0.)
        allowance = self.resistance_tolerance_nm + self.damping_allowance * v.abs()
        hold = (opposing - allowance).clamp_min(0.)
        yield_power = (ext * v).clamp_min(0.)
        excess = (v.abs() / self.max_joint_speed - 1.).clamp_min(0.)
        bump = (1. - free) * gate * presence

        self._power_sum.add_(power * weight)
        self._hold_sum.add_(hold * weight)
        self._yield_sum.add_(yield_power * opened)
        self._speed_sum.add_(excess.square() * opened + bump)
        self._open_sum.add_(opened)
        self.passive_observed_gate.copy_(gate)
        self._substeps += 1
        self._snapshot = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buf in (self.filtered_force_w, self.filtered_joint_torque, self.activation, self.style_relief,
                    self.motion_blend, self.passive_observed_gate,
                    self._power_sum, self._hold_sum, self._yield_sum, self._speed_sum, self._open_sum):
            buf[ids] = 0
        for force in self._contact_forces:
            force[ids] = 0
        self.joint_return_gate[ids] = 1

    def __call__(self, env, asset_cfg, force_filter_time_s, force_deadband_bw, force_scale_bw,
                 power_scale_w, torque_scale_nm, damping_allowance, max_joint_speed, limit_margin,
                 passive_joint_torque_range_nm, yield_scale_w,
                 resistance_tolerance_nm=0.5, overspeed_weight=0.3,
                 command_name='base_velocity', rest_joint_pos=None, all_arm_links=True,
                 style_relief_release_s=1.0, arm_style_scale=0.5, body_style_scale=0.5):
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
        # At zero command the style relief rises with the load and, after release, fades over the
        # return instead of snapping back while the body still recovers. Arm loads and whole-body
        # follow (pull, yaw torque, or a still-ramping follow target) get separate style scales; the
        # PPOAMP floor (force_style_reward_scale) must equal the smaller one: scale = 1-(1-floor)*act.
        floor = min(arm_style_scale, body_style_scale)
        body = env.reward_manager.get_term_cfg('track_lin_vel_xy_exp').func
        yaw = env.reward_manager.get_term_cfg('track_ang_vel_z_exp').func
        follow = torch.maximum(
            torch.maximum(body.force_gate, yaw.force_gate),
            torch.maximum(_smoothstep(body.force_target_velocity_b.norm(dim=-1), (0.02, 0.10)),
                          _smoothstep(yaw.force_target_yaw_rate.abs(), (0.05, 0.20))))
        load = torch.maximum(
            gate.amax(-1) * (1. - arm_style_scale) / max(1. - floor, 1e-6),
            follow[:, None] * (1. - body_style_scale) / max(1. - floor, 1e-6))
        relief = torch.maximum(load, self.style_relief * math.exp(-env.step_dt / style_relief_release_s))
        self.style_relief.copy_(torch.where(zero_command[:, None], relief, torch.zeros_like(relief)))
        self.activation.copy_(torch.where(
            zero_command[:, None], self.style_relief,
            # Commanded walking keeps the arm-load relief it had (arm_style_scale), not the new floor.
            torch.maximum(force_act, joint_act) * (1. - arm_style_scale) / max(1. - floor, 1e-6)))
        env.extras['amp_force_activation'] = self.activation.amax(dim=-1).detach()
        log_style = env.extras.setdefault('log', {})
        log_style['Interaction/style_follow_gate'] = (follow * zero_command).mean()

        target = torch.where(zero_command[:, None, None], 1. - gate, torch.ones_like(gate))
        recovering = torch.lerp(self.joint_return_gate, target, -math.expm1(-env.step_dt / 0.3))
        self.joint_return_gate.copy_(torch.minimum(target, recovering))

        linear = torch.maximum(command[:, :2].norm(dim=-1), data.root_lin_vel_b.torch[:, :2].norm(dim=-1))
        yaw = torch.maximum(command[:, 2].abs(), data.root_ang_vel_b.torch[:, 2].abs())
        moving = torch.maximum((linear - 0.05) / 0.20, (yaw - 0.10) / 0.40).clamp(0., 1.)
        self.motion_blend.lerp_(moving * moving * (3. - 2. * moving), -math.expm1(-env.step_dt / 0.3))

        power = self._power_sum / steps
        hold = self._hold_sum / steps
        mass = self._open_sum.sum((1, 2)).clamp_min(1.)
        yield_w = self._yield_sum.sum((1, 2)) / mass / self.yield_scale_w
        speed = self._speed_sum.sum((1, 2)) / mass
        cost = (power / self.power_scale_w + hold / self.torque_scale_nm).sum((1, 2))
        cost = cost + self.overspeed_weight * speed

        log = env.extras.setdefault('log', {})
        log['Interaction/arm_positive_power_w'] = power.sum((1, 2)).mean()
        log['Interaction/arm_hold_nm'] = hold.sum((1, 2)).mean()
        log['Interaction/arm_yield_w'] = (self._yield_sum.sum((1, 2)) / mass).mean()
        log['Interaction/arm_speed_sq'] = speed.mean()
        log['Interaction/arm_load_fraction'] = gate.mean()
        return yield_w - cost
