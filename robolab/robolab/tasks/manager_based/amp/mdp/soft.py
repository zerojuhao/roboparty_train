"""Task partitions, AMP route, yaw compliance, and stepping rewards for RP1 AMP Soft."""

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

from .body_compliance import direct_tanh_target, load_presence
from .commands import FilteredUniformVelocityCommand
from .rewards import feet_air_time_positive_biped


class SoftVelocityCommand(FilteredUniformVelocityCommand):
    """Nonzero user commands identify locomotion; zero commands allow yielding."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        fraction = env.cfg.compliance_env_fraction
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('compliance_env_fraction must be within [0, 1]')
        count = round(self.num_envs * fraction)
        commanded_fraction = env.cfg.command_compliance_env_fraction
        if not math.isfinite(commanded_fraction) or not 0 <= commanded_fraction <= 1 - fraction:
            raise ValueError('command_compliance_env_fraction must fit outside zero-command environments')
        commanded_count = round(self.num_envs * commanded_fraction)
        if 0 < fraction < 1 and not 0 < count < self.num_envs:
            raise ValueError('Soft training needs at least one environment of each type per rank')
        if cfg.heading_command:
            raise ValueError('Soft commands do not support heading overrides')
        self.compliance_envs = torch.arange(self.num_envs, device=self.device) >= self.num_envs - count
        self.interaction_envs = torch.arange(self.num_envs, device=self.device) >= self.num_envs - count - commanded_count

    def _resample_command(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)
        fixed = env_ids[self.compliance_envs[env_ids]]
        self.is_standing_env[fixed] = True
        self.vel_command_b[fixed] = 0


def amp_task_route(env):
    """Use full-body AMP for every command and lower-body AMP at zero command."""
    zero_command = (env.command_manager.get_command('base_velocity') == 0).all(-1)
    return zero_command[:, None].float()


class yaw_compliance(ManagerTermBase):
    """Track yaw commands, or sustained explicit torso yaw torque at zero command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('std', 'max_yaw_rate', 'reference_length', 'torque_scale_bw', 'filter_time_s',
                     'target_tanh_k', 'target_accel'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('torque_deadband_bw', 'target_sustain_s'):
            if not math.isfinite(p.get(name, 0.)) or p.get(name, 0.) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        lo, hi = p['target_torque_range_nm']
        if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
            raise ValueError('target_torque_range_nm must be finite, nonnegative and increasing')

        n, device = env.num_envs, env.device
        self.filtered_torque_z = torch.zeros(n, device=device)
        self.target_yaw_rate = torch.zeros(n, device=device)
        self.force_target_yaw_rate = torch.zeros(n, device=device)
        self.force_gate = torch.zeros(n, device=device)
        self.target_load_age = torch.zeros(n, device=device)
        self._force_error_sum = torch.zeros(n, device=device)
        self._force_count = torch.zeros(n, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        samples = self._force_count[ids].sum()
        err = self._force_error_sum[ids].sum() / samples.clamp_min(1.)
        log = self._env.extras.setdefault('log', {})
        log['Metrics/force_target/error_vel_yaw'] = err
        log['Interaction/force_target_error_yaw_rps'] = err
        for buf in (self.filtered_torque_z, self.target_yaw_rate, self.force_target_yaw_rate,
                    self.force_gate, self.target_load_age, self._force_error_sum, self._force_count):
            buf[ids] = 0

    def __call__(self, env, command_name, std, max_yaw_rate, reference_length,
                 torque_deadband_bw, torque_scale_bw, filter_time_s,
                 target_torque_range_nm=(10., 20.), target_tanh_k=2.0, target_sustain_s=0.3,
                 target_accel=2.0):
        robot = env.scene['robot']
        push = env.event_manager.get_term_cfg('sustained_standing_push').func
        torque_b = quat_apply_inverse(robot.data.root_quat_w.torch, push.explicit_torso_torque_w)
        torque_z = torque_b[:, 2]
        self.filtered_torque_z.lerp_(torque_z, -math.expm1(-env.step_dt / filter_time_s))

        bw = robot.data.body_mass.torch.sum(-1) * abs(env.cfg.sim.gravity[2])
        denom = (bw * reference_length).clamp_min(1e-6)
        ratio = torch.minimum(self.filtered_torque_z.abs(), torque_z.abs()) / denom
        presence = load_presence((ratio - torque_deadband_bw).clamp_min(0.), torque_scale_bw)
        self.force_gate.copy_(presence)

        command = env.command_manager.get_command(command_name)
        zero_command = (command == 0).all(-1)
        yaw_rate = robot.data.root_ang_vel_b.torch[:, 2]
        sign = self.filtered_torque_z.sign()

        magnitude = torch.minimum(self.filtered_torque_z.abs(), torque_z.abs())
        target_mag = direct_tanh_target(magnitude, target_torque_range_nm, target_tanh_k) * max_yaw_rate
        if target_sustain_s > 0:
            above = magnitude >= target_torque_range_nm[0]
            self.target_load_age.copy_(torch.where(
                above, self.target_load_age + env.step_dt, torch.zeros_like(self.target_load_age)))
            target_mag = target_mag * (self.target_load_age >= target_sustain_s - 1e-6).to(target_mag.dtype)

        change = sign * target_mag - self.force_target_yaw_rate
        self.force_target_yaw_rate.add_(change * (
            target_accel * env.step_dt / change.abs().clamp_min(1e-6)).clamp_max(1.))
        self.target_yaw_rate.copy_(torch.where(zero_command, self.force_target_yaw_rate, command[:, 2]))

        error = yaw_rate - self.target_yaw_rate
        mask = zero_command & (presence > 0.5)
        self._force_error_sum.add_(error.abs() * mask)
        self._force_count.add_(mask)
        return torch.exp(-error.square() / std**2)


def reward_motion_target(env, command_name):
    """Resolve motion intent from force-target trackers without changing the actor command."""
    command = env.command_manager.get_command(command_name)
    linear = env.reward_manager.get_term_cfg('track_lin_vel_xy_exp').func
    yaw = env.reward_manager.get_term_cfg('track_ang_vel_z_exp').func
    target = torch.cat((linear.target_velocity_b, yaw.target_yaw_rate[:, None]), dim=-1)
    return torch.where((command == 0).all(-1, keepdim=True), target, command)


class soft_feet_air_time(ManagerTermBase):
    """Credit completed steps toward the motion target, scaled by capped lift above the last support height."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        if not all(math.isfinite(p[k]) and p[k] >= 0 for k in ('threshold', 'command_yaw_threshold') if k in p):
            raise ValueError('air-time thresholds must be finite and non-negative')
        if not 0 < p.get('target_height', 0.05) < math.inf:
            raise ValueError('air-time target height must be finite and positive')
        sensor_cfg, asset_cfg = p['sensor_cfg'], p['asset_cfg']
        sensor, robot = env.scene.sensors[sensor_cfg.name], env.scene[asset_cfg.name]
        names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        if len(names) != 2 or names != [robot.body_names[i] for i in asset_cfg.body_ids]:
            raise ValueError('Air-time height requires two corresponding foot sensors and bodies')
        self.support_height = torch.zeros(env.num_envs, 2, device=env.device)
        self.peak_lift = torch.zeros_like(self.support_height)
        self.have_support = torch.zeros_like(self.support_height, dtype=torch.bool)
        self.eligible = torch.zeros_like(self.have_support)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.support_height[ids] = 0
        self.peak_lift[ids] = 0
        self.have_support[ids] = False
        self.eligible[ids] = False

    def __call__(self, env, command_name, threshold, sensor_cfg, asset_cfg,
                 target_air_time=0.3, target_height=0.05, command_yaw_threshold=None):
        # Steps count while the command or force target asks for motion; commanded turns use their own gate.
        target = reward_motion_target(env, command_name)
        commanded = (env.command_manager.get_command(command_name) != 0).any(-1)
        yaw_gate = torch.where(commanded, threshold if command_yaw_threshold is None else command_yaw_threshold, threshold)
        activity = (target[:, :2].norm(dim=-1) > threshold) | (target[:, 2].abs() > yaw_gate)
        sensor, robot = env.scene.sensors[sensor_cfg.name], env.scene[asset_cfg.name]
        height = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
        first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].bool()
        airborne = sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0
        supported = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
        self.eligible &= activity[:, None] & self.have_support
        self.peak_lift.copy_(torch.where(
            airborne & self.eligible, torch.maximum(self.peak_lift, height - self.support_height), self.peak_lift))
        fraction = self.peak_lift.clamp(0, target_height) / target_height
        height_score = fraction.tanh() / math.tanh(1.)
        reward = feet_air_time_positive_biped(
            env, command_name, threshold, sensor_cfg, target_air_time,
            activity=activity.float(), linear_duration=True)
        reward *= (height_score * first * self.eligible).amax(-1)
        self.support_height.copy_(torch.where(supported, height, self.support_height))
        self.have_support |= supported
        self.peak_lift.masked_fill_(supported | ~activity[:, None], 0.)
        self.eligible.copy_(torch.where(supported, activity[:, None], self.eligible))
        return reward
