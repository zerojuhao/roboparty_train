"""Command tracking and force-driven XY velocity targets at zero command."""

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def direct_tanh_target(magnitude, bounds, k):
    """Dead zone followed by normalized tanh in [0, 1]."""
    lo, hi = bounds
    if not all(math.isfinite(v) for v in (lo, hi, k)) or not 0 <= lo < hi or k <= 0:
        raise ValueError('Direct tanh requires finite increasing nonnegative bounds and positive k')
    x = ((magnitude - lo) / (hi - lo)).clamp(0., 1.)
    return torch.tanh(k * x) / math.tanh(k)


def load_presence(ratio, threshold):
    """Smoothstep presence in [0, 1]."""
    fraction = (ratio.abs() / max(threshold, 1e-6)).clamp(0., 1.)
    return fraction.square() * (3. - 2. * fraction)


def force_follow_gate(magnitude, body_weight, deadband_bw, scale_bw, force_range_n=None):
    if force_range_n is None:
        return load_presence((magnitude / body_weight.clamp_min(1e-6) - deadband_bw).clamp_min(0.), scale_bw)
    lo, hi = force_range_n
    if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
        raise ValueError('force_range_n must be finite, nonnegative and increasing')
    return load_presence((magnitude - lo).clamp_min(0.), hi - lo)


class body_compliance(ManagerTermBase):
    """Track XY velocity commands, or sustained horizontal external pulls at zero command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        for name in ('std', 'target_accel', 'force_scale_bw', 'force_filter_time_s', 'target_tanh_k'):
            if not math.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if len(p['max_speed']) != 3 or not all(math.isfinite(v) and v > 0 for v in p['max_speed']):
            raise ValueError('max_speed must contain positive forward, backward and lateral speeds')
        for name in ('force_deadband_bw', 'target_sustain_s'):
            if not math.isfinite(p.get(name, 0.)) or p.get(name, 0.) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        for key in ('force_range_n', 'target_force_range_n'):
            lo, hi = p[key]
            if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
                raise ValueError(f'{key} must be finite, nonnegative and increasing')

        n, device = env.num_envs, env.device
        self.filtered_force_w = torch.zeros(n, 3, device=device)
        self.filtered_force_magnitude = torch.zeros(n, device=device)
        self.translation_gate = torch.zeros(n, device=device)
        self.force_gate = torch.zeros(n, device=device)
        self.force_target_velocity_b = torch.zeros(n, 2, device=device)
        self.target_velocity_b = torch.zeros(n, 2, device=device)
        self.target_load_age = torch.zeros(n, device=device)
        self._force_error_sum = torch.zeros(n, device=device)
        self._force_count = torch.zeros(n, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        samples = self._force_count[ids].sum()
        err = self._force_error_sum[ids].sum() / samples.clamp_min(1.)
        log = self._env.extras.setdefault('log', {})
        log['Metrics/force_target/error_vel_xy'] = err
        log['Interaction/force_target_error_xy_mps'] = err
        for buf in (self.filtered_force_w, self.filtered_force_magnitude, self.translation_gate,
                    self.force_gate, self.force_target_velocity_b, self.target_velocity_b,
                    self.target_load_age, self._force_error_sum, self._force_count):
            buf[ids] = 0

    def __call__(self, env, command_name, std, max_speed, target_accel, force_deadband_bw, force_scale_bw,
                 force_filter_time_s, force_range_n, target_force_range_n, target_tanh_k=2.0,
                 target_sustain_s=0.3):
        robot = env.scene['robot']
        push = env.event_manager.get_term_cfg('sustained_standing_push').func
        force_w = push.applied_force.sum(dim=1)
        alpha = -math.expm1(-env.step_dt / force_filter_time_s)
        self.filtered_force_w.lerp_(force_w, alpha)
        self.filtered_force_magnitude.lerp_(push.applied_force.norm(dim=-1).sum(-1), alpha)

        heading = yaw_quat(robot.data.root_quat_w.torch)
        force_b = quat_apply_inverse(heading, self.filtered_force_w)[:, :2]
        magnitude = torch.minimum(force_b.norm(dim=-1), force_w[:, :2].norm(dim=-1))
        bw = robot.data.body_mass.torch.sum(dim=1) * abs(env.cfg.sim.gravity[2])
        presence = force_follow_gate(magnitude, bw, force_deadband_bw, force_scale_bw, force_range_n)
        self.translation_gate.copy_(presence)
        self.force_gate.copy_(presence)

        command = env.command_manager.get_command(command_name)
        zero_command = (command == 0).all(dim=1)
        velocity = quat_apply_inverse(heading, robot.data.root_lin_vel_w.torch)[:, :2]
        direction = force_b / force_b.norm(dim=-1).clamp_min(1e-6)[:, None]
        # Fastest follow speed along the pull: an ellipse matching the commanded walking range.
        forward, backward, lateral = max_speed
        along = direction[:, 0] / torch.where(direction[:, 0] >= 0, forward, backward)
        limit = (along.square() + (direction[:, 1] / lateral).square()).clamp_min(1e-6).rsqrt()

        target_speed = direct_tanh_target(magnitude, target_force_range_n, target_tanh_k) * limit
        if target_sustain_s > 0:
            above = magnitude >= target_force_range_n[0]
            self.target_load_age.copy_(torch.where(
                above, self.target_load_age + env.step_dt, torch.zeros_like(self.target_load_age)))
            target_speed = target_speed * (self.target_load_age >= target_sustain_s - 1e-6).to(target_speed.dtype)

        # The follow target speeds up and slows down at a bounded rate instead of jumping.
        change = direction * target_speed[:, None] - self.force_target_velocity_b
        self.force_target_velocity_b.add_(change * (
            target_accel * env.step_dt / change.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1.))
        self.target_velocity_b.copy_(torch.where(zero_command[:, None], self.force_target_velocity_b, command[:, :2]))

        error_sq = (velocity - self.target_velocity_b).square().sum(dim=-1)
        mask = zero_command & (presence > 0.5)
        self._force_error_sum.add_(error_sq.sqrt() * mask)
        self._force_count.add_(mask)
        return torch.exp(-error_sq / std**2)
