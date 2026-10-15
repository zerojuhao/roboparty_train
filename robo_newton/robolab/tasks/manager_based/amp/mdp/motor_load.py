"""Control-step motor peaks sampled before every physics integration."""

import torch


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


class MotorLoadTracker:
    def __init__(self, num_envs, num_joints, device):
        self.torque_ratio_peak = torch.zeros(num_envs, num_joints, device=device)
        self.torque_delta_ratio_peak = torch.zeros(num_envs, device=device)
        self.braking_power_peak = torch.zeros(num_envs, device=device)
        self._previous_torque = torch.zeros_like(self.torque_ratio_peak)
        self._previous_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def begin_step(self):
        self.torque_ratio_peak.zero_()
        self.torque_delta_ratio_peak.zero_()
        self.braking_power_peak.zero_()

    def reset(self, env_ids):
        self.torque_ratio_peak[env_ids] = 0
        self.torque_delta_ratio_peak[env_ids] = 0
        self.braking_power_peak[env_ids] = 0
        self._previous_torque[env_ids] = 0
        self._previous_valid[env_ids] = False

    def sample(self, data):
        torque = _as_torch(data.applied_torque)
        velocity = _as_torch(data.joint_vel)
        limits = _as_torch(data.joint_effort_limits).clamp_min(1.0e-6)
        self.torque_ratio_peak.copy_(torch.maximum(self.torque_ratio_peak, torque.abs() / limits))
        braking_power = torch.relu(-torque * velocity).sum(dim=1)
        self.braking_power_peak.copy_(torch.maximum(self.braking_power_peak, braking_power))
        delta = ((torque - self._previous_torque) / limits).abs().amax(dim=1)
        delta = torch.where(self._previous_valid, delta, 0.0)
        self.torque_delta_ratio_peak.copy_(torch.maximum(self.torque_delta_ratio_peak, delta))
        self._previous_torque.copy_(torque)
        self._previous_valid.fill_(True)
