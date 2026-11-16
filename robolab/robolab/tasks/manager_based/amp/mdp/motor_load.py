"""Control-step motor peaks sampled before every physics integration."""

import torch


class MotorLoadTracker:
    def __init__(self, num_envs, num_joints, device):
        self.torque_ratio_peak = torch.zeros(num_envs, num_joints, device=device)

    def begin_step(self):
        self.torque_ratio_peak.zero_()

    def reset(self, env_ids):
        self.torque_ratio_peak[env_ids] = 0

    def sample(self, data):
        torque = data.applied_torque.torch
        limits = data.joint_effort_limits.torch.clamp_min(1.0e-6)
        self.torque_ratio_peak.copy_(torch.maximum(self.torque_ratio_peak, torque.abs() / limits))
