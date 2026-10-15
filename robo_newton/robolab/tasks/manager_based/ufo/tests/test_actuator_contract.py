"""Peak torque-speed limits must not disable braking at overspeed."""

import unittest

import torch

from robolab.assets.robots.roboparty_actuators import RoboPartyActuator


class TestActuatorContract(unittest.TestCase):
    def make_actuator(self, speeds):
        actuator = object.__new__(RoboPartyActuator)
        actuator._joint_vel = torch.tensor([speeds], dtype=torch.float32)
        actuator.effort_limit = torch.full_like(actuator._joint_vel, 100.0)
        actuator.velocity_limit = torch.full_like(actuator._joint_vel, 20.0)
        actuator._keep_velocity = torch.full_like(actuator._joint_vel, 10.0)
        return actuator

    def test_four_quadrant_limits(self):
        actuator = self.make_actuator([-30, -20, -15, 0, 15, 20, 30])
        positive = actuator._clip_effort(torch.full_like(actuator._joint_vel, 200.0))
        negative = actuator._clip_effort(torch.full_like(actuator._joint_vel, -200.0))
        torch.testing.assert_close(positive, torch.tensor([[100., 100., 100., 100., 50., 0., 0.]]))
        torch.testing.assert_close(negative, torch.tensor([[0., 0., -50., -100., -100., -100., -100.]]))

    def test_braking_remains_available_above_no_load_speed(self):
        actuator = self.make_actuator([-200, -30, 30, 200])
        braking = -25.0 * actuator._joint_vel.sign()
        torch.testing.assert_close(actuator._clip_effort(braking), braking)
        torch.testing.assert_close(
            actuator._clip_effort(torch.zeros_like(braking)), torch.zeros_like(braking),
        )


if __name__ == "__main__":
    unittest.main()
