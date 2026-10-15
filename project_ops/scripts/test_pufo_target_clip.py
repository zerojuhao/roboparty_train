"""Target clipping must not project the physical state or rewrite action history."""

import unittest
from types import SimpleNamespace

import numpy as np

from compare_pufo_rewards import MeasuredSimulation, sim


class TargetClipTests(unittest.TestCase):
    def robot(self, cls=sim.Simulation, enabled=True):
        robot = object.__new__(cls)
        robot.default = np.array([-.3, .3, 0.])
        robot.action_scale = .4
        robot.joint_limits = np.array([[-2.83, .06], [-.06, 2.83], [-np.inf, np.inf]])
        robot.clip_joint_targets = enabled
        robot.data = SimpleNamespace(qpos=np.array([-2.9, 2.9, 0.]), qvel=np.ones(3))
        return robot

    def test_clips_after_scale_offset_and_preserves_state_history(self):
        for cls in (sim.Simulation, MeasuredSimulation):
            robot = self.robot(cls)
            for action in (np.array([-1., 1., 2.]), np.array([1., -1., -2.])):
                before = action.copy()
                qpos, qvel = robot.data.qpos.copy(), robot.data.qvel.copy()
                target = robot.prepare_action_target(action, 8.)
                scaled = np.clip(action * 8., -8., 8.)
                raw = robot.default + .4 * scaled
                np.testing.assert_array_equal(target, np.clip(raw, *robot.joint_limits.T))
                np.testing.assert_array_equal(robot.raw_joint_target, raw)
                np.testing.assert_array_equal(robot.last_action, scaled)
                np.testing.assert_array_equal(action, before)
                np.testing.assert_array_equal(robot.data.qpos, qpos)
                np.testing.assert_array_equal(robot.data.qvel, qvel)

    def test_in_range_is_unchanged_and_ab_switch_restores_legacy(self):
        robot = self.robot()
        action = np.array([.01, -.01, .1])
        np.testing.assert_array_equal(robot.prepare_action_target(action, 8.), robot.default + .4 * action * 8.)
        robot.clip_joint_targets = False
        np.testing.assert_array_equal(robot.prepare_action_target(np.ones(3), 8.), robot.default + 3.2)

    def test_nan_is_not_hidden_as_a_valid_target(self):
        robot = self.robot()
        self.assertTrue(np.isnan(robot.prepare_action_target(np.full(3, np.nan), 8.)).all())


if __name__ == '__main__':
    unittest.main()
