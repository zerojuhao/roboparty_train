import unittest

import test_pufo_stair_push as push
from test_compare_pufo_rewards import TestRewardComparison


class TestStairPush(unittest.TestCase):
    def test_push_ramp_and_release(self):
        for t, expected in [(0,0), (3,0), (3.375,40), (3.75,80), (9,80), (11.625,40), (12,0), (16,0)]:
            self.assertAlmostEqual(push.push_force(t,80), expected)

    def test_hold_masks_commands(self):
        class Robot:
            def observe(self, heading, velocity, heading_valid, velocity_valid):
                self.heading_valid, self.velocity_valid = heading_valid, velocity_valid
                return heading, velocity
        robot = Robot()
        push.hold_observation(robot)
        self.assertFalse(robot.heading_valid.any())
        self.assertFalse(robot.velocity_valid.any())

    def test_geometry_exactly_15cm_with_30cm_treads(self):
        model = push.stair_model()
        for i in range(6):
            geom = model.geom(f'push_stair_{i+1}')
            self.assertAlmostEqual(geom.pos[2]+geom.size[2], .15*(i+1))
            self.assertAlmostEqual(geom.pos[0]-geom.size[0], .3*i)

    def test_applied_force_and_measurement_preserve_physics(self):
        helper = TestRewardComparison()
        baseline, measured = helper.make_robot(False), helper.make_robot(True)
        for robot in (baseline, measured):
            robot.data.xfrc_applied[1, 2] = 3.
        for _ in range(20):
            baseline.advance(push.np.zeros(2), 8.)
            measured.advance(push.np.zeros(2), 8.)
            measured.feet_state()
            push.np.testing.assert_array_equal(baseline.data.qpos, measured.data.qpos)
            push.np.testing.assert_array_equal(baseline.data.qvel, measured.data.qvel)


if __name__ == '__main__':
    unittest.main()
