import unittest

import compare_pufo_rewards as comparison

np, sim = comparison.np, comparison.sim


class TestRewardComparison(unittest.TestCase):
    def make_robot(self, measured):
        xml = '''<mujoco><option timestep="0.0025"/><worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body name="left_ankle_roll_link" pos="0 0 .1">
            <joint name="left" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
          <body name="right_ankle_roll_link" pos="1 0 .1">
            <joint name="right" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body></worldbody><actuator>
            <motor joint="left"/><motor joint="right"/>
          </actuator></mujoco>'''
        cls = comparison.MeasuredSimulation if measured else sim.Simulation
        robot = cls.__new__(cls)
        robot.model = sim.mujoco.MjModel.from_xml_string(xml)
        robot.data = sim.mujoco.MjData(robot.model)
        robot.qids = robot.vids = robot.aids = np.arange(2)
        robot.default = np.zeros(2)
        robot.action_scale = .4
        robot.clip_joint_targets = True
        robot.joint_limits = np.tile([-np.inf, np.inf], (2, 1))
        robot.kp = np.full(2, 120.)
        robot.kd = np.full(2, 9.)
        robot.limit = np.full(2, 100.)
        robot.speed, robot.keep = np.full(2, 20.), np.full(2, 10.)
        robot.feet = [1, 2]
        robot.pd_samples = []
        sim.mujoco.mj_forward(robot.model, robot.data)
        return robot

    def test_measurement_does_not_change_physics(self):
        baseline, measured = self.make_robot(False), self.make_robot(True)
        for i in range(40):
            action = np.array([np.sin(i), np.cos(i)]) * .01
            baseline.advance(action, 8.)
            measured.advance(action, 8.)
            measured.feet_state()
            np.testing.assert_array_equal(measured.data.qpos, baseline.data.qpos)
            np.testing.assert_array_equal(measured.data.qvel, baseline.data.qvel)
        self.assertEqual(len(measured.pd_samples), 160)

    def test_contact_force_is_upward_and_velocity_is_zero_when_settled(self):
        robot = self.make_robot(True)
        for _ in range(100):
            robot.advance(np.zeros(2), 8.)
        velocity, force = robot.feet_state()
        self.assertTrue((force > 9).all())
        np.testing.assert_allclose(velocity, 0., atol=1e-6)

    def test_future_window_and_norm(self):
        z = sim.torch.tensor([[3., 4.], [0., 5.], [-3., 4.], [5., 0.]])
        result = sim.weighted_context_window(z, 1, .8, 3).numpy()
        expected = np.array([0., 5.]) + .8 * np.array([-3., 4.]) + .64 * np.array([5., 0.])
        expected *= 5 / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        np.testing.assert_allclose(sim.weighted_context_window(z, 3, .8, 5), z[3], atol=1e-6)

    def test_empty_common_interval_is_not_reported_as_improvement(self):
        self.assertIsNone(comparison.summarize({}, 2., 1.))

    def test_metric_units_and_squared_reward_terms(self):
        trace = dict(
            pd=np.stack([np.full((40, 2), 2.), np.full((40, 2), 3.)], axis=1),
            foot_v=np.zeros((11, 2)), foot_force=np.ones((11, 2)) * 10,
            joint_pos=np.ones((11, 2)) * .1, reference=np.zeros((11, 2)),
            targets=np.zeros((10, 2)),
        )
        result = comparison.summarize(trace, 0, .2)
        self.assertEqual(result['joint_speed_rms_rad_s'], 2.)
        self.assertEqual(result['joint_vel_l2_mean'], 8.)
        self.assertEqual(result['joint_acc_l2_mean'], 0.)
        self.assertEqual(result['torque_rms_nm'], 3.)
        self.assertEqual(result['abs_mechanical_power_mean_w'], 12.)
        self.assertEqual(result['sound_suppression_proxy_mean'], 0.)
        self.assertAlmostEqual(result['joint_tracking_rmse_rad'], .1)


if __name__ == '__main__':
    unittest.main()
