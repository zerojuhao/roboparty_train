"""Regression checks for the stability report's units and reference alignment."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_pufo_depth_stability import metrics


class StabilityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.tile([0., 0., .8], (21, 1))
        self.targets = np.zeros((20, 24))
        self.joints = np.zeros((21, 24))
        self.quaternions = np.tile([0., 0., 0., 1.], (21, 1))

    def report(self, reference=None, heading=None):
        return metrics(self.positions, self.targets, self.joints,
                       np.zeros(24) if reference is None else reference,
                       self.quaternions, np.array([1., 0.]) if heading is None else heading)

    def test_static_upright_is_not_jitter_or_fall(self):
        result = self.report()
        self.assertEqual(result["late_target_increment_rms_rad"], 0.)
        self.assertEqual(result["late_joint_error_rms_rad"], 0.)
        self.assertEqual(result["max_tilt_degrees"], 0.)
        self.assertEqual(result["max_backward_excursion_m"], 0.)
        self.assertIsNone(result["first_root_below_035m_s"])

    def test_displacement_uses_initial_heading(self):
        self.positions[:, 1] = np.linspace(0., -.2, 21)
        result = self.report(heading=np.array([0., 1.]))
        self.assertAlmostEqual(result["forward_displacement_m"], -.2)
        self.assertAlmostEqual(result["max_backward_excursion_m"], .2)

    def test_tilt_is_independent_of_yaw(self):
        self.quaternions[:] = Rotation.from_euler("xyz", [20., 0., 90.], degrees=True).as_quat()
        result = self.report()
        self.assertAlmostEqual(result["max_tilt_degrees"], 20.)
        self.assertAlmostEqual(result["late_tilt_rms_degrees"], 20.)

    def test_target_increment_is_radians_per_control_step(self):
        self.targets[:] = np.arange(20)[:, None] * .01
        self.assertAlmostEqual(self.report()["late_target_increment_rms_rad"], .01)

    def test_follow_reference_aligns_late_state_including_final_sample(self):
        self.joints[:] = np.arange(21)[:, None] * .01
        result = self.report(reference=self.joints[10:] - .025)
        self.assertAlmostEqual(result["late_joint_error_rms_rad"], .025)

    def test_early_motion_is_excluded_from_late_metrics(self):
        self.targets[:9] = 1.
        self.joints[:10] = 1.
        result = self.report()
        self.assertEqual(result["late_target_increment_rms_rad"], 0.)
        self.assertEqual(result["late_joint_error_rms_rad"], 0.)

    def test_height_threshold_time_uses_50_hz(self):
        self.positions[7:, 2] = .3
        self.assertAlmostEqual(self.report()["first_root_below_035m_s"], .14)


if __name__ == "__main__":
    unittest.main()
