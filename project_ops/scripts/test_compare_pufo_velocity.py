import unittest

import numpy as np

from compare_pufo_velocity import error_metrics


class VelocityMetricsTests(unittest.TestCase):
    def test_stopped_rate_is_conditional_not_all_samples(self):
        truth = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        prediction = np.array([[.3, 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        result = error_metrics(prediction, truth)
        self.assertEqual(result["false_motion_when_stopped_fraction"], .5)
        self.assertEqual(result["stopped_samples"], 2)
        self.assertAlmostEqual(result["mae_m_s"][0], .1)
        self.assertAlmostEqual(result["rmse_m_s"][0], np.sqrt(.03))

    def test_missing_stopped_samples_is_not_reported_as_perfect_accuracy(self):
        result = error_metrics(np.ones((2, 3)), np.ones((2, 3)))
        self.assertIsNone(result["stopped_mae_m_s"])
        self.assertIsNone(result["false_motion_when_stopped_fraction"])

    def test_nonfinite_or_empty_samples_are_rejected(self):
        for prediction in (np.empty((0, 3)), np.array([[np.nan, 0., 0.]])):
            with self.assertRaises(ValueError):
                error_metrics(prediction, prediction)


if __name__ == "__main__":
    unittest.main()
