"""Regression tests for deterministic virtual edge segmentation."""

import unittest
from types import SimpleNamespace

import numpy as np

from robolab.terrains.virtual_obstacle.edge_cylinder import GreedyconcatEdgeCylinder


class TestGreedyEdgeCylinder(unittest.TestCase):
    @staticmethod
    def make_detector(point_distance_threshold=0.06):
        cfg = SimpleNamespace(
            angle_threshold=70.0,
            adjacent_angle_threshold=30.0,
            point_distance_threshold=point_distance_threshold,
            min_points=3,
        )
        return GreedyconcatEdgeCylinder(cfg)

    @staticmethod
    def edge_chain():
        return np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 2.0, 0.01, 0.0],
                [2.0, 0.01, 0.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_segmentation_is_independent_of_edge_input_order(self):
        detector = self.make_detector()
        edges = self.edge_chain()
        expected = detector.process_edges(edges)

        for order in ([2, 0, 1], [1, 2, 0], [0, 2, 1]):
            np.testing.assert_array_equal(expected, detector.process_edges(edges[list(order)]))

    def test_linearity_threshold_comes_from_configuration(self):
        edges = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 2.0, 0.055, 0.0],
                [2.0, 0.055, 0.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        merged = self.make_detector(0.06).process_edges(edges)
        split = self.make_detector(0.05).process_edges(edges)

        self.assertEqual(merged.shape, (1, 6))
        self.assertEqual(split.shape, (1, 6))
        np.testing.assert_allclose(merged[0, 3:], [3.0, 0.0, 0.0])
        np.testing.assert_allclose(split[0, 3:], [2.0, 0.055, 0.0])


if __name__ == "__main__":
    unittest.main()
