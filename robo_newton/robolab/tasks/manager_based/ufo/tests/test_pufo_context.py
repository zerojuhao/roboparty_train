import unittest

import torch

from robolab.tasks.manager_based.ufo.mdp.pufo_observations import (
    align_world_vector,
    heading_observation,
    velocity_observation,
)


class TestPufoContext(unittest.TestCase):
    def test_invalid_context_is_zero(self):
        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        valid = torch.tensor([[False], [True]])

        torch.testing.assert_close(
            heading_observation(current, target, valid),
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        )
        torch.testing.assert_close(
            velocity_observation(
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]), current, valid
            ),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )

    def test_alignment_uses_one_fixed_yaw_rotation(self):
        vector = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        reference = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        torch.testing.assert_close(
            align_world_vector(vector, reference, target),
            torch.tensor([[0.0, 1.0], [-2.0, 0.0]]),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_degenerate_valid_heading_is_zero_and_invalid(self):
        current = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        valid = torch.ones(2, 1, dtype=torch.bool)
        torch.testing.assert_close(
            heading_observation(current, target, valid),
            torch.zeros(2, 2),
        )


if __name__ == "__main__":
    unittest.main()
