"""Graph diagnostics must retain the first failure and ignore unused capacity."""

import unittest

import torch
import warp as wp

from robolab.utils.physics_nan import check_worlds, check_constraints, check_contacts, remember_input, check_bodies
from robolab.utils.physics_nan import check_active, freeze_failed, mark_frozen


class TestPhysicsNaNProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_first_stage_and_last_valid_input_are_retained(self):
        values = wp.from_torch(torch.tensor([[1., 2.], [3., 4.]]))
        saved = wp.zeros_like(values)
        flags = wp.zeros(2, dtype=int, device="cpu")
        wp.launch(remember_input, values.shape, inputs=[values, saved, flags], device="cpu")
        wp.to_torch(values)[1, 0] = torch.nan
        for code in (7, 9):
            wp.launch(check_worlds, values.shape, inputs=[values, flags, code], device="cpu")
        wp.launch(remember_input, values.shape, inputs=[values, saved, flags], device="cpu")
        self.assertEqual(flags.numpy().tolist(), [0, 7])
        self.assertEqual(saved.numpy().tolist(), [[1., 2.], [3., 4.]])

    def test_unused_constraint_slots_are_ignored(self):
        values = wp.from_torch(torch.tensor([[1., float("nan")], [float("inf"), 0.]]))
        counts = wp.array([1, 1], dtype=int, device="cpu")
        flags = wp.zeros(2, dtype=int, device="cpu")
        wp.launch(check_constraints, values.shape, inputs=[values, counts, flags, 5], device="cpu")
        self.assertEqual(flags.numpy().tolist(), [0, 5])

    def test_pooled_contacts_map_back_to_world_and_ignore_unused_slots(self):
        values = wp.from_torch(torch.tensor([[1.], [float("nan")], [float("nan")]]))
        worlds = wp.array([0, 2, 1], dtype=int, device="cpu")
        count = wp.array([2], dtype=int, device="cpu")
        flags = wp.zeros(3, dtype=int, device="cpu")
        wp.launch(check_contacts, values.shape, inputs=[values, worlds, count, flags, 4], device="cpu")
        self.assertEqual(flags.numpy().tolist(), [0, 0, 4])

    def test_newton_bodies_ignore_global_entities_and_map_to_worlds(self):
        values = wp.from_torch(torch.tensor([[float("nan")], [1.], [float("nan")], [float("inf")]]))
        worlds = wp.array([-1, 0, 2, 1], dtype=int, device="cpu")
        flags = wp.zeros(3, dtype=int, device="cpu")
        wp.launch(check_bodies, values.shape, inputs=[values, worlds, flags, 8], device="cpu")
        self.assertEqual(flags.numpy().tolist(), [0, 8, 8])

    def test_cholesky_ignores_inactive_worlds_and_freezes_first_failed_context(self):
        values = wp.from_torch(torch.tensor([[float("nan")], [float("nan")], [1.]]))
        done = wp.array([True, False, False], dtype=bool, device="cpu")
        flags = wp.zeros(3, dtype=int, device="cpu")
        frozen = wp.zeros_like(flags)
        saved = wp.zeros_like(values)
        wp.launch(check_active, values.shape, inputs=[values, done, flags, 9], device="cpu")
        wp.to_torch(values)[:] = torch.tensor([[3.], [4.], [5.]])
        wp.launch(freeze_failed, values.shape, inputs=[values, saved, flags, frozen], device="cpu")
        wp.launch(mark_frozen, 3, inputs=[flags, frozen], device="cpu")
        wp.to_torch(values)[:] = 42.
        wp.launch(freeze_failed, values.shape, inputs=[values, saved, flags, frozen], device="cpu")
        self.assertEqual(flags.numpy().tolist(), [0, 9, 0])
        self.assertEqual(saved.numpy().tolist(), [[0.], [4.], [0.]])
