"""Captured EPA slivers and ordinary signed affine coordinates, on CPU/CUDA."""

import os
import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import collision_gjk
from robolab.utils import mjwarp_contact_guard as guard


stock_affine = collision_gjk._tri_affine_coord


@wp.kernel
def evaluate(vertices: wp.array2d[wp.vec3], output: wp.array2d[wp.vec3]):
    i = wp.tid()
    a, b, c, p = vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[i, 3]
    output[i, 0] = stock_affine(a, b, c, p)
    output[i, 1] = guard._guarded_affine(a, b, c, p)


@wp.kernel
def attach(pt: collision_gjk.Polytope, idx: int, output: wp.array[float]):
    output[0] = guard._guarded_attach(pt, idx, 0, 1, 2)


class TestContactGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("PUFO_GUARD_TEST_DEVICE", "cpu")
        guard.install()

    @classmethod
    def tearDownClass(cls):
        guard.restore()

    def weights(self, cases):
        vertices = wp.array(np.asarray(cases, dtype=np.float32), dtype=wp.vec3, device=self.device)
        output = wp.empty((len(cases), 2), dtype=wp.vec3, device=self.device)
        wp.launch(evaluate, dim=len(cases), inputs=[vertices, output], device=self.device)
        return output.numpy()

    def attach_triangle(self, vertices, exhausted=False):
        pt = collision_gjk.Polytope()
        support = np.zeros((6, 3), dtype=np.float32)
        support[::2] = vertices
        pt.vert = wp.array(support, dtype=wp.vec3, device=self.device)
        pt.face = wp.zeros(1, dtype=int, device=self.device)
        pt.face_pr = wp.zeros(1, dtype=wp.vec3, device=self.device)
        pt.face_norm2 = wp.zeros(1, dtype=float, device=self.device)
        pt.nface = int(exhausted)
        output = wp.zeros(1, dtype=float, device=self.device)
        wp.launch(attach, dim=1, inputs=[pt, int(exhausted), output], device=self.device)
        return int(pt.face.numpy()[0]), float(output.numpy()[0]), pt.face_pr.numpy()[0]

    def test_epa_sliver_is_excluded_from_selection_but_kept_for_horizon(self):
        face, distance, projection = self.attach_triangle([
            [-.0080909729, -.526614726, -.000185370445],
            [-.0080909729, -.526614606, -2.18153e-5],
            [-.0080909729, -.526614606, .106870413],
        ])
        self.assertTrue(face & 0x40000000)
        self.assertFalse(face & 0x80000000)
        self.assertGreater(distance, 0)
        self.assertTrue(np.isfinite(projection).all())

    def test_regular_face_remains_selectable(self):
        face, distance, _ = self.attach_triangle([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
        self.assertEqual(face, (1 << 10) + (2 << 20))
        self.assertAlmostEqual(distance, 1.)

    def test_full_face_storage_never_reads_or_writes_past_capacity(self):
        face, distance, _ = self.attach_triangle([[0, 0, 1], [1, 0, 1], [0, 1, 1]], exhausted=True)
        self.assertEqual((face, distance), (0, 0.))

    def test_captured_slivers_remain_convex_and_finite(self):
        cases = [
            [[.209621429, -.372461319, -1.26361847e-5],
             [.209623337, -.372461319, -1.26361847e-5],
             [-.390407562, -.372463226, -1.26361847e-5], [0, 0, -1.26361847e-5]],
            [[-.0080909729, -.526614726, -.000185370445],
             [-.0080909729, -.526614606, -2.18153e-5],
             [-.0080909729, -.526614606, .106870413], [-.00809097197, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]],
        ]
        # The newly captured case bypassed the old squared-area threshold.
        a, b, c, _ = np.asarray(cases[1], dtype=np.float32)
        scale = max(np.dot(b-a, b-a), np.dot(c-a, c-a), np.dot(c-b, c-b))
        self.assertGreater(np.dot(np.cross(b-a, c-a), np.cross(b-a, c-a)) / scale**2, 1e-12)
        values = self.weights(cases)[:, 1]
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all() and (values <= 1).all())
        np.testing.assert_allclose(values.sum(-1), 1, atol=1e-6)

    def test_normal_faces_preserve_inside_and_signed_outside_coordinates(self):
        cases = [[[0, 0, 0], [1, 0, 0], [0, 1, 0], p]
                 for p in ([.2, .3, 0], [2, 2, 0], [-.5, .5, 0])]
        values = self.weights(cases)
        np.testing.assert_array_equal(values[:, 0], values[:, 1])
        self.assertLess(values[1, 1].min(), 0)

    def test_well_conditioned_random_faces_are_unchanged(self):
        rng = np.random.default_rng(42)
        cases = []
        while len(cases) < 1000:
            a, b, c = rng.uniform(-1, 1, (3, 3))
            if np.linalg.norm(np.cross(b-a, c-a)) < .1:
                continue
            weights = rng.dirichlet([1, 1, 1])
            p = weights @ np.stack((a, b, c))
            cases.append([a, b, c, p])
        values = self.weights(cases)
        np.testing.assert_array_equal(values[:, 0], values[:, 1])


if __name__ == "__main__":
    unittest.main()
