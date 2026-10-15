"""Aggregation, diagnostics and host-sync contracts for finite checks."""

import unittest
from unittest.mock import patch

import torch

from rsl_rl.utils.finite import check_finite_tree, nonfinite_flag


class TestFinite(unittest.TestCase):
    def test_empty_integer_and_nested_trees(self):
        for value in ({}, [], torch.empty(0), torch.ones(3, dtype=torch.uint8), None):
            check_finite_tree(value, "tree")
        check_finite_tree({"a": [torch.ones(2), (torch.tensor(3.),)]}, "tree")

    def test_nan_inf_scalar_complex_and_noncontiguous_diagnostics(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError, r"tree.a\[1\].bad"):
                    check_finite_tree({"a": [torch.ones(2), {"bad": torch.tensor(value)}]}, "tree")
        with self.assertRaisesRegex(FloatingPointError, "complex"):
            check_finite_tree(torch.tensor(complex(1, float("nan"))), "complex")
        tensor = torch.ones(3, 4).t()
        tensor[2, 1] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, r"bad_rows=\[2\], bad_cols=\[1\]"):
            check_finite_tree(tensor, "strided")

    def test_fast_path_does_not_run_diagnostics(self):
        with patch("rsl_rl.utils.finite.raise_nonfinite_details") as diagnostic:
            check_finite_tree({str(i): torch.ones(4) for i in range(50)}, "tree")
        diagnostic.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_flag_has_no_host_reads_and_check_has_one(self):
        tree = {str(i): torch.ones(1024, device="cuda") for i in range(24)}
        check_finite_tree(tree, "warmup")
        for fn, expected in ((nonfinite_flag, 0), (lambda x: check_finite_tree(x, "tree"), 1)):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                fn(tree)
            reads = sum(e.count for e in prof.key_averages() if e.key == "aten::_local_scalar_dense")
            self.assertEqual(reads, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mixed_devices_and_nonfinite_cuda_tensor(self):
        tree = {"cpu": torch.ones(3), "gpu": torch.ones(3, device="cuda")}
        check_finite_tree(tree, "tree")
        tree["gpu"][1] = float("nan")
        flag = nonfinite_flag(tree, device="cuda")
        self.assertEqual(flag.device.type, "cuda")
        self.assertEqual(flag.item(), 1)
        with self.assertRaisesRegex(FloatingPointError, "tree.gpu"):
            check_finite_tree(tree, "tree")


if __name__ == "__main__":
    unittest.main()
