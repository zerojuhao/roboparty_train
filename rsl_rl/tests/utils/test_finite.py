"""Aggregation, diagnostics and host-sync contracts for finite checks."""

import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from rsl_rl.utils.finite import check_finite_tree, nonfinite_flag
from rsl_rl.utils import check_nan


class TestRolloutFinite(unittest.TestCase):
    def test_nested_observation_diagnostics_and_priority(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                obs = TensorDict({"critic": TensorDict({"velocity": torch.tensor([[value]])}, [1])}, [1])
                with self.assertRaisesRegex(ValueError, "critic.velocity"):
                    check_nan(obs, torch.full((1,), float("nan")), torch.zeros(1, dtype=torch.bool))

    def test_reward_and_done_diagnostics_preserve_nan_only_contract(self):
        obs = TensorDict({"policy": torch.ones(2, 3)}, [2])
        with self.assertRaisesRegex(ValueError, "rewards returned"):
            check_nan(obs, torch.tensor([float("nan"), 1.]), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "dones returned"):
            check_nan(obs, torch.ones(2), torch.tensor([0., float("nan")]))
        check_nan(obs, torch.full((2,), float("inf")), torch.ones(2, dtype=torch.bool))
        check_nan(TensorDict({}, [2]), torch.ones(2), torch.zeros(2, dtype=torch.long))

    def test_literal_and_nested_keys_with_same_diagnostic_path_are_both_checked(self):
        obs = TensorDict({"critic.velocity": torch.full((1, 1), float("nan")),
                          "critic": TensorDict({"velocity": torch.ones(1, 1)}, [1])}, [1])
        with self.assertRaisesRegex(ValueError, "critic.velocity"):
            check_nan(obs, torch.ones(1), torch.zeros(1))

    def test_normal_rollout_has_one_scalar_read(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                obs = TensorDict({str(i): torch.ones(4, 3, device=device) for i in range(16)}, [4])
                rewards = torch.ones(4, device=device)
                dones = torch.zeros(4, device=device, dtype=torch.bool)
                check_nan(obs, rewards, dones)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    check_nan(obs, rewards, dones)
                reads = sum(e.count for e in prof.key_averages() if e.key == "aten::_local_scalar_dense")
                self.assertEqual(reads, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mixed_device_observations_are_supported(self):
        obs = TensorDict({"cpu": torch.ones(2, 3), "gpu": torch.ones(2, 3, device="cuda")}, [2])
        check_nan(obs, torch.ones(2, device="cuda"), torch.zeros(2))
        obs["cpu"][0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "cpu"):
            check_nan(obs, torch.ones(2, device="cuda"), torch.zeros(2))


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
