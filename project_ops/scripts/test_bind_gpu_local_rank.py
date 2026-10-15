"""Startup placement selection without changing the test process's affinity."""

import unittest
from unittest.mock import patch

import bind_gpu_local_rank as binding


class TestRankPlacement(unittest.TestCase):
    def test_visible_mapping_and_existing_cpu_restrictions(self):
        with patch.object(binding.subprocess, "check_output", return_value="00000000:49:00.0\n") as query, \
                patch.object(binding.Path, "read_text", return_value="16-31,80-95\n") as read, \
                patch.object(binding.os, "sched_getaffinity", return_value={0, 18, 19, 85}):
            result = binding.rank_placement({"LOCAL_RANK": "0", "CUDA_VISIBLE_DEVICES": "2,0"})
        self.assertEqual(result["gpu"], "2")
        self.assertEqual(result["pci"], "0000:49:00.0")
        self.assertEqual(result["cpus"], [18, 19, 85])
        self.assertEqual(result["original_cpus"], [0, 18, 19, 85])
        self.assertEqual(query.call_args.args[0][2], "2")
        read.assert_called_once()

    def test_uuid_selection(self):
        with patch.object(binding.subprocess, "check_output", return_value="0000:16:00.0\n") as query, \
                patch.object(binding.Path, "read_text", return_value="0-3\n"), \
                patch.object(binding.os, "sched_getaffinity", return_value={0, 1}):
            binding.rank_placement({"LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "3,GPU-test"})
        self.assertEqual(query.call_args.args[0][2], "GPU-test")

    def test_disjoint_affinity_fails_instead_of_expanding_permissions(self):
        with patch.object(binding.subprocess, "check_output", return_value="0000:16:00.0\n"), \
                patch.object(binding.Path, "read_text", return_value="0-3\n"), \
                patch.object(binding.os, "sched_getaffinity", return_value={4}), \
                self.assertRaisesRegex(ValueError, "no allowed CPUs"):
            binding.rank_placement({"LOCAL_RANK": "0", "CUDA_VISIBLE_DEVICES": "0"})

    def test_invalid_rank_or_visibility_fails_before_query(self):
        for env in ({"LOCAL_RANK": "-1", "CUDA_VISIBLE_DEVICES": "0"},
                    {"LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "0"}, {"LOCAL_RANK": "0"}):
            with self.subTest(env=env), patch.object(binding.subprocess, "check_output") as query, \
                    self.assertRaises(ValueError):
                binding.rank_placement(env)
            query.assert_not_called()

    def test_malformed_pci_response_fails(self):
        for response in ("invalid\n", "0000:16:00.0\n0000:38:00.0\n"):
            with self.subTest(response=response), \
                    patch.object(binding.subprocess, "check_output", return_value=response), \
                    self.assertRaises(ValueError):
                binding.rank_placement({"LOCAL_RANK": "0", "CUDA_VISIBLE_DEVICES": "0"})

    def test_dry_run_does_not_bind_or_exec(self):
        with patch.object(binding, "rank_placement", return_value={"cpus": [1]}), \
                patch.object(binding.os, "sched_setaffinity") as bind, \
                patch.object(binding.os, "execv") as execute:
            binding.main(["--dry-run", "train.py", "--task", "RP1-PUFO"])
        bind.assert_not_called()
        execute.assert_not_called()

    def test_binding_precedes_exec_and_preserves_training_arguments(self):
        events = []
        with patch.object(binding, "rank_placement", return_value={"cpus": [1, 2]}), \
                patch.object(binding.os, "sched_setaffinity", side_effect=lambda pid, cpus: events.append("bind")), \
                patch.object(binding.os, "execv", side_effect=lambda executable, argv: events.append(argv)):
            binding.main(["train.py", "--resume", "agent.profile_update_rank=-1"])
        self.assertEqual(events, ["bind", [binding.sys.executable, "train.py", "--resume",
                                          "agent.profile_update_rank=-1"]])


if __name__ == "__main__":
    unittest.main()
