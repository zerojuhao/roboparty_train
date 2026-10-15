"""Reward logging and isolated evaluation contracts without a simulator."""

import os
import json
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rsl_rl.runners.fbcpr_runner import FbCprRunner, _joint_pos_metrics, _joint_state_observation, _distance_proximity
from rsl_rl.runners.tracking_evaluation import (
    PersistentTrackingEvaluator, isolated_evaluator_device, merge_tracking_shards, motion_shard,
)
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent


def _finite_worker(rank, rendezvous, world_size=2, backend="gloo"):
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend, init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
    try:
        runner = object.__new__(FbCprRunner)
        runner.device, runner.is_distributed, runner.gpu_global_rank = device, True, rank
        try:
            runner._check_finite_rollout(torch.tensor([float("nan") if rank == 1 else 1.], device=device), "test")
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Every rank must fail before a gradient collective")
        runner.log_dir = None
        runner.env_iterations = 0
        runner.replay = {"train": ReplayBuffer(4, 1)}
        def step(action):
            if rank == 1:
                raise RuntimeError("MuJoCo Warp capacity overflow: worlds=[3] flags=[256]")
            return {"state": torch.zeros(1, device=device)}, torch.zeros(1), torch.zeros(1), {}
        runner.env = SimpleNamespace(step=step)
        try:
            runner._step_training_environment(torch.zeros(1), {})
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Every rank must receive the physics capacity failure")
        for exception in (RuntimeError("PUFO root-ground ray missed terrain"),
                          ValueError("injected observation shape error")):
            def failed_sensor_step(action):
                if rank == 1:
                    raise exception
                return {"state": torch.zeros(1, device=device)}, torch.zeros(1), torch.zeros(1), {}
            runner.env = SimpleNamespace(step=failed_sensor_step)
            try:
                runner._step_training_environment(torch.zeros(1), {})
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Every rank must receive the environment exception")
    finally:
        torch.distributed.destroy_process_group()


def _shard_worker(rank, rendezvous):
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=8, timeout=timedelta(seconds=40))
    try:
        runner = object.__new__(FbCprRunner)
        runner.is_distributed, runner.is_main_process = True, rank == 0
        runner.gpu_global_rank = rank
        runner._eval_group = None
        runner._tracking_evaluator = None
        runner.writer = Mock()
        averaged = runner._average_metrics({"train_loss": torch.tensor(float(rank)),
                                           "Reward/raw/action_rate": torch.tensor(float(rank + 10))})
        assert averaged["train_loss"].item() == 3.5
        assert averaged["Reward/raw/action_rate"].item() == 13.5
        keys = [str(i) for i in range(9)]
        runner.cfg = SimpleNamespace(eval_num_motions=9)
        runner._mgr_env = SimpleNamespace(command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(motion_lib=SimpleNamespace(_motion_data_keys=keys))))
        runner._finalize_tracking_eval = lambda step, metrics, clips, count: metrics

        def shard(step, ids):
            if step == 2 and rank == 3:
                raise RuntimeError("injected shard failure")
            return {"motion_keys": keys, "motion_ids": ids,
                    "sums": {"emd": float(sum(ids))}, "counts": {"emd": len(ids)},
                    "clips": {i: {"emd": float(i)} for i in ids}}

        runner._run_eval_shard = shard
        assert runner._run_tracking_eval(0)["emd"] == 4.
        runner.cfg.eval_num_motions = 3
        assert runner._run_tracking_eval(1)["emd"] == 1.
        runner.cfg.eval_num_motions = 9
        try:
            runner._run_tracking_eval(2)
        except RuntimeError as exc:
            assert "injected shard failure" in str(exc)
        else:
            raise AssertionError("All ranks must receive the evaluation failure")
    finally:
        torch.distributed.destroy_process_group()


class TestFbCprEvaluation(unittest.TestCase):
    def test_autoreset_requires_final_observations_and_valid_environment_mask(self):
        runner = object.__new__(FbCprRunner)
        runner.device, runner.is_distributed = "cpu", False
        runner.gpu_global_rank, runner.env_iterations, runner.log_dir = 0, 0, None
        runner.replay = {"train": ReplayBuffer(8, 2, end_key="episode_end", transition_outcomes=True)}
        for infos in ({}, {"final_obs": {"state": torch.ones(2, 1)}, "_final_obs": torch.tensor([False, False])}):
            runner.env = SimpleNamespace(step=lambda action: ({"state": torch.zeros(2, 1)},
                                                               torch.zeros(2), torch.tensor([True, False]), infos))
            with self.assertRaises(FloatingPointError):
                runner._step_training_environment(torch.zeros(2, 1), {})

    def test_profile_runs_update_once_and_preserves_metrics(self):
        runner = object.__new__(FbCprRunner)
        runner.device = "cpu"
        runner.gpu_global_rank = 0
        runner.replay = {"train": object()}
        expected = {"loss": torch.tensor(4.)}

        def update(replay, **kwargs):
            self.assertIs(replay, runner.replay)
            self.assertEqual(kwargs, {"depth_gradient_diagnostics": False})
            torch.ones(2).sum()
            return expected

        runner.agent = SimpleNamespace(update=Mock(side_effect=update))
        with tempfile.TemporaryDirectory() as directory:
            runner.log_dir = directory
            result = runner._profile_agent_update(123, depth_gradient_diagnostics=False)
            self.assertIs(result, expected)
            runner.agent.update.assert_called_once()
            trace = json.loads((Path(directory) / "update_profile_rank0_123.json").read_text())
            self.assertTrue(any(event["name"] == "fbcpr/minibatch_update"
                                for event in trace["traceEvents"]))
            self.assertTrue((Path(directory) / "update_profile_rank0_123.txt").is_file())

    def test_profile_startup_barrier_is_only_for_all_rank_capture(self):
        runner = object.__new__(FbCprRunner)
        runner.device, runner.gpu_global_rank = "cpu", 0
        runner.replay = {"train": object()}
        for distributed in (False, True):
            for synchronize in (False, True):
                with self.subTest(distributed=distributed, synchronize=synchronize):
                    events = []
                    runner.is_distributed = distributed
                    runner.agent = SimpleNamespace(update=Mock(
                        side_effect=lambda replay: events.append("update") or {}
                    ))
                    with tempfile.TemporaryDirectory() as directory, patch(
                        "torch.distributed.barrier", side_effect=lambda: events.append("barrier")
                    ):
                        runner.log_dir = directory
                        runner._profile_agent_update(123, synchronize_ranks=synchronize)
                    self.assertEqual(events, (["barrier"] if distributed and synchronize else []) + ["update"])
                    runner.agent.update.assert_called_once_with(runner.replay)

    def test_observation_metrics_include_last_joint_for_each_robot_size(self):
        for num_dofs in (24, 29):
            state = torch.zeros(5, 2 * num_dofs + 6)
            target = state.clone()
            state[:, num_dofs - 1] = 1.
            state[:, num_dofs:] = 100.
            measured = _joint_state_observation(state, num_dofs)
            reference = _joint_state_observation(target, num_dofs)
            self.assertEqual(measured.shape, (5, num_dofs))
            self.assertEqual(_distance_proximity(measured, reference, prefix="obs_state_")["obs_state_distance"], 1.)
        with self.assertRaises(ValueError):
            _joint_state_observation(torch.zeros(1, 23), 24)

    def test_eight_process_sharding_empty_shards_and_collective_failure(self):
        with tempfile.TemporaryDirectory() as path:
            torch.multiprocessing.spawn(_shard_worker, args=(os.path.join(path, "rendezvous"),), nprocs=8)

    def test_two_process_nonfinite_failure_is_collective(self):
        with tempfile.TemporaryDirectory() as path:
            torch.multiprocessing.spawn(_finite_worker, args=(os.path.join(path, "rendezvous"),), nprocs=2)

    @unittest.skipUnless(os.environ.get("PUFO_TEST_EIGHT_GPU") == "1", "Opt-in eight-GPU NCCL check")
    def test_eight_gpu_environment_errors_exit_collectively(self):
        with tempfile.TemporaryDirectory() as path:
            torch.multiprocessing.spawn(_finite_worker,
                                        args=(os.path.join(path, "rendezvous"), 8, "nccl"), nprocs=8)

    def test_metrics_accumulate_all_phases_and_sparse_diagnostics(self):
        sums, counts = {}, {}
        for i in range(20):
            metrics = {"loss": torch.tensor(float(i))}
            if i == 19:
                metrics["sparse"] = torch.tensor(7.)
            FbCprRunner._accumulate_update_metrics(sums, counts, metrics)
        self.assertEqual((sums["loss"] / counts["loss"]).item(), 9.5)
        self.assertEqual((sums["sparse"] / counts["sparse"]).item(), 7.0)

    def test_joint_metrics_differentiate_time_not_joint_axis(self):
        dt = 0.1
        target = torch.zeros(5, 2)
        pos = torch.arange(5).float()[:, None].repeat(1, 2) * dt
        metrics = _joint_pos_metrics(pos, target, dt)
        self.assertAlmostEqual(metrics["joint_velocity_l2_rad_s"], 2**0.5, places=5)
        self.assertAlmostEqual(metrics["joint_acceleration_l2_rad_s2"], 0., places=4)
        constant = torch.tensor([[1., -3.]]).repeat(5, 1)
        metrics = _joint_pos_metrics(constant, target, dt)
        self.assertEqual(metrics["joint_velocity_l2_rad_s"], 0.)
        self.assertNotIn("mpjpe_l", metrics)

    def test_remote_nonfinite_rank_stops_finite_peer(self):
        runner = self.make_runner()
        runner.is_distributed = True
        with patch("torch.distributed.all_reduce", side_effect=lambda value, **kwargs: value.fill_(1)):
            with self.assertRaisesRegex(FloatingPointError, "Another rank"):
                runner._check_finite_rollout({"obs": torch.ones(2, 3)}, "step")

    def test_local_nonfinite_any_observation_group_is_detected(self):
        runner = self.make_runner()
        with self.assertRaisesRegex(FloatingPointError, "visual"):
            runner._check_finite_rollout({"policy": torch.ones(2, 3), "visual": torch.tensor([float('nan')])}, "step")

    def test_finite_rollout_reduces_once_without_leaf_diagnostics(self):
        runner = self.make_runner()
        runner.is_distributed = True
        with patch("torch.distributed.all_reduce") as reduce, patch(
            "rsl_rl.runners.fbcpr_runner.raise_nonfinite_details"
        ) as diagnostic:
            runner._check_finite_rollout({str(i): torch.ones(2, 3) for i in range(24)}, "step")
        reduce.assert_called_once()
        diagnostic.assert_not_called()

    def test_clipped_infinite_action_is_still_rejected(self):
        runner = self.make_runner()
        runner.action_rescale = 8.0
        action = torch.tensor([float("inf")])
        env_action = runner._env_action(action)
        self.assertTrue(torch.isfinite(env_action).all())
        with self.assertRaisesRegex(FloatingPointError, "actions.action"):
            runner._check_finite_rollout({"action": action, "env_action": env_action}, "actions")

    def make_runner(self):
        runner = object.__new__(FbCprRunner)
        runner.device = "cpu"
        runner.num_envs = 2
        runner.is_distributed = False
        runner.is_main_process = True
        runner.gpu_global_rank = 0
        runner.cfg = SimpleNamespace(eval_num_motions=0, prioritization=False)
        runner._tracking_evaluator = None
        runner.writer = Mock()
        runner._reward_term_names = ("action_rate", "positive")
        runner._reward_term_scales = {"action_rate": -0.1, "positive": 2.0}
        return runner

    def test_pre_reset_rewards_are_not_recomputed_or_scaled_by_dt(self):
        runner = self.make_runner()
        cache = torch.tensor([[6.0, -0.4], [10.0, -0.8]])
        runner._mgr_env = SimpleNamespace(reward_manager=SimpleNamespace(
            active_terms=["positive", "action_rate"], _step_reward=cache,
        ))
        rewards = runner._evaluate_reward_terms()
        torch.testing.assert_close(rewards["action_rate"], torch.tensor([4.0, 8.0]))
        torch.testing.assert_close(rewards["positive"], torch.tensor([3.0, 5.0]))
        cache.zero_()
        torch.testing.assert_close(rewards["action_rate"], torch.tensor([4.0, 8.0]))

    def test_interval_logs_do_not_require_episode_completion(self):
        runner = self.make_runner()
        sums, counts = {}, {}
        for rewards in (
            {"action_rate": torch.tensor([4., 8.]), "positive": torch.tensor([3., 5.])},
            {"action_rate": torch.tensor([2., 6.]), "positive": torch.tensor([1., 3.])},
        ):
            metrics = FbCprAuxAgent._replay_reward_metrics(runner, rewards, runner._reward_term_scales)
            runner._accumulate_update_metrics(sums, counts, metrics)
        values = {key: float(value / counts[key]) for key, value in sums.items()}
        self.assertEqual(values["Reward/raw/action_rate"], 5.0)
        self.assertAlmostEqual(values["Reward/weighted/action_rate"], -0.5)
        self.assertEqual(values["Reward/weighted/positive"], 6.0)
        self.assertEqual(values["Reward/weighted/total"], 5.5)
        self.assertTrue(all(count == 2 for count in counts.values()))

    def test_distributed_reward_mean_uses_same_reduction_as_losses(self):
        runner = self.make_runner()
        runner.is_distributed = True
        metrics = {"Reward/raw/action_rate": torch.tensor(3.), "Reward/raw/positive": torch.tensor(2.),
                   "critic_loss": torch.tensor(5.)}
        with patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.all_reduce", side_effect=lambda value: value.add_(torch.tensor([9., 6., 7.]))
        ) as reduce:
            values = runner._average_metrics(metrics)
        reduce.assert_called_once()
        self.assertEqual(values["Reward/raw/action_rate"], 6.0)
        self.assertEqual(values["Reward/raw/positive"], 4.0)
        self.assertEqual(values["critic_loss"], 6.0)

    def test_worker_does_not_inherit_torchrun_or_wandb_identity(self):
        with patch.dict(os.environ, {"RANK": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "5",
                                     "MASTER_PORT": "12345", "TORCHELASTIC_RUN_ID": "x",
                                     "WANDB_RUN_ID": "training", "CUDA_VISIBLE_DEVICES": "0,1"}):
            env = FbCprRunner._eval_process_environment()
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "WANDB_RUN_ID"):
            self.assertNotIn(key, env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["WANDB_MODE"], "disabled")

    def test_evaluator_isolates_parent_logical_device_and_preserves_parent_env(self):
        for visible, expected in (("7,3,5", "3"), ("GPU-a,GPU-b", "GPU-b")):
            original = {"CUDA_VISIBLE_DEVICES": visible, "HEADLESS": "1"}
            device, environment = isolated_evaluator_device("cuda:1", original)
            self.assertEqual(device, "cuda:0")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], expected)
            self.assertEqual(environment["HEADLESS"], "1")
            self.assertEqual(original["CUDA_VISIBLE_DEVICES"], visible)

    def test_evaluator_isolation_without_mask_and_with_cpu(self):
        self.assertEqual(isolated_evaluator_device("cuda:3", {}),
                         ("cuda:0", {"CUDA_VISIBLE_DEVICES": "3"}))
        self.assertEqual(isolated_evaluator_device("cpu", {"CUDA_VISIBLE_DEVICES": "7"}),
                         ("cpu", {"CUDA_VISIBLE_DEVICES": "7"}))
        with patch("torch.cuda.current_device", return_value=1):
            self.assertEqual(isolated_evaluator_device("cuda", {"CUDA_VISIBLE_DEVICES": "2,6"}),
                             ("cuda:0", {"CUDA_VISIBLE_DEVICES": "6"}))

    def test_evaluator_isolation_rejects_out_of_range_device(self):
        for visible in ("", "0", "4,5"):
            with self.assertRaisesRegex(ValueError, "outside CUDA_VISIBLE_DEVICES"):
                isolated_evaluator_device("cuda:2", {"CUDA_VISIBLE_DEVICES": visible})

    def test_eval_merges_results_without_mutating_collector(self):
        runner = self.make_runner()
        motion_lib = SimpleNamespace(_motion_data_keys=["clip0", "clip1"], update_sampling_weights=Mock(return_value=True))
        runner._mgr_env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(motion_lib=motion_lib)))
        runner.env, runner.history, runner._context, runner._z = Mock(), object(), object(), object()
        before = (runner.env, runner.history, runner._context, runner._z)
        runner._run_eval_shard = Mock(return_value={
            "motion_keys": ["clip0", "clip1"], "motion_ids": [0, 1],
            "sums": {"emd": 2.0}, "counts": {"emd": 2},
            "clips": {0: {"emd": 1.0}, 1: {"emd": 1.0}},
        })
        runner._finalize_tracking_eval = Mock(side_effect=lambda step, metrics, clips, count: metrics)
        self.assertEqual(runner._run_tracking_eval(10), {"emd": 1.0})
        self.assertEqual(before, (runner.env, runner.history, runner._context, runner._z))
        runner.env.reset.assert_not_called()
        runner._run_eval_shard.assert_called_once_with(10, [0, 1])
        runner._finalize_tracking_eval.assert_called_once()

    def test_full_eval_updates_reset_and_expert_priorities_by_motion_id(self):
        runner = self.make_runner()
        runner.is_main_process = False
        runner.cfg = SimpleNamespace(
            prioritization=True,
            prioritization_min_val=0.5,
            prioritization_max_val=2.0,
            prioritization_scale=2.0,
            prioritization_mode="exp",
        )
        motion_lib = SimpleNamespace(
            _num_unique_motions=2,
            update_sampling_weights=Mock(return_value=True),
        )
        runner._mgr_env = SimpleNamespace(command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(motion_lib=motion_lib)
        ))
        expert = SimpleNamespace(
            motion_ids=[1, 0],
            update_priorities=Mock(),
        )
        runner.replay = {"expert_slicer": expert}
        runner._finalize_tracking_eval(
            10,
            {},
            {0: {"emd": 0.5, "obs_state_emd": 0.5},
             1: {"emd": 2.0, "obs_state_emd": 2.0}},
            2,
        )
        reset_weights = motion_lib.update_sampling_weights.call_args.args[0]
        expert_weights = expert.update_priorities.call_args.args[0]
        torch.testing.assert_close(reset_weights, torch.tensor([2.0, 16.0]))
        torch.testing.assert_close(expert_weights, torch.tensor([16.0, 2.0]))

    def test_worker_failure_is_reported(self):
        runner = self.make_runner()
        runner._mgr_env = SimpleNamespace(command_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(motion_lib=SimpleNamespace(_motion_data_keys=["clip0"]))
        ))
        runner._run_eval_shard = Mock(side_effect=RuntimeError("worker failed"))
        with self.assertRaisesRegex(RuntimeError, "Tracking evaluation failed.*worker failed"):
            runner._run_tracking_eval(10)

    def test_uneven_shards_cover_every_clip_once(self):
        for count in (1, 7, 9, 1166):
            ids = [motion_shard(count, rank, 8) for rank in range(8)]
            self.assertEqual(sorted(i for shard in ids for i in shard), list(range(count)))
            self.assertLessEqual(max(map(len, ids)) - min(map(len, ids)), 1)

    def test_merge_weights_by_clip_count_not_rank(self):
        keys = [str(i) for i in range(9)]
        results = []
        for rank in range(8):
            ids = motion_shard(9, rank, 8)
            results.append({"motion_keys": keys, "motion_ids": ids,
                            "sums": {"emd": float(sum(ids))}, "counts": {"emd": len(ids)},
                            "clips": {i: {"emd": float(i)} for i in ids}})
        self.assertEqual(merge_tracking_shards(results, keys, 9)["metrics"]["emd"], 4.)
        results[0]["motion_ids"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            merge_tracking_shards(results, keys, 9)

    def test_merge_propagates_rank_failure(self):
        with self.assertRaisesRegex(RuntimeError, "rank 3.*failed"):
            merge_tracking_shards([{"error": "rank 3: worker failed"}], [], 0)

    def test_persistent_worker_reuses_pid_and_loads_new_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = PersistentTrackingEvaluator(
                entrypoint=os.path.join(os.path.dirname(__file__), "evaluation_worker_stub.py"),
                request={}, device="cpu", environment=dict(os.environ), log_dir=directory, rank=0, timeout=20,
            )
            process = worker.process
            try:
                first = worker.evaluate({"value": torch.tensor(1.)}, 10, [0])
                second = worker.evaluate({"value": torch.tensor(2.)}, 20, [0])
                self.assertEqual(first["worker_pid"], second["worker_pid"])
                self.assertEqual(second["rounds"], 2)
                self.assertEqual((first["value"], second["value"]), (1., 2.))
            finally:
                worker.close()
            self.assertIsNotNone(process.poll())
            worker.close()

    def test_eval_preserves_collection_and_global_step_logging(self):
        self._check_collection_loop(initial_iterations=0)

    def test_resume_empty_replay_waits_for_pair_after_startup_boundary(self):
        self._check_collection_loop(initial_iterations=99)

    def test_natural_episode_reset_survives_coincident_evaluation(self):
        self._check_collection_loop(initial_iterations=0, log_every_update=True,
                                    natural_reset_at=6)

    def test_resume_warmup_collects_without_optimizer_updates(self):
        self._check_collection_loop(initial_iterations=99, warmup_steps=10)

    def test_each_update_phase_logs_16_minibatch_mean_and_sparse_diagnostics(self):
        self._check_collection_loop(initial_iterations=0, log_every_update=True, num_agent_updates=16)

    def test_profile_selects_one_minibatch_without_changing_update_cadence(self):
        for rank in (0, -1):
            with self.subTest(profile_rank=rank):
                self._check_collection_loop(initial_iterations=0, log_every_update=True,
                                            num_agent_updates=16, profile_phase=2,
                                            profile_rank=rank)

    def test_each_update_logging_waits_for_resume_warmup(self):
        self._check_collection_loop(initial_iterations=99, warmup_steps=10,
                                    log_every_update=True, num_agent_updates=16)

    def test_initial_episode_phase_randomization_is_deterministic_and_rank_local(self):
        def make_runner(rank):
            runner = object.__new__(FbCprRunner)
            runner.cfg = SimpleNamespace(seed=17)
            runner.gpu_global_rank = rank
            runner.env_iterations = 23
            runner.max_episode_length_s = 10.0
            runner.control_dt = 0.02
            runner.is_main_process = False
            runner._mgr_env = SimpleNamespace(
                max_episode_length=500,
                episode_length_buf=torch.zeros(128, dtype=torch.long),
            )
            runner._reward_episode_valid = torch.ones(128, dtype=torch.bool)
            return runner

        first = make_runner(0)
        phases = first._randomize_initial_episode_phases(True).clone()
        repeated = make_runner(0)._randomize_initial_episode_phases(True)
        other_rank = make_runner(1)._randomize_initial_episode_phases(True)
        torch.testing.assert_close(phases, repeated)
        self.assertFalse(torch.equal(phases, other_rank))
        self.assertGreaterEqual(int(phases.min()), 0)
        self.assertLess(int(phases.max()), 500)
        torch.testing.assert_close(first._reward_episode_valid, phases == 0)

    def test_partial_initial_episode_return_is_not_logged(self):
        runner = object.__new__(FbCprRunner)
        runner.is_main_process = True
        runner._reward_buf = torch.tensor([3.0, 7.0])
        runner._reward_episode_valid = torch.tensor([False, True])
        runner._rewbuffer = []
        runner._finish_reward_episodes(torch.tensor([0, 1]))
        self.assertEqual(runner._rewbuffer, [7.0])
        torch.testing.assert_close(runner._reward_buf, torch.zeros(2))
        self.assertTrue(runner._reward_episode_valid.all())

    def _check_collection_loop(self, initial_iterations, warmup_steps=0,
                               log_every_update=False, num_agent_updates=1,
                               natural_reset_at=None, profile_phase=0, profile_rank=0):
        runner = self.make_runner()
        runner.cfg = SimpleNamespace(num_seed_steps=0, update_agent_every_samples=2,
                                     num_agent_updates=num_agent_updates, num_env_steps=0,
                                     log_every_update=log_every_update,
                                     profile_update_phase=profile_phase,
                                     profile_update_rank=profile_rank)
        runner.eval_interval_steps = 6
        runner.log_interval_steps = 8
        runner.checkpoint_interval_steps = 10000
        runner.env_iterations = runner.tot_iterations = initial_iterations
        runner._replay_warmup_until = initial_iterations * 2 + warmup_steps if warmup_steps else 0
        runner._expert_built = True
        runner.visual_enabled = log_every_update
        runner.history = Mock()
        runner.max_episode_length_s = 10.
        runner._reward_buf = torch.zeros(2)
        runner.control_dt = .02
        runner._rewbuffer = []
        runner._z = runner._context = None
        runner.action_rescale = 1.
        runner.log_dir = '/unused'
        runner._mgr_env = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long),
                                         action_manager=SimpleNamespace(total_action_dim=1))
        resets = []
        steps = []

        def observation(state):
            depth = state.to(torch.uint8).reshape(2, 1, 1, 1).expand(2, 1, 2, 2).clone()
            return {"state": state, "depth_image": depth}

        def reset():
            resets.append(runner.env_iterations)
            runner._mgr_env.episode_length_buf.zero_()
            return observation(torch.full((2, 1), float(100 * len(resets)))), {}

        def step(action):
            steps.append(action.clone())
            runner._mgr_env.episode_length_buf.add_(1)
            state = 100 * len(resets) + runner._mgr_env.episode_length_buf.float()[:, None]
            dones = torch.zeros(2, dtype=torch.bool)
            infos = {}
            if len(steps) == natural_reset_at:
                final_obs = observation(state.clone())
                final_obs["transition_kinematics"] = torch.ones(2, 4)
                infos = {"final_obs": final_obs, "_final_obs": torch.tensor([True, False])}
                dones[0] = True
                runner._mgr_env.episode_length_buf[0] = 0
                state[0] = 100.
            return observation(state), None, dones, infos

        runner.env = SimpleNamespace(reset=reset, step=step)
        runner.agent = Mock()
        runner.agent.act.return_value = torch.zeros(2, 1)
        runner.agent.maybe_update_rollout_context.return_value = torch.zeros(2, 1)
        runner.agent.advance_rollout_context.return_value = SimpleNamespace(z=torch.zeros(2, 1))
        runner.agent.context_observations.return_value = {}
        runner.agent.context_transition.return_value = {"heading_next": torch.zeros(2, 2),
                                                         "velocity_next": torch.zeros(2, 3)}
        runner._root_heading_velocity = lambda: (torch.zeros(2, 2), torch.zeros(2, 2))
        sampled_lengths = []
        diagnostic_steps = []

        def update(replay, **kwargs):
            replay["train"].sample(4)
            sampled_lengths.append(len(replay["train"]))
            if kwargs["depth_gradient_diagnostics"]:
                diagnostic_steps.append(runner.env_iterations * 2)
            self.assertEqual(kwargs["reward_term_scales"], runner._reward_term_scales)
            # Distinct from current rollout rewards, so logging cannot use them.
            return {"critic_loss": torch.tensor(float(len(sampled_lengths))), "Reward/raw/action_rate": torch.tensor(12.),
                    "Reward/weighted/action_rate": torch.tensor(-1.2),
                    "Reward/weighted/total": torch.tensor(-1.2)}

        runner.agent.update.side_effect = update
        def profiled_update(step, *, synchronize_ranks, **kwargs):
            self.assertEqual(synchronize_ranks, profile_rank == -1)
            return runner.agent.update(runner.replay, **kwargs)

        runner._profile_agent_update = Mock(side_effect=profiled_update)
        runner._obs_dict = lambda obs: {key: value for key, value in obs.items() if key != "transition_kinematics"}
        runner._current_obs_values = lambda obs: obs
        runner._physics_snapshot = lambda: {}
        runner._evaluate_reward_terms = lambda: {"action_rate": torch.ones(2), "positive": torch.ones(2)}
        runner._run_tracking_eval = Mock()
        runner.replay = {"train": ReplayBuffer(capacity=100, num_envs=2, end_key="episode_end",
                                               transition_outcomes=True)}
        runner.learn(9)
        self.assertEqual(runner._profile_agent_update.call_count, int(profile_phase > 0))
        if profile_phase:
            self.assertEqual(runner._profile_agent_update.call_args.args[0],
                             2 * initial_iterations + 6 + 2 * (profile_phase - 1))
        base_step = 2 * initial_iterations
        self.assertEqual([call.args[0] if call.args else call.kwargs['global_step']
                          for call in runner._run_tracking_eval.call_args_list],
                         [base_step + offset for offset in (0, 6, 12, 18)])
        self.assertEqual(resets, [initial_iterations])
        self.assertEqual(runner.history.add.call_count, 9)
        reset_calls = [call.args[0].tolist() for call in runner.history.reset.call_args_list]
        self.assertEqual(reset_calls, [[0, 1]] + [
            [0] if i == natural_reset_at else [] for i in range(1, 10)
        ])
        if runner.visual_enabled:
            context_calls = runner.agent.advance_rollout_context.call_args_list
            self.assertIsNone(context_calls[0].args[0])
            self.assertTrue(all(call.args[0] is runner._context for call in context_calls[1:]))
        else:
            context_calls = runner.agent.maybe_update_rollout_context.call_args_list
            self.assertIsNone(context_calls[0].args[0])
            self.assertTrue(all(call.args[0] is runner._z for call in context_calls[1:]))
        for i, call in enumerate(runner.agent.act.call_args_list):
            expected = torch.full((2, 1), 100. + i)
            if natural_reset_at is not None and i >= natural_reset_at:
                expected[0] -= natural_reset_at
            torch.testing.assert_close(call.args[0]['state'], expected)
            torch.testing.assert_close(call.args[0]['depth_image'][:, 0, 0, 0],
                                       expected[:, 0].to(torch.uint8))
        expected_rewards = torch.full((2,), 9 * 1.9 * runner.control_dt)
        if natural_reset_at is not None:
            expected_rewards[0] = (9 - natural_reset_at) * 1.9 * runner.control_dt
        torch.testing.assert_close(runner._reward_buf, expected_rewards)
        self.assertEqual(len(runner._rewbuffer), int(natural_reset_at is not None))
        self.assertEqual(sampled_lengths[0], max(3, warmup_steps // 2 + 1))
        logged = runner.writer.log_metrics.call_args_list
        smoothed = [call for call in logged if "train/critic_loss" in call.args[0]]
        loss_steps = [call.args[1] for call in smoothed]
        expected_log_offsets = ([12] if warmup_steps else [8, 16])
        expected_diagnostic_offsets = ([12] if warmup_steps else [8, 16])
        if log_every_update:
            expected_log_offsets = list(range(12 if warmup_steps else 6, 19, 2))
            expected_diagnostic_offsets = ([12] if warmup_steps else [6, 14])
            self.assertEqual([call.args[0]['train/critic_loss'] for call in smoothed],
                             [i * num_agent_updates + (num_agent_updates + 1) / 2
                              for i in range(len(smoothed))])
            self.assertEqual(diagnostic_steps,
                             [base_step + offset for offset in expected_diagnostic_offsets])
        self.assertEqual(loss_steps, [base_step + offset for offset in expected_log_offsets])
        self.assertTrue(all(call.args[0]['Reward/raw/action_rate'] == 12. for call in smoothed))
        diagnostic_logs = [
            call for call in smoothed
            if "diagnostics/rollout/reset_fraction" in call.args[0]
        ]
        self.assertEqual(
            [call.args[1] for call in diagnostic_logs],
            [base_step + offset for offset in expected_diagnostic_offsets],
        )
        self.assertTrue(all("diagnostics/replay/trajectory_count" in call.args[0]
                            for call in diagnostic_logs))
        self.assertFalse(any(key.startswith(('Episode_Reward/', 'train/Reward/'))
                             for call in smoothed for key in call.args[0]))
        buffer = runner.replay['train']
        expected_boundaries = torch.zeros(9, 2, 1, dtype=torch.bool)
        if natural_reset_at is not None:
            expected_boundaries[natural_reset_at - 1, 0] = True
        torch.testing.assert_close(buffer.storage['truncated'][:9], expected_boundaries)
        torch.testing.assert_close(buffer.storage['transition_truncated'][:9], expected_boundaries)
        if runner.visual_enabled:
            expected_starts = torch.zeros_like(expected_boundaries)
            expected_starts[0] = True
            if natural_reset_at is not None:
                expected_starts[natural_reset_at, 0] = True
            torch.testing.assert_close(buffer.storage['depth_episode_start'][:9], expected_starts)
        self.assertFalse(buffer.storage['terminated'][:9].any())
        sampled = buffer.sample(200)
        # Evaluation preserves continuity; ordinary reset boundaries still exclude invalid pairs.
        torch.testing.assert_close(sampled['next']['observation']['state'] - sampled['observation']['state'],
                                   torch.ones(200, 1))


if __name__ == "__main__":
    unittest.main()
