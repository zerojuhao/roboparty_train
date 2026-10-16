import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rsl_rl.runners.fbcpr_runner import FbCprRunner
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.storage.replay_checkpoint import load_replay, save_replay


def buffer(num_envs=2, steps=4):
    result = ReplayBuffer(12, num_envs, output_key_t=("observation",), output_key_tp1=("observation",))
    for i in range(steps):
        result.extend({"observation": {"state": torch.full((1, num_envs, 1), float(i)),
                                       "depth": torch.full((1, num_envs, 2, 3), i, dtype=torch.uint8)},
                       "truncated": torch.zeros(1, num_envs, 1, dtype=torch.bool)})
    return result


def runner(rank=0, distributed=False, steps=4):
    obj = object.__new__(FbCprRunner)
    obj.device = "cpu"
    obj.cfg = SimpleNamespace()
    obj.num_envs = 2
    obj.is_distributed = distributed
    obj.gpu_global_rank = rank
    obj.is_main_process = rank == 0
    obj._eval_group = None
    obj.env_iterations = 100
    obj.tot_iterations = 80
    obj.action_rescale = 8.
    obj.checkpoint_replay = True
    obj.replay_checkpoint_keep = 1
    obj.resume_replay_warmup_steps = 20
    obj.replay = {"train": buffer(steps=steps)}
    obj.agent = Mock()
    obj.agent.state_dict.return_value = {"weight": torch.tensor([1.])}
    obj.agent.optimizer_state_dict.return_value = {"step": 80}
    return obj


def distributed_worker(rank, rendezvous, directory):
    from datetime import timedelta
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=8, timeout=timedelta(seconds=60))
    try:
        obj = runner(rank, True)
        obj.replay["train"].storage["observation"]["state"] += rank * 100
        path = Path(directory) / "model_1600.pt"
        obj.save(path)
        restored = runner(rank, True, steps=0)
        restored.load(path)
        assert restored.replay['train'].storage['observation']['state'][0, 0, 0].item() == rank * 100
        assert restored._replay_warmup_until == 0
        def fail(*args, **kwargs):
            if rank == 3:
                raise OSError("injected disk failure")
            return save_replay(*args, **kwargs)
        with patch('rsl_rl.runners.fbcpr_runner.save_replay', side_effect=fail):
            try:
                obj.save(Path(directory) / "model_failed.pt")
            except RuntimeError as exc:
                assert 'injected disk failure' in str(exc)
            else:
                raise AssertionError('Every rank must see shard write failure')
        assert not (Path(directory) / 'model_failed.pt').exists()
    finally:
        torch.distributed.destroy_process_group()


class ReplayCheckpointTests(unittest.TestCase):
    def test_legacy_training_contract_metadata_is_not_retained(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            source = runner()
            source._reward_term_scales = {"unchanged": -0.1}
            source.save(path)
            dest = runner(steps=0)
            dest._reward_term_scales = dict(source._reward_term_scales)
            dest.load(path)
            dest.agent.load_state_dict.assert_called_once()
            dest.agent.load_optimizer_state_dict.assert_called_once_with({"step": 80})
            self.assertNotIn("training_contract", torch.load(path, weights_only=False))

    def test_changed_or_legacy_rewards_recollect_replay_and_preserve_optimizer(self):
        for saved in ({"old": -0.1}, {"new": -0.2}, None):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "model.pt"
                source = runner()
                source._reward_term_scales = saved
                source.save(path)
                if saved is None:
                    payload = torch.load(path, weights_only=False)
                    payload.pop("reward_term_scales")
                    torch.save(payload, path)
                dest = runner(steps=0)
                dest._reward_term_scales = {"new": -0.1}
                dest.load(path)
                dest.agent.load_state_dict.assert_called_once()
                dest.agent.load_optimizer_state_dict.assert_called_once_with({"step": 80})
                self.assertEqual((dest.env_iterations, dest.tot_iterations, dest._replay_warmup_until), (100, 80, 220))
                self.assertTrue(dest.replay["train"].empty())
                self.assertTrue(any(Path(d).glob("replay_*/manifest.json")))

    def test_unchanged_rewards_restore_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            source = runner()
            source._reward_term_scales = {"new": -0.1}
            source.save(path)
            dest = runner(steps=0)
            dest._reward_term_scales = dict(source._reward_term_scales)
            dest.load(path)
            self.assertEqual(len(dest.replay["train"]), 4)
            self.assertEqual(dest._replay_warmup_until, 0)

    def test_legacy_replay_is_recollected_without_discarding_weights_or_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            source = runner()
            source.save(path)
            dest = runner(steps=0)
            dest.replay["train"] = ReplayBuffer(12, 2, end_key="episode_end", transition_outcomes=True)
            dest.load(path)
            dest.agent.load_state_dict.assert_called_once()
            dest.agent.load_optimizer_state_dict.assert_called_once_with({"step": 80})
            self.assertEqual((dest.env_iterations, dest.tot_iterations, dest._replay_warmup_until), (100, 80, 220))
            self.assertTrue(dest.replay["train"].empty())

    def test_partial_full_and_wrapped_buffer_roundtrip(self):
        for steps in (0, 4, 6, 9):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as d:
                source = buffer(steps=steps)
                path = Path(d) / 'rank_0'
                save_replay(source, path, rank=0, world_size=1, env_iterations=steps, chunk_bytes=24)
                restored = buffer(steps=0)
                load_replay(restored, path, rank=0, world_size=1, env_iterations=steps)
                self.assertEqual((len(restored), restored._idx, restored._is_full),
                                 (len(source), source._idx, source._is_full))
                if steps:
                    torch.testing.assert_close(restored.storage['observation']['state'][:len(source)],
                                               source.storage['observation']['state'][:len(source)])
                    torch.testing.assert_close(restored.storage['observation']['depth'][:len(source)],
                                               source.storage['observation']['depth'][:len(source)])
                    torch.manual_seed(11)
                    expected = source.sample(100)['observation']['state']
                    torch.manual_seed(11)
                    torch.testing.assert_close(restored.sample(100)['observation']['state'], expected)

    def test_contract_mismatch_and_missing_chunk_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rank_0'
            save_replay(buffer(), path, rank=0, world_size=1, env_iterations=4)
            for kwargs in ({'rank': 1}, {'world_size': 2}, {'env_iterations': 5}):
                options = dict(rank=0, world_size=1, env_iterations=4)
                options.update(kwargs)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_replay(buffer(steps=0), path, **options)
            next(path.glob('tensor_*.pt')).unlink()
            with self.assertRaises(FileNotFoundError):
                load_replay(buffer(steps=0), path, rank=0, world_size=1, env_iterations=4)

    def test_model_and_replay_restore_and_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model_200.pt'
            source = runner()
            source.save(path)
            dest = runner(steps=0)
            dest.checkpoint_replay = False  # Saving policy must not disable restoring existing replay.
            dest.load(path)
            self.assertEqual((dest.env_iterations, dest.tot_iterations, dest._replay_warmup_until), (100, 80, 0))
            dest.agent.load_optimizer_state_dict.assert_called_once_with({'step': 80})
            self.assertTrue(dest.replay['train'].storage['truncated'][3].all())
            self.assertFalse(source.replay['train'].storage['truncated'][3].any())
            dest.replay['train'].extend({'observation': {'state': torch.full((2, 2, 1), 100.),
                                                         'depth': torch.zeros(2, 2, 2, 3, dtype=torch.uint8)},
                                        'truncated': torch.zeros(2, 2, 1, dtype=torch.bool)})
            sampled = dest.replay['train'].sample(100)
            delta = sampled['next']['observation']['state'] - sampled['observation']['state']
            self.assertTrue((delta <= 1).all())

    def test_disabled_missing_empty_and_legacy_replay_warm_up(self):
        for mode in ('disabled', 'missing', 'empty', 'legacy'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'model_200.pt'
                source = runner(steps=0 if mode == 'empty' else 4)
                source.checkpoint_replay = mode != 'disabled'
                source.save(path)
                payload = torch.load(path, weights_only=False)
                if mode == 'missing':
                    shutil.rmtree(Path(d) / payload['replay_checkpoint'] / 'rank_0')
                if mode == 'legacy':
                    for key in ('replay_checkpoint', 'world_size', 'num_envs'):
                        payload.pop(key)
                    torch.save(payload, path)
                dest = runner(steps=0)
                dest.load(path)
                self.assertEqual(dest._replay_warmup_until, 220)
                self.assertTrue(dest._resumed)

    def test_retention_keeps_models_and_latest_replay(self):
        with tempfile.TemporaryDirectory() as d:
            obj = runner()
            old, new = Path(d) / 'model_200.pt', Path(d) / 'model_202.pt'
            obj.save(old)
            obj.env_iterations += 1
            obj.save(new)
            self.assertEqual(len(list(Path(d).glob('replay_*'))), 1)
            self.assertTrue(old.is_file() and new.is_file())
            dest = runner(steps=0)
            dest.load(old)
            self.assertGreater(dest._replay_warmup_until, 0)
            dest.load(new)
            self.assertEqual(dest._replay_warmup_until, 0)

    def test_inference_load_skips_replay_and_training_topology(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model_200.pt'
            source = runner()
            source.save(path)
            dest = runner(steps=0)
            dest.num_envs = 1
            dest.load(path, load_optimizer=False)
            dest.agent.load_state_dict.assert_called_once()
            dest.agent.load_optimizer_state_dict.assert_not_called()
            self.assertEqual(len(dest.replay['train']), 0)
            with self.assertRaisesRegex(ValueError, 'num_envs'):
                dest.load(path)

    def test_cli_default_and_boolean_flags(self):
        path = Path(__file__).resolve().parents[3] / 'robo_newton/scripts/rsl_rl/cli_args.py'
        spec = importlib.util.spec_from_file_location('checkpoint_cli_args', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = argparse.ArgumentParser()
        module.add_rsl_rl_args(parser)
        for args, expected in (([], True), (['--checkpoint-replay'], True), (['--no-checkpoint-replay'], False)):
            cfg = SimpleNamespace(checkpoint_replay=True, logger='tensorboard')
            module.update_rsl_rl_cfg(cfg, parser.parse_args(args))
            self.assertEqual(cfg.checkpoint_replay, expected)

    def test_eight_rank_checkpoint_restore_and_collective_write_failure(self):
        with tempfile.TemporaryDirectory() as d:
            torch.multiprocessing.spawn(distributed_worker, args=(os.path.join(d, 'rendezvous'), d), nprocs=8)


if __name__ == '__main__':
    unittest.main()
