"""Immediate, independent train/eval logging without contacting W&B."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rsl_rl.runners.fbcpr_runner import FbCprRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from rsl_rl.utils import wandb_log_writer


class TestWandbMetrics(unittest.TestCase):
    @unittest.skipIf(wandb_log_writer.wandb is None, "W&B SDK not installed")
    def test_real_offline_sdk_preserves_training_and_eval_same_step(self):
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "WANDB_MODE": "offline", "WANDB_DIR": directory, "WANDB_SILENT": "true",
        }):
            writer = WandbLogWriter(directory, "logging-regression-test")
            try:
                writer.log_metrics({"train/critic_loss": .5}, 8192)
                writer.log_metrics({"eval/emd": 2.}, 8192)
                writer.log_metrics({"train/critic_loss": .25}, 16384)
            finally:
                writer.stop()
                writer.close()
            files = list(Path(directory).rglob("run-*.wandb"))
            self.assertEqual(len(files), 1)
            store = DataStore()
            rows = []
            store.open_for_scan(str(files[0]))
            try:
                while (data := store.scan_data()) is not None:
                    record = wandb_internal_pb2.Record()
                    record.ParseFromString(data)
                    if record.HasField("history"):
                        rows.append({item.key or "/".join(item.nested_key): json.loads(item.value_json)
                                     for item in record.history.item})
            finally:
                store.close()
            self.assertEqual([row["env_steps"] for row in rows], [8192, 8192, 16384])
            self.assertEqual([row["_step"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[0]["train/critic_loss"], .5)
            self.assertEqual(rows[1]["eval/emd"], 2.)
            self.assertEqual(rows[2]["train/critic_loss"], .25)

    def test_batches_commit_independently_at_same_env_step(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "rsl_rl.utils.wandb_log_writer.wandb"
        ) as wandb:
            writer = WandbLogWriter(directory, "test")
            try:
                train = {"Reward/raw/action_rate": 12., "train/critic_loss": .25}
                writer.log_metrics(train, 8192)
                # Already committed before evaluation or the next training phase.
                wandb.log.assert_called_once_with({**train, "env_steps": 8192}, commit=True)
                writer.log_metrics({"eval/emd": 2.}, 8192)
                writer.log_metrics({"train/critic_loss": .2}, 16384)
                self.assertEqual(wandb.log.call_count, 3)
                self.assertTrue(all(call.kwargs == {"commit": True} for call in wandb.log.call_args_list))
                self.assertEqual([call.args[0]["env_steps"] for call in wandb.log.call_args_list],
                                 [8192, 8192, 16384])
                wandb.define_metric.assert_any_call("eval/*", step_metric="env_steps")
                wandb.define_metric.assert_any_call("diagnostics/*", step_metric="env_steps")
                self.assertEqual(wandb.define_metric.call_count, 7)
                events = EventAccumulator(directory).Reload()
                self.assertEqual([event.step for event in events.Scalars("train/critic_loss")], [8192, 16384])
                self.assertEqual(events.Scalars("eval/emd")[0].step, 8192)
            finally:
                writer.close()

    def test_plain_tensorboard_writer_and_disabled_writer(self):
        runner = object.__new__(FbCprRunner)
        with tempfile.TemporaryDirectory() as directory:
            runner.writer = SummaryWriter(directory)
            try:
                runner._write_metrics({"train/loss": .5}, 1024)
                events = EventAccumulator(directory).Reload()
                self.assertEqual(events.Scalars("train/loss")[0].value, .5)
            finally:
                runner.writer.close()
        runner.writer = None
        runner._write_metrics({"train/loss": .5}, 1024)

    def test_terminal_includes_both_reward_namespaces_and_auxiliary_losses(self):
        metrics = {"Reward/raw/action_rate": 12., "Reward/weighted/action_rate": -1.2,
                   "Reward/weighted/total": -1.2, "critic_loss": .5, "aux_critic_loss": .25,
                   "velocity_estimator_loss": .1}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            FbCprRunner._print_training_metrics(metrics, 8192, 1, 1600)
        for text in ("Reward/raw: action_rate=12.0000", "Reward/weighted: action_rate=-1.2000",
                     "aux_critic_loss=0.2500", "velocity_estimator_loss=0.1000"):
            self.assertIn(text, output.getvalue())

    def test_evaluation_commits_without_training_batch(self):
        runner = object.__new__(FbCprRunner)
        runner.writer = Mock()
        runner._write_metrics({"eval/emd": 2.}, 0)
        runner.writer.log_metrics.assert_called_once_with({"eval/emd": 2.}, 0)


if __name__ == "__main__":
    unittest.main()
