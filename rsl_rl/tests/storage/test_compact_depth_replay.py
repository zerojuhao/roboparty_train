"""Compact history must reproduce the collector's delayed uint8 stacks exactly."""

from pathlib import Path
import tempfile
import unittest

import torch

from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.storage.replay_checkpoint import load_replay, save_replay


OFFSETS = (35, 30, 25, 20, 15, 10, 5, 0)


def make_buffer(capacity=150, offsets=OFFSETS):
    return ReplayBuffer(capacity, 3, output_key_t=("observation",), output_key_tp1=("observation",),
                        compact_depth_history=True, depth_history_offsets=offsets)


def records(count):
    depth = torch.empty(count, 3, 8, 2, 3, dtype=torch.uint8)
    starts = torch.zeros(count, 3, 1, dtype=torch.bool)
    for env in range(3):
        episode_start = 0
        for t in range(count):
            if t == 0 or t % (13 + env * 7) == 0 or t == 26:
                episode_start = t
                starts[t, env] = True
            delay = (env + episode_start) % 3
            for frame, offset in enumerate(OFFSETS):
                depth[t, env, frame] = (max(episode_start, t - delay - offset) + env * 71) % 256
    return {
        "observation": {"depth_image": depth, "state": torch.arange(count).view(count, 1, 1).expand(-1, 3, -1),
                        "env_id": torch.arange(3).view(1, 3, 1).expand(count, -1, -1)},
        "truncated": starts.clone(), "depth_episode_start": starts,
    }


def slice_records(data, start, end):
    return {key: {k: v[start:end] for k, v in value.items()} if isinstance(value, dict)
            else value[start:end] for key, value in data.items()}


class CompactDepthReplayTests(unittest.TestCase):
    def test_terminal_outcomes_async_resets_ring_sequences_and_checkpoint(self):
        offsets = (2, 0)
        buffer = ReplayBuffer(36, 3, output_key_t=("observation", "action"),
                              output_key_tp1=("observation", "terminated", "truncated"),
                              end_key="episode_end", transition_outcomes=True,
                              compact_depth_history=True, depth_history_offsets=offsets)
        expected = {}
        states = [0, 100, 200]
        episode_starts = [0, 0, 0]
        frames = [[], [], []]
        for t in range(40):
            current, outcome, done, terminated, starts = [], [], [], [], []
            for env, period in enumerate((1, 3, 7)):
                frames[env].append(states[env] % 256)
                depth = [frames[env][max(episode_starts[env], t - offset)] for offset in offsets]
                next_depth = [((states[env] + 1) % 256) if offset == 0
                              else frames[env][max(episode_starts[env], t + 1 - offset)] for offset in offsets]
                current.append((states[env], depth))
                outcome.append((states[env] + 1, next_depth))
                done.append((t + 1) % period == 0)
                terminated.append(done[-1] and env == 1)
                starts.append(t == episode_starts[env])
                expected[t, env] = (current[-1], outcome[-1], done[-1], terminated[-1])
                if done[-1]:
                    states[env] += 1000
                    episode_starts[env] = t + 1
                else:
                    states[env] += 1
            def observation(values):
                return {"state": torch.tensor([value[0] for value in values]).reshape(1, 3, 1),
                        "depth_image": torch.tensor([value[1] for value in values], dtype=torch.uint8).reshape(1, 3, 2, 1, 1),
                        "time": torch.full((1, 3, 1), t),
                        "env": torch.arange(3).reshape(1, 3, 1)}
            buffer.extend({"observation": observation(current), "action": torch.full((1, 3, 1), t),
                           "episode_end": torch.tensor(done).reshape(1, 3, 1),
                           "terminated": torch.tensor(terminated).reshape(1, 3, 1),
                           "truncated": torch.tensor([d and not term for d, term in zip(done, terminated)]).reshape(1, 3, 1),
                           "depth_episode_start": torch.tensor(starts).reshape(1, 3, 1)},
                          final_observation=observation(outcome))
            if t in (2, 11, 25, 39):
                for seq_length in (1, 2):
                    sampled = buffer.sample(512, seq_length)
                    times = sampled["observation"]["time"].flatten().tolist()
                    envs = sampled["observation"]["env"].flatten().tolist()
                    for row, key in enumerate(zip(times, envs)):
                        before, after, end, term = expected[key]
                        self.assertEqual(sampled["observation"]["state"][row].item(), before[0])
                        self.assertEqual(sampled["next"]["observation"]["state"][row].item(), after[0])
                        torch.testing.assert_close(sampled["observation"]["depth_image"][row, :, 0, 0], torch.tensor(before[1], dtype=torch.uint8))
                        torch.testing.assert_close(sampled["next"]["observation"]["depth_image"][row, :, 0, 0], torch.tensor(after[1], dtype=torch.uint8))
                        self.assertEqual(sampled["next"]["terminated"][row].item(), term)
                        self.assertEqual(sampled["next"]["truncated"][row].item(), end and not term)
                    if seq_length == 2:
                        self.assertFalse(sampled["next"]["terminated"][::2].any())
                        self.assertFalse(sampled["next"]["truncated"][::2].any())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay"
            save_replay(buffer, path, rank=0, world_size=1, env_iterations=40, chunk_bytes=80)
            restored = ReplayBuffer(36, 3, output_key_t=("observation", "action"),
                                    output_key_tp1=("observation", "terminated", "truncated"),
                                    end_key="episode_end", transition_outcomes=True,
                                    compact_depth_history=True, depth_history_offsets=offsets)
            load_replay(restored, path, rank=0, world_size=1, env_iterations=40)
            torch.manual_seed(12)
            source_sample = buffer.sample(512)
            torch.manual_seed(12)
            restored_sample = restored.sample(512)
            for name in ("state", "depth_image"):
                torch.testing.assert_close(source_sample["next"]["observation"][name], restored_sample["next"]["observation"][name])
            restored.close_trajectory()
            self.assertEqual(set(restored.terminal_observations), set(buffer.terminal_observations))
            next(path.glob("terminal_*.pt")).unlink()
            with self.assertRaises(FileNotFoundError):
                load_replay(restored, path, rank=0, world_size=1, env_iterations=40)

    def assert_samples_match(self, buffer, data, seq_length=1):
        batch = buffer.sample(4096, seq_length=seq_length)
        for obs in (batch["observation"], batch["next"]["observation"]):
            time = obs["state"].flatten().long()
            env = obs["env_id"].flatten().long()
            torch.testing.assert_close(obs["depth_image"], data["observation"]["depth_image"][time, env])

    def test_partial_full_wrap_async_resets_delays_and_sequences(self):
        data = records(130)
        buffer = make_buffer()
        for start, end in ((0, 20), (20, 50), (50, 87), (87, 130)):
            buffer.extend(slice_records(data, start, end))
            self.assert_samples_match(buffer, data)
            self.assert_samples_match(buffer, data, seq_length=4)
        self.assertEqual(buffer.storage["observation"]["depth_image"].shape, (50, 3, 2, 3))
        self.assertEqual(buffer.storage["observation"]["depth_image"].numel() * 8,
                         50 * 3 * 8 * 2 * 3)

    def test_eval_marker_does_not_reset_previous_frames(self):
        data = records(40)
        data["truncated"][25] = True
        buffer = make_buffer()
        buffer.extend(slice_records(data, 0, 26))
        buffer.close_trajectory()
        buffer.extend(slice_records(data, 26, 40))
        self.assert_samples_match(buffer, data)
        indices = (torch.tensor([25, 26]), torch.tensor([0, 0]))
        rebuilt = buffer._restore_depth({}, indices)["depth_image"]
        torch.testing.assert_close(rebuilt, data["observation"]["depth_image"][indices])

    def test_checkpoint_roundtrip_and_continued_ring_append(self):
        data = records(130)
        for count in (20, 50, 87):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                source = make_buffer()
                for start in range(0, count, 20):
                    source.extend(slice_records(data, start, min(start + 20, count)))
                path = Path(directory) / "replay"
                save_replay(source, path, rank=0, world_size=1, env_iterations=count, chunk_bytes=80)
                restored = make_buffer()
                load_replay(restored, path, rank=0, world_size=1, env_iterations=count)
                self.assert_samples_match(restored, data)
                restored.extend(slice_records(data, count, count + 20))
                self.assert_samples_match(restored, data)
                with self.assertRaisesRegex(ValueError, "depth"):
                    load_replay(make_buffer(offsets=(30, 25, 20, 15, 10, 5, 1, 0)), path,
                                rank=0, world_size=1, env_iterations=count)

    def test_invalid_depth_and_too_small_capacity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            make_buffer(capacity=60)
        data = records(4)
        data["observation"]["depth_image"] = data["observation"]["depth_image"].float()
        with self.assertRaisesRegex(ValueError, "uint8"):
            make_buffer().extend(data)

    def test_resume_preserves_pending_camera_reset(self):
        data = records(40)
        source = make_buffer()
        source.extend(slice_records(data, 0, 26))
        source.close_trajectory()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay"
            save_replay(source, path, rank=0, world_size=1, env_iterations=26)
            restored = make_buffer()
            load_replay(restored, path, rank=0, world_size=1, env_iterations=26)
            continuation = slice_records(data, 26, 40)
            continuation["depth_episode_start"] = continuation["depth_episode_start"].clone()
            continuation["depth_episode_start"][0] = False
            restored.extend(continuation)
            self.assert_samples_match(restored, data)
            self.assertTrue(restored.storage["depth_episode_start"][26].all())

    def test_full_stack_checkpoint_cannot_be_loaded_as_compact(self):
        data = records(20)
        source = ReplayBuffer(150, 3, output_key_t=("observation",), output_key_tp1=("observation",))
        source.extend(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay"
            save_replay(source, path, rank=0, world_size=1, env_iterations=20)
            with self.assertRaisesRegex(ValueError, "Replay checkpoint mismatch"):
                load_replay(make_buffer(), path, rank=0, world_size=1, env_iterations=20)


if __name__ == "__main__":
    unittest.main()
