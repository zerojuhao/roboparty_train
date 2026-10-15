"""Sampling caches must preserve seeded sampling and observe every invalidation."""

from unittest.mock import patch

import pytest
import torch

from rsl_rl.storage.replay_buffer import ReplayBuffer, TrajectoryDictBuffer
from rsl_rl.storage.replay_checkpoint import load_replay, save_replay


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def uncached_metadata(buffer, seq_length):
    offset = int(bool(buffer.output_key_tp1))
    if isinstance(buffer, TrajectoryDictBuffer):
        eligible = buffer.lengths >= seq_length + offset
        return (buffer.lengths[eligible], buffer.start_idx[eligible],
                buffer.lengths[eligible] - seq_length - offset + 1,
                eligible)
    buffer._refresh_traj()
    windows = buffer._valid_windows(seq_length)
    eligible = windows > 0
    return (buffer.lengths[eligible], buffer.start_idx[eligible],
            windows[eligible], windows[eligible].float())


def assert_seeded_samples_match(buffer, seq_length):
    torch.manual_seed(123)
    with patch.object(buffer, "_sampling_metadata", side_effect=lambda n: uncached_metadata(buffer, n)):
        expected = buffer.sample(120, seq_length)
    torch.manual_seed(123)
    actual = buffer.sample(120, seq_length)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def records(start, count, device):
    times = torch.arange(start, start + count, device=device).reshape(-1, 1, 1)
    envs = torch.arange(3, device=device).reshape(1, -1, 1)
    return {"observation": {"state": times * 10 + envs},
            "truncated": (times + envs) % 7 == 6}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("next_keys", [(), ("observation",)])
def test_online_cache_matches_original_sampling_through_wrap_and_close(device, next_keys):
    buffer = ReplayBuffer(60, 3, device=device, output_key_t=("observation",), output_key_tp1=next_keys)
    previous = None
    for start, count in ((0, 12), (12, 8), (20, 13), (33, 20)):
        buffer.extend(records(start, count, device))
        for seq_length in (1, 2, 3):
            assert_seeded_samples_match(buffer, seq_length)
        metadata = buffer._sampling_metadata(1)
        assert metadata is buffer._sampling_metadata(1)
        assert metadata is not previous
        buffer.close_trajectory()
        assert_seeded_samples_match(buffer, 1)
        previous = buffer._sampling_metadata(1)
        assert previous is not metadata


@pytest.mark.parametrize("device", DEVICES)
def test_expert_cache_uses_live_priorities(device):
    episodes = []
    for i, length in enumerate((2, 5, 9)):
        ends = torch.zeros(length, 1, dtype=torch.bool, device=device)
        ends[-1] = True
        episodes.append({"motion_id": torch.full((length, 1), i, device=device),
                         "observation": {"state": torch.arange(length, device=device)[:, None] + i * 100},
                         "truncated": ends})
    buffer = TrajectoryDictBuffer(episodes, device=device)
    for seq_length in (1, 2, 3):
        assert_seeded_samples_match(buffer, seq_length)
    metadata = buffer._sampling_metadata(1)
    buffer.update_priorities([0, 0, 1])
    assert buffer._sampling_metadata(1) is metadata
    for seq_length in (1, 2, 3):
        assert_seeded_samples_match(buffer, seq_length)
        assert (buffer.sample(120, seq_length)["observation"]["state"] >= 200).all()
    buffer.update_priorities([1, 0], idxs=[0, 2])
    assert (buffer.sample(120)["observation"]["state"] < 100).all()


@pytest.mark.parametrize("device", DEVICES)
def test_loading_into_populated_replay_invalidates_cached_metadata(tmp_path, device):
    def make_buffer():
        return ReplayBuffer(60, 3, device=device, output_key_t=("observation",),
                            output_key_tp1=("observation",))
    source, dest = make_buffer(), make_buffer()
    source.extend(records(0, 17, device))
    dest.extend(records(100, 8, device))
    previous = dest._sampling_metadata(1)
    path = tmp_path / "replay"
    save_replay(source, path, rank=0, world_size=1, env_iterations=17)
    load_replay(dest, path, rank=0, world_size=1, env_iterations=17)
    assert dest._sampling_metadata(1) is not previous
    for seq_length in (1, 2, 3):
        torch.manual_seed(37)
        expected = source.sample(120, seq_length)
        torch.manual_seed(37)
        torch.testing.assert_close(dest.sample(120, seq_length), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_depth_restore_reuses_indices_without_scalar_readback(device):
    buffer = ReplayBuffer(60, 3, device=device, output_key_t=("observation",),
                          output_key_tp1=("observation",), compact_depth_history=True,
                          depth_history_offsets=(3, 1, 0))
    data = records(0, 12, device)
    data["observation"]["depth_image"] = torch.zeros(12, 3, 3, 2, 2, dtype=torch.uint8, device=device)
    buffer.extend(data)
    offsets, lookback = buffer._depth_offsets, buffer._depth_lookback
    indices = (torch.tensor([5, 8], device=device), torch.tensor([0, 2], device=device))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        buffer._restore_depth({}, indices)
        buffer._restore_depth({}, indices)
    assert buffer._depth_offsets is offsets
    assert buffer._depth_lookback is lookback
    assert not {"aten::item", "aten::_local_scalar_dense", "aten::arange"} & {
        event.key for event in prof.key_averages()
    }
