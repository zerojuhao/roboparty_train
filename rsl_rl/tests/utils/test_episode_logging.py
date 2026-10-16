"""Buffered episode logging preserves the original order, windows and reset sums."""

from unittest.mock import Mock

import pytest
import torch

from rsl_rl.utils.logger import Logger


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_logger(device, rnd=False, steps=24, num_envs=32):
    logger = Logger(None, {"algorithm": {"rnd_cfg": {"weight": 1.0} if rnd else None},
                           "num_steps_per_env": steps}, {}, num_envs, False, 1, 0, device)
    logger.writer = Mock()
    return logger


def legacy_step(logger, rewards, dones, intrinsic=None):
    if intrinsic is not None:
        logger.cur_ereward_sum += rewards
        logger.cur_ireward_sum += intrinsic
        logger.cur_reward_sum += rewards + intrinsic
    else:
        logger.cur_reward_sum += rewards
    logger.cur_episode_length += 1
    ids = (dones > 0).nonzero(as_tuple=False)
    pairs = [(logger.cur_reward_sum, logger.rewbuffer), (logger.cur_episode_length, logger.lenbuffer)]
    if intrinsic is not None:
        pairs += [(logger.cur_ereward_sum, logger.erewbuffer), (logger.cur_ireward_sum, logger.irewbuffer)]
    for sums, dest in pairs:
        dest.extend(sums[ids][:, 0].cpu().numpy().tolist())
        sums[ids] = 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rnd", [False, True])
def test_episode_statistics_match_legacy_with_overflow_empty_steps_and_partial_rollouts(device, rnd):
    torch.manual_seed(31)
    actual, reference = make_logger(device, rnd), make_logger(device, rnd)
    fields = ["rewbuffer", "lenbuffer"] + (["erewbuffer", "irewbuffer"] if rnd else [])
    sums = ["cur_reward_sum", "cur_episode_length"] + (["cur_ereward_sum", "cur_ireward_sum"] if rnd else [])
    for step in range(79):
        rewards = torch.randn(32, device=device)
        dones = torch.rand(32, device=device) < .25
        if step < 5:
            dones.zero_()
        if step % 7 == 0 and step > 5:
            dones.fill_(True)
        intrinsic = torch.rand(32, device=device) if rnd and step % 3 else None
        legacy_step(reference, rewards, dones, intrinsic)
        actual.process_env_step(rewards, dones, {}, intrinsic)
        for name in sums:
            torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
        assert len(actual._pending_episode_stats) < 24
        assert all(not values.requires_grad for values in actual._pending_episode_stats)
        if step % 24 == 23 or step in (7, 78):
            actual._flush_episode_stats()
            for name in fields:
                assert list(getattr(actual, name)) == list(getattr(reference, name))
    actual._flush_episode_stats()
    for name in fields:
        assert list(getattr(actual, name)) == list(getattr(reference, name))


@pytest.mark.parametrize("device", DEVICES)
def test_nonflush_steps_have_no_scalar_readback_or_nonzero(device):
    logger = make_logger(device)
    rewards = torch.ones(32, device=device)
    dones = torch.zeros(32, device=device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        logger.process_env_step(rewards, dones, {})
    operations = {event.key for event in prof.key_averages()}
    assert not {"aten::item", "aten::_local_scalar_dense", "aten::nonzero"} & operations


def test_log_flushes_partial_rollout_and_disabled_logging_does_not_allocate(capsys):
    logger = make_logger("cpu", num_envs=2)
    logger.process_env_step(torch.tensor([1., 2.]), torch.ones(2), {})
    assert not logger.rewbuffer
    logger.log(0, 0, 1, .1, .1, {}, .001, torch.ones(2), None)
    assert list(logger.rewbuffer) == [1., 2.]
    assert not logger._pending_episode_stats
    logger.writer.add_scalar.assert_any_call("Train/mean_reward", 1.5, 0)
    logger.writer = None
    logger.process_env_step(torch.ones(2), torch.ones(2), {})
    assert not logger._pending_episode_stats


@pytest.mark.parametrize("device", DEVICES)
def test_extras_preserve_sample_weighting_missing_keys_and_input_values(device):
    logger = make_logger(device)
    first = {"Episode_Reward/test": torch.tensor(2., device=device),
             "Braking/conditional": torch.tensor([1., 3., 5.], device=device),
             "integer": torch.tensor([1, 2], device=device), "python": 3.0,
             "empty": torch.empty(0, device=device),
             "half": torch.tensor([1., 3.], device=device, dtype=torch.float16)}
    second = {"Episode_Reward/test": torch.tensor([4., 6.], device=device),
              "Braking/conditional": torch.tensor([9.], device=device),
              "integer": torch.tensor(3, device=device), "python": 7.0}
    logger.ep_extras = [first, second]
    result = logger._episode_extras_means()
    assert list(result) == list(first)
    assert result["Episode_Reward/test"] == 4.0
    assert result["Braking/conditional"] == 4.5
    assert result["integer"] == 2.0
    assert result["python"] == 5.0
    assert result["half"] == 2.0
    assert torch.isnan(torch.tensor(result["empty"]))
    assert first["Episode_Reward/test"].ndim == 0
    assert second["integer"].ndim == 0
    assert isinstance(first["python"], float)
    assert make_logger(device)._episode_extras_means() == {}


@pytest.mark.parametrize("device", DEVICES)
def test_extras_batch_host_reads_and_concatenate_once_per_metric(device):
    logger = make_logger(device)
    logger.ep_extras = [{f"metric/{i}": torch.ones(3, device=device) * step for i in range(8)}
                        for step in range(24)]
    logger._episode_extras_means()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        result = logger._episode_extras_means()
    assert set(result.values()) == {11.5}
    counts = {event.key: event.count for event in prof.key_averages()}
    assert counts.get("aten::_local_scalar_dense", 0) == 0
    # stack itself is implemented as one cat, in addition to one cat per metric.
    assert counts.get("aten::cat", 0) == 9


def test_log_keeps_all_episode_tags_and_console_values(capsys):
    logger = make_logger("cpu")
    logger.ep_extras = [{"Episode_Reward/custom": torch.tensor([2., 4.]), "other": 7.}]
    logger.log(0, 0, 1, .1, .1, {}, .001, torch.ones(2), None)
    logger.writer.add_scalar.assert_any_call("Episode_Reward/custom", 3.0, 0)
    logger.writer.add_scalar.assert_any_call("Episode/other", 7.0, 0)
    output = capsys.readouterr().out
    assert "Episode_Reward/custom:" in output and "3.0000" in output
    assert "Mean episode other:" in output and "7.0000" in output
    assert logger.ep_extras == []
