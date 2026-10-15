"""Tests for action-outcome alignment in the FB-CPR replay buffer."""

import torch
import pytest

from rsl_rl.storage.replay_buffer import ReplayBuffer, TrajectoryDictBuffer


def _expert_episode(motion_id: int, length: int = 8) -> dict:
    truncated = torch.zeros(length, 1, dtype=torch.bool)
    truncated[-1] = True
    return {
        "motion_id": torch.full((length, 1), motion_id, dtype=torch.long),
        "observation": {"state": torch.full((length, 1), float(motion_id))},
        "truncated": truncated,
    }


def test_expert_trajectory_priorities_control_sampling() -> None:
    buffer = TrajectoryDictBuffer(
        [_expert_episode(0), _expert_episode(1)],
        output_key_t=("motion_id",), output_key_tp1=(),
    )
    buffer.update_priorities(torch.tensor([1.0, 8.0]))
    torch.manual_seed(7)
    sampled = buffer.sample(90_000)["motion_id"].flatten()
    fractions = torch.bincount(sampled, minlength=2).float() / sampled.numel()
    torch.testing.assert_close(fractions, torch.tensor([1 / 9, 8 / 9]), atol=0.01, rtol=0)


def test_expert_trajectory_priorities_validate_inputs_atomically() -> None:
    buffer = TrajectoryDictBuffer(
        [_expert_episode(0), _expert_episode(1)],
        output_key_t=("motion_id",), output_key_tp1=(),
    )
    original = buffer.priorities.clone()
    for invalid in (torch.tensor([1.0, float("nan")]), torch.tensor([1.0, -1.0])):
        try:
            buffer.update_priorities(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid expert priorities were accepted")
        torch.testing.assert_close(buffer.priorities, original)


def test_transition_outcome_is_returned_with_next_observation() -> None:
    """Next samples carry the outcome belonging to the sampled action."""
    buffer = ReplayBuffer(
        capacity=8,
        num_envs=1,
        device="cpu",
        output_key_t=("observation", "action", "z", "aux_rewards"),
        output_key_tp1=("observation", "transition_terminated", "transition_truncated"),
        end_key="transition_truncated",
    )
    transitions = {
        "observation": {"state": torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)},
        "action": torch.zeros(4, 1, 1),
        "z": torch.zeros(4, 1, 1),
        "aux_rewards": {"aux_reward": torch.zeros(4, 1, 1)},
        "truncated": torch.tensor([[[False]], [[False]], [[True]], [[False]]]),
        "transition_terminated": torch.tensor([[[False]], [[True]], [[False]], [[False]]]),
        "transition_truncated": torch.tensor([[[False]], [[False]], [[True]], [[False]]]),
    }
    buffer.extend(transitions)

    # The replay sampler advances the time index for all next fields.  The
    # next observation at t=0 therefore carries the outcome of action t=1,
    # which is exactly the alignment consumed by the critic target.
    sampled = buffer.sample(batch_size=1, seq_length=1)
    state = int(sampled["observation"]["state"].item())
    next_state = int(sampled["next"]["observation"]["state"].item())
    assert next_state == (state + 1) % 4
    assert torch.equal(
        sampled["next"]["transition_terminated"], transitions["transition_terminated"][next_state]
    )
    assert torch.equal(
        sampled["next"]["transition_truncated"], transitions["transition_truncated"][next_state]
    )


def test_online_replay_samples_valid_windows_uniformly_across_trajectories() -> None:
    buffer = ReplayBuffer(
        capacity=30,
        num_envs=2,
        device="cpu",
        output_key_t=("observation",),
        output_key_tp1=("observation",),
    )
    truncated = torch.zeros(15, 2, 1, dtype=torch.bool)
    truncated[2::3, 0] = True
    truncated[-1, 1] = True
    env_id = torch.arange(2).view(1, 2, 1).expand(15, -1, -1)
    buffer.extend({"observation": {"env_id": env_id}, "truncated": truncated})

    torch.manual_seed(11)
    sampled = buffer.sample(50_000)
    env_one_fraction = sampled["observation"]["env_id"].float().mean()
    # Five length-3 trajectories contribute two valid pairs each; the
    # length-15 trajectory contributes fourteen.
    torch.testing.assert_close(env_one_fraction, torch.tensor(14 / 24), atol=0.01, rtol=0)

    diagnostics = buffer.diagnostics()
    assert diagnostics["replay/trajectory_count"].item() == 6
    assert diagnostics["replay/trajectory_length_min"].item() == 3
    assert diagnostics["replay/valid_window_count"].item() == 24


def test_online_replay_keeps_single_valid_window_trajectory_sampleable() -> None:
    buffer = ReplayBuffer(
        capacity=4,
        num_envs=1,
        device="cpu",
        output_key_t=("observation",),
        output_key_tp1=("observation",),
    )
    buffer.extend({
        "observation": {"state": torch.arange(2).reshape(2, 1, 1)},
        "truncated": torch.tensor([[[False]], [[True]]]),
    })

    sampled = buffer.sample(8)
    torch.testing.assert_close(sampled["observation"]["state"], torch.zeros(8, 1, dtype=torch.long))
    torch.testing.assert_close(sampled["next"]["observation"]["state"], torch.ones(8, 1, dtype=torch.long))
    assert buffer.diagnostics()["replay/valid_window_count"].item() == 1


@pytest.mark.parametrize("length", [1, 2, 5])
@pytest.mark.parametrize("compact", [False, True])
def test_autoreset_preserves_every_action_outcome(length, compact):
    buffer = ReplayBuffer(
        capacity=20, num_envs=1, output_key_t=("observation", "action"),
        output_key_tp1=("observation", "terminated", "truncated"),
        end_key="episode_end", transition_outcomes=True,
        compact_depth_history=compact, depth_history_offsets=(1, 0),
    )
    for t in range(length + 2):
        state = t if t < length else 100 + t - length
        end = t == length - 1
        obs = {"state": torch.tensor([[[state]]])}
        if compact:
            start = 0 if t < length else length
            obs["depth_image"] = torch.tensor([max(start, t - 1), t], dtype=torch.uint8).reshape(1, 1, 2, 1, 1)
        flags = torch.tensor([[[end]]])
        final = {"state": torch.tensor([[[length]]])}
        if compact:
            final["depth_image"] = torch.tensor([length - 1, length], dtype=torch.uint8).reshape(1, 1, 2, 1, 1)
        buffer.extend({"observation": obs, "action": torch.tensor([[[state]]]),
                       "episode_end": flags, "terminated": torch.zeros_like(flags), "truncated": flags,
                       "depth_episode_start": torch.tensor([[[t in (0, length)]]])},
                      final_observation=final if end else None)
    torch.manual_seed(1)
    sampled = buffer.sample(4096)
    state = sampled["observation"]["state"].flatten()
    nxt = sampled["next"]["observation"]["state"].flatten()
    assert set(state.tolist()) == set(range(length)) | {100}
    torch.testing.assert_close(nxt, state + 1)
    torch.testing.assert_close(sampled["next"]["truncated"].flatten(), state == length - 1)
    assert not sampled["next"]["terminated"].any()
    if compact:
        final_depth = sampled["next"]["observation"]["depth_image"][state == length - 1]
        torch.testing.assert_close(final_depth[:, :, 0, 0], torch.tensor([length - 1, length], dtype=torch.uint8).expand(len(final_depth), -1))


def test_last_valid_expert_pair_is_sampleable():
    buffer = TrajectoryDictBuffer([{"observation": {"state": torch.arange(4)[:, None]},
                                    "motion_id": torch.zeros(4, 1, dtype=torch.long),
                                    "truncated": torch.tensor([[False], [False], [False], [True]])}])
    torch.manual_seed(2)
    sampled = buffer.sample(2048)
    assert set(sampled["observation"]["state"].flatten().tolist()) == {0, 1, 2}
    torch.testing.assert_close(sampled["next"]["observation"]["state"], sampled["observation"]["state"] + 1)


def test_terminal_action_is_available_before_reset_record_and_missing_outcomes_fail():
    buffer = ReplayBuffer(8, 1, output_key_t=("observation",),
                          output_key_tp1=("observation", "terminated"),
                          end_key="episode_end", transition_outcomes=True)
    data = {"observation": {"state": torch.tensor([[[10]]])},
            "truncated": torch.tensor([[[False]]]), "terminated": torch.tensor([[[True]]]),
            "episode_end": torch.tensor([[[True]]])}
    with pytest.raises(ValueError, match="pre-reset final"):
        buffer.extend(data)
    buffer.extend(data, final_observation={"state": torch.tensor([[[11]]])})
    sampled = buffer.sample(8)
    assert (sampled["next"]["observation"]["state"] == 11).all()
    assert sampled["next"]["terminated"].all()
    assert buffer.diagnostics()["replay/valid_window_count"].item() == 1


def test_resume_gap_keeps_known_pairs_and_excludes_unfinished_action():
    buffer = ReplayBuffer(8, 1, output_key_t=("observation",), output_key_tp1=("observation",),
                          end_key="episode_end", transition_outcomes=True)
    for states in ([0, 1], [100, 101]):
        data = {"observation": {"state": torch.tensor(states).reshape(2, 1, 1)},
                "truncated": torch.zeros(2, 1, 1, dtype=torch.bool),
                "episode_end": torch.zeros(2, 1, 1, dtype=torch.bool)}
        buffer.extend(data)
        buffer.close_trajectory()
    sampled = buffer.sample(512)
    assert set(sampled["observation"]["state"].flatten().tolist()) == {0, 100}
    torch.testing.assert_close(sampled["next"]["observation"]["state"], sampled["observation"]["state"] + 1)
