# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the storage."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 6
NUM_STEPS = 8
OBS_DIM = 4
NUM_ACTIONS = 3


def _make_storage_and_obs() -> tuple:
    """Create a storage and observations for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return storage, obs


def _fill_with_identifiable_data(storage: RolloutStorage, obs: TensorDict) -> dict:
    """Fill storage with data where each transition has a unique identifier.

    Returns a dict mapping step to the stored values for verification.
    """
    stored = {}
    for step in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        # Use step-based values so each transition is identifiable
        t.actions = torch.full((NUM_ENVS, NUM_ACTIONS), float(step))
        t.values = torch.full((NUM_ENVS, 1), float(step) * 10)
        t.actions_log_prob = torch.full((NUM_ENVS,), float(step) * 0.1)
        t.distribution_params = (
            torch.full((NUM_ENVS, NUM_ACTIONS), float(step)),
            torch.full((NUM_ENVS, NUM_ACTIONS), 1.0),
        )
        t.rewards = torch.full((NUM_ENVS,), float(step) * 100)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)
        stored[step] = {
            "actions": t.actions.clone(),
            "values": t.values.clone(),
            "rewards": t.rewards.clone(),
        }

    # Set returns and advantages for mini-batch testing
    storage.returns = torch.randn_like(storage.returns)
    storage.advantages = torch.randn_like(storage.advantages)

    return stored


class TestMiniBatchGenerator:
    """Tests for ``mini_batch_generator`` data completeness."""

    def test_all_transitions_visited_in_one_epoch(self) -> None:
        """Every transition should appear exactly once in a single epoch."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 4
        batch_size = NUM_ENVS * NUM_STEPS
        mini_batch_size = batch_size // num_mini_batches

        all_actions = []
        batch_count = 0
        for batch in storage.mini_batch_generator(num_mini_batches, num_epochs=1):
            all_actions.append(batch.actions)
            batch_count += 1

        assert batch_count == num_mini_batches

        collected = torch.cat(all_actions, dim=0)
        expected_count = num_mini_batches * mini_batch_size
        assert collected.shape[0] == expected_count

        # Check all unique action values from the stored transitions appear
        flat_stored = storage.actions.flatten(0, 1)
        for val in flat_stored[:expected_count]:
            assert torch.any(torch.all(collected == val.unsqueeze(0), dim=-1))

    def test_epoch_count_is_respected(self) -> None:
        """The generator should yield exactly num_epochs * num_mini_batches batches."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 3
        num_epochs = 5
        batch_count = sum(1 for _ in storage.mini_batch_generator(num_mini_batches, num_epochs))

        assert batch_count == num_mini_batches * num_epochs

    def test_batch_fields_are_consistent(self) -> None:
        """All fields within a batch should index the same transitions."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        for batch in storage.mini_batch_generator(2, num_epochs=1):
            # actions and values should be from the same transitions
            # We stored actions = step, values = step * 10
            # So values / 10 should approximately equal actions (for each step)
            action_ids = batch.actions[:, 0]  # All actions in a step are the same value
            value_ids = batch.values[:, 0] / 10.0
            assert torch.allclose(action_ids, value_ids, atol=1e-5), (
                "Actions and values should index the same transitions"
            )


class TestCompactDepthHistory:
    """Tests for reconstructing depth stacks from one stored frame per rollout step."""

    NUM_ENVS = 2
    NUM_STEPS = 4
    OFFSETS = (2, 1, 0)

    @staticmethod
    def _observation(states: list[int], depth_stacks: list[list[int]]) -> TensorDict:
        num_envs = len(states)
        return TensorDict(
            {
                "policy": TensorDict(
                    {
                        "state": torch.tensor(states, dtype=torch.float32).reshape(num_envs, 1),
                        "depth_image": torch.tensor(depth_stacks, dtype=torch.uint8).reshape(
                            num_envs, len(TestCompactDepthHistory.OFFSETS), 1, 1
                        ),
                    },
                    batch_size=[num_envs],
                ),
                "critic": torch.tensor(states, dtype=torch.float32).reshape(num_envs, 1),
            },
            batch_size=[num_envs],
        )

    @classmethod
    def _storage(cls) -> RolloutStorage:
        obs = cls._observation([0, 1], [[0, 0, 0], [0, 0, 0]])
        return RolloutStorage(
            "rl",
            cls.NUM_ENVS,
            cls.NUM_STEPS,
            obs,
            [NUM_ACTIONS],
            compact_depth_history=True,
            depth_history_offsets=cls.OFFSETS,
        )

    @staticmethod
    def _add(storage: RolloutStorage, obs: TensorDict, dones: list[bool]) -> None:
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.hidden_states = (None, None)
        transition.actions = torch.zeros(TestCompactDepthHistory.NUM_ENVS, NUM_ACTIONS)
        transition.values = torch.zeros(TestCompactDepthHistory.NUM_ENVS, 1)
        transition.actions_log_prob = torch.zeros(TestCompactDepthHistory.NUM_ENVS)
        transition.distribution_params = (
            torch.zeros(TestCompactDepthHistory.NUM_ENVS, NUM_ACTIONS),
            torch.ones(TestCompactDepthHistory.NUM_ENVS, NUM_ACTIONS),
        )
        transition.rewards = torch.zeros(TestCompactDepthHistory.NUM_ENVS)
        transition.dones = torch.tensor(dones)
        storage.add_transition(transition)

    @staticmethod
    def _assert_batches(storage: RolloutStorage, expected: dict[int, list[int]]) -> None:
        storage.returns.zero_()
        storage.advantages.zero_()
        visited = set()
        for batch in storage.mini_batch_generator(num_mini_batches=2, num_epochs=1):
            states = batch.observations["policy", "state"].flatten().long()
            depths = batch.observations["policy", "depth_image"][:, :, 0, 0]
            for state, depth in zip(states.tolist(), depths):
                torch.testing.assert_close(depth, torch.tensor(expected[state], dtype=torch.uint8))
                visited.add(state)
        assert visited == set(expected)

    def test_minibatches_preserve_episode_delay_and_clamp_at_resets(self) -> None:
        storage = self._storage()
        # Env 1 uses delay=1 in its first episode, then delay=0 after its reset.
        first_rollout = (
            ([0, 1], [[10, 10, 10], [20, 20, 20]], [False, False]),
            ([2, 3], [[10, 10, 11], [20, 20, 20]], [False, True]),
            ([4, 5], [[10, 11, 12], [100, 100, 100]], [False, False]),
            ([6, 7], [[11, 12, 13], [100, 100, 101]], [False, False]),
        )
        for states, depth, dones in first_rollout:
            self._add(storage, self._observation(states, depth), dones)

        compact = storage.observations["policy", "depth_image"]
        assert compact.shape == (self.NUM_STEPS, self.NUM_ENVS, 1, 1)
        self._assert_batches(
            storage,
            {
                0: [10, 10, 10],
                1: [20, 20, 20],
                2: [10, 10, 11],
                3: [20, 20, 20],
                4: [10, 11, 12],
                5: [100, 100, 100],
                6: [11, 12, 13],
                7: [100, 100, 101],
            },
        )

        storage.clear()
        second_rollout = (
            ([8, 9], [[12, 13, 14], [100, 101, 102]]),
            ([10, 11], [[13, 14, 15], [101, 102, 103]]),
            ([12, 13], [[14, 15, 16], [102, 103, 104]]),
            ([14, 15], [[15, 16, 17], [103, 104, 105]]),
        )
        for states, depth in second_rollout:
            self._add(storage, self._observation(states, depth), [False, False])
        expected = {
            state: stack
            for states, depth in second_rollout
            for state, stack in zip(states, depth)
        }
        self._assert_batches(storage, expected)

    def test_prefix_can_span_more_than_one_rollout(self) -> None:
        offsets = (5, 2, 0)
        obs = self._observation([0, 1], [[0, 0, 0], [0, 0, 0]])
        storage = RolloutStorage(
            "rl",
            self.NUM_ENVS,
            self.NUM_STEPS,
            obs,
            [NUM_ACTIONS],
            compact_depth_history=True,
            depth_history_offsets=offsets,
        )
        for rollout in range(3):
            expected = {}
            for local_step in range(self.NUM_STEPS):
                step = rollout * self.NUM_STEPS + local_step
                states = [step * self.NUM_ENVS + env for env in range(self.NUM_ENVS)]
                depth = [
                    [max(0, step - offset) + env * 100 for offset in offsets]
                    for env in range(self.NUM_ENVS)
                ]
                self._add(storage, self._observation(states, depth), [False, False])
                expected.update(zip(states, depth))
            self._assert_batches(storage, expected)
            if rollout < 2:
                storage.clear()

    def test_minibatches_reuse_episode_start_metadata(self) -> None:
        storage = self._storage()
        for step in range(self.NUM_STEPS):
            states = [step * self.NUM_ENVS + env for env in range(self.NUM_ENVS)]
            depth = [[max(0, step - offset) for offset in self.OFFSETS] for _ in range(self.NUM_ENVS)]
            self._add(storage, self._observation(states, depth), [False, False])
        storage.returns.zero_()
        storage.advantages.zero_()

        original_restore = storage._restore_depth
        starts_seen = []

        def record_starts(observations, time_idx, env_idx, *, starts=None):
            starts_seen.append(starts)
            return original_restore(observations, time_idx, env_idx, starts=starts)

        storage._restore_depth = record_starts
        list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=3))

        assert len(starts_seen) == 6
        assert starts_seen[0] is not None
        assert all(starts is starts_seen[0] for starts in starts_seen)

    def test_invalid_depth_contract_is_rejected(self) -> None:
        obs = self._observation([0, 1], [[0, 0, 0], [0, 0, 0]])
        obs["policy", "depth_image"] = obs["policy", "depth_image"].float()
        with pytest.raises(ValueError, match="uint8"):
            RolloutStorage(
                "rl",
                self.NUM_ENVS,
                self.NUM_STEPS,
                obs,
                [NUM_ACTIONS],
                compact_depth_history=True,
                depth_history_offsets=self.OFFSETS,
            )


class TestRecurrentMiniBatchGenerator:
    """Tests for ``recurrent_mini_batch_generator`` — trajectory counting, env/trajectory alignment."""

    def test_trajectory_count_matches_env_slice(self) -> None:
        """The number of trajectories in each batch should match dones in the corresponding env slice."""
        storage, obs = _make_storage_and_obs()

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == 3:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        mini_batch_size = NUM_ENVS // 2
        batches = list(storage.recurrent_mini_batch_generator(2, num_epochs=1))
        # Batch 0 covers envs [0..2]: env 0 has 2 trajectories, envs 1-2 have 1 each → 4 total
        assert batches[0].observations["policy"].shape[1] == mini_batch_size + 1
        # Batch 1 covers envs [3..5]: no dones → 3 trajectories (one each)
        assert batches[1].observations["policy"].shape[1] == mini_batch_size

    def test_obs_and_actions_refer_to_same_env_slice(self) -> None:
        """Trajectory-indexed obs and env-indexed actions must refer to the same environments, even with dones."""
        storage, _obs = _make_storage_and_obs()

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = TensorDict(
                {"policy": torch.arange(NUM_ENVS).float().unsqueeze(1).expand(-1, OBS_DIM)},
                batch_size=[NUM_ENVS],
            )
            t.hidden_states = (None, None)
            t.actions = torch.arange(NUM_ENVS).float().unsqueeze(1).expand(-1, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == 3:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        for batch in storage.recurrent_mini_batch_generator(2, num_epochs=1):
            obs_vals = batch.observations["policy"]  # [time, num_trajs, obs_dim]
            masks = batch.masks  # [time, num_trajs]
            obs_env_ids = set(obs_vals[masks][:, 0].unique().long().tolist())

            action_env_ids = set(batch.actions[0, :, 0].unique().long().tolist())

            assert obs_env_ids == action_env_ids, f"Obs env ids {obs_env_ids} != action env ids {action_env_ids}"

    def test_epoch_count_is_respected(self) -> None:
        """Generator should yield exactly num_mini_batches * num_epochs batches."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 3
        num_epochs = 5
        batch_count = sum(1 for _ in storage.recurrent_mini_batch_generator(num_mini_batches, num_epochs))
        assert batch_count == num_mini_batches * num_epochs

    def test_hidden_states_align_with_trajectory_starts(self) -> None:
        """Hidden states in each batch should correspond to trajectory start points, including after dones."""
        storage, obs = _make_storage_and_obs()
        rnn_hidden_dim = 8
        done_step = 3

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            h_actor = torch.full((1, NUM_ENVS, rnn_hidden_dim), float(step))
            t.hidden_states = (h_actor, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == done_step:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        batches = list(storage.recurrent_mini_batch_generator(2, num_epochs=1))

        # Batch 0 covers envs [0,1,2]. Env 0 has 2 trajectories (start at step 0, restart at step 4).
        # Trajectory order from boolean indexing: env0-start0, env0-start4, env1-start0, env2-start0
        h0 = batches[0].hidden_states[0]
        assert h0 is not None
        assert h0.shape[1] == 4, f"Expected 4 trajectories, got {h0.shape[1]}"
        assert h0[0, 0, 0].item() == 0.0, "Env 0 first trajectory should start with step-0 hidden state"
        assert h0[0, 1, 0].item() == float(done_step + 1), (
            f"Env 0 second trajectory should start with step-{done_step + 1} hidden state"
        )

        # Batch 1 covers envs [3,4,5]. No dones → all hidden states from step 0.
        h1 = batches[1].hidden_states[0]
        assert h1 is not None
        assert torch.allclose(h1, torch.zeros_like(h1)), "Envs without dones should all have step-0 hidden states"


class TestDistillationStorage:
    """Tests for distillation-mode storage."""

    def test_generator_yields_per_timestep_batches(self) -> None:
        """Distillation generator should yield one batch per timestep."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.privileged_actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)

        batch_count = sum(1 for _ in storage.generator())
        assert batch_count == NUM_STEPS

    def test_rl_storage_rejects_distillation_generator(self) -> None:
        """An RL-mode storage should not allow the distillation generator."""
        storage, _obs = _make_storage_and_obs()
        with pytest.raises(ValueError):
            next(storage.generator())


class TestStorageOverflow:
    """Tests for storage capacity enforcement."""

    def test_overflow_raises_error(self) -> None:
        """Adding more transitions than capacity should raise OverflowError."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        with pytest.raises(OverflowError):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)


class TestStorageClear:
    """Tests for the clear() method."""

    def test_clear_allows_new_transitions(self) -> None:
        """After clear(), storage should accept new transitions starting from index 0."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        storage.clear()
        assert storage.step == 0

        # Should not raise
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.randn(NUM_ENVS)
        t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)
        assert storage.step == 1

    def test_clear_data_is_overwritten(self) -> None:
        """New data written after clear() should replace old data at the same index."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        old_actions_step0 = storage.actions[0].clone()
        storage.clear()

        new_actions = torch.full((NUM_ENVS, NUM_ACTIONS), 999.0)
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = new_actions
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.randn(NUM_ENVS)
        t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)

        assert torch.allclose(storage.actions[0], new_actions)
        assert not torch.allclose(storage.actions[0], old_actions_step0)
