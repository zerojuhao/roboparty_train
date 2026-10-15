"""Dict replay buffers, ported from UFO `humanoidverse/agents/buffers/`.

`ReplayBuffer` stores time-major ``(time, num_envs, ...)`` data.  Keeping the
environment axis intact is essential: the next observation for ``(t, env)``
is ``(t + 1, env)``, never another environment from the same simulator step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List

import numpy as np
import torch


def _tree_map(func, tree):
    if isinstance(tree, Mapping):
        return {k: _tree_map(func, v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_tree_map(func, v) for v in tree)
    return func(tree)


def _to_torch(value, device=None):
    tensor = torch.as_tensor(value)
    if device:
        return tensor.to(device=device)
    return tensor


class ZBuffer:
    def __init__(self, capacity: int, dim: int, device, dtype=torch.float32):
        self._storage = torch.zeros((capacity, dim), device=device, dtype=dtype)
        self._idx = 0
        self._is_full = False
        self.capacity = capacity
        self.device = device

    def __len__(self):
        return self.capacity if self._is_full else self._idx

    def empty(self):
        return self._idx == 0 and not self._is_full

    def add(self, data: torch.Tensor):
        if self._idx + data.shape[0] >= self.capacity:
            diff = self.capacity - self._idx
            self._storage[self._idx : self._idx + data.shape[0]] = data[:diff]
            self._storage[: data.shape[0] - diff] = data[diff:]
            self._is_full = True
        else:
            self._storage[self._idx : self._idx + data.shape[0]] = data
        self._idx = (self._idx + data.shape[0]) % self.capacity

    def sample(self, num, device=None):
        idx = np.random.randint(0, len(self), size=num)
        return self._storage[idx].clone().to(device if device is not None else self.device)


class TrajectoryDictBuffer:
    """Static buffer over a list of full episodes (used for expert data)."""

    def __init__(self, episodes: List[dict], device="cpu", seq_length: int = 1,
                 output_key_t=("observation",), output_key_tp1=("observation",), end_key="truncated"):
        self.seq_length = seq_length
        self.device = device
        self.output_key_t = list(output_key_t)
        self.output_key_tp1 = list(output_key_tp1)
        self.end_key = end_key

        self.storage: Dict = {}
        self.motion_ids = []

        def add_new_data(data, storage):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    storage.setdefault(k, {})
                    add_new_data(v, storage[k])
                else:
                    storage.setdefault(k, []).append(_to_torch(v, device=self.device))

        for ep in episodes:
            add_new_data(ep, self.storage)
            self.motion_ids.append(ep["motion_id"][0].item())

        def concat_dict(storage):
            for k, v in storage.items():
                if isinstance(v, Mapping):
                    concat_dict(v)
                else:
                    storage[k] = torch.cat(v, dim=0)

        concat_dict(self.storage)

        done = self.storage[self.end_key].squeeze(-1) if self.storage[self.end_key].ndim == 2 else self.storage[self.end_key]
        self.start_idx, self.stop_idx, self.lengths = _find_start_stop_traj(done, at_capacity=True, cursor=None)
        self.priorities = torch.ones_like(self.lengths, dtype=torch.float32, device=self.lengths.device)
        self.file_names = []
        self._sampling_cache = {}

    @property
    def capacity(self):
        return self.storage[self.end_key].shape[0]

    def __len__(self):
        return self.capacity

    def _sampling_metadata(self, seq_length):
        offset = int(bool(self.output_key_tp1))
        key = (seq_length, offset)
        if key not in self._sampling_cache:
            trajectory_ids = (self.lengths >= seq_length + offset).nonzero().flatten()
            lengths = self.lengths[trajectory_ids]
            self._sampling_cache[key] = (
                lengths, self.start_idx[trajectory_ids],
                lengths - seq_length - offset + 1, trajectory_ids,
            )
        return self._sampling_cache[key]

    def sample(self, batch_size: int, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        assert batch_size % seq_length == 0
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        lengths, starts, valid_windows, trajectory_ids = self._sampling_metadata(seq_length)
        idxs = _get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=lengths,
            start_idx=starts,
            storage_length=self.capacity,
            # Read live weights so evaluation-driven priority updates take effect immediately.
            priorities=self.priorities[trajectory_ids],
            valid_windows=valid_windows,
        )
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = _tree_map(lambda x: x[idxs], self.storage[k])
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            output["next"][k] = _tree_map(lambda x: x[idxs], self.storage[k])
        return output

    def update_priorities(self, priorities, idxs=None) -> None:
        """Update trajectory sampling weights without changing within-trajectory sampling."""
        values = torch.as_tensor(
            priorities, dtype=self.priorities.dtype, device=self.priorities.device
        ).reshape(-1)
        if idxs is None:
            idxs = torch.arange(len(self.priorities), device=self.priorities.device)
        else:
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=self.priorities.device).reshape(-1)
        if values.numel() != idxs.numel():
            raise ValueError(
                f"Expected one priority per trajectory index, got {values.numel()} and {idxs.numel()}"
            )
        if idxs.numel() and (int(idxs.min()) < 0 or int(idxs.max()) >= len(self.priorities)):
            raise IndexError("Trajectory priority index is out of range")
        if not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError("Trajectory priorities must be finite and non-negative")
        updated = self.priorities.clone()
        updated[idxs] = values
        if float(updated.sum()) <= 0.0:
            raise ValueError("At least one trajectory priority must be positive")
        self.priorities.copy_(updated)


class ReplayBuffer:
    """Time-major ring-buffer matching UFO's TrajectoryDictBufferMultiDim."""

    def __init__(self, capacity: int, num_envs: int, device="cpu",
                 output_key_t=("observation", "action", "z", "aux_rewards"),
                 output_key_tp1=("observation", "terminated"), end_key="truncated",
                 compact_depth_history: bool = False,
                 depth_history_offsets=(35, 30, 25, 20, 15, 10, 5, 0),
                 transition_outcomes: bool = False):
        if capacity < 2 * num_envs:
            raise ValueError("replay capacity must hold at least two vector-environment steps")
        self.transition_capacity = capacity
        self.capacity = capacity // num_envs
        self.num_envs = num_envs
        self.device = device
        self.output_key_t = list(output_key_t)
        self.output_key_tp1 = list(output_key_tp1)
        self.end_key = end_key
        self.transition_outcomes = transition_outcomes
        self.terminal_observations = {}
        self.compact_depth_history = compact_depth_history
        self.depth_history_offsets = tuple(depth_history_offsets) if compact_depth_history else ()
        if compact_depth_history:
            offsets = self.depth_history_offsets
            if (not offsets or offsets[-1] != 0 or any(type(x) is not int or x < 0 for x in offsets)
                    or any(a <= b for a, b in zip(offsets, offsets[1:]))):
                raise ValueError("Depth history offsets must be descending nonnegative integers ending in zero")
            if self.capacity < offsets[0] + 3:
                raise ValueError("Replay capacity must hold depth history and a sampleable transition")
            self._depth_offsets = torch.tensor(offsets, device=device)
            self._depth_lookback = torch.arange(offsets[0] + 1, device=device)
        self._depth_new_segment = True
        self.storage = None
        self._idx = 0
        self._is_full = False
        self._recompute = True
        self._sampling_cache = {}

    def __len__(self):
        return self.capacity if self._is_full else self._idx

    def size(self):
        return len(self) * self.num_envs

    def empty(self):
        return len(self) == 0

    def close_trajectory(self):
        """Prevent the next append from joining a previous simulator instance."""
        if not self.empty():
            self.storage[self.end_key][(self._idx - 1) % self.capacity] = True
            self._recompute = True
        self._depth_new_segment = True

    @torch.no_grad()
    def extend(self, data: Dict, *, final_observation=None) -> None:
        """Append one or more time-major vector-environment observations.

        Every tensor leaf must have shape ``(time, num_envs, ...)``.
        In transition-outcome mode, flags belong to the stored action. Only
        reset outcomes need separate observations; other outcomes share the
        following record. This avoids duplicating the large visual replay.
        """
        n = data["truncated"].shape[0]
        if not 0 < n <= self.capacity:
            raise ValueError("Replay append must contain between one and capacity time steps")
        if self.transition_outcomes:
            ends = data[self.end_key].bool()
            if final_observation is None and bool(ends.any()):
                raise ValueError("Reset transitions require pre-reset final observations")
            data = {**data, "terminal_valid": ends.clone()}
            for offset in range(n):
                time = (self._idx + offset) % self.capacity
                self.terminal_observations.pop(time, None)
                ids = ends[offset].reshape(-1).nonzero().flatten()
                if ids.numel():
                    self.terminal_observations[time] = {
                        "env_ids": ids.to(self.device),
                        "observation": _tree_map(
                            lambda x: x[offset, ids].detach().to(self.device).clone(), final_observation
                        ),
                    }
        if self.compact_depth_history:
            data = self._compact_depth(data)
        if self.storage is None:
            self.storage = {}
            self._init_storage(data, self.storage, self.capacity, self.device)
        n = data["truncated"].shape[0]
        if data["truncated"].shape[1] != self.num_envs:
            raise ValueError(
                f"expected time-major data with {self.num_envs} environments, "
                f"got shape {tuple(data['truncated'].shape)}"
            )
        end = self._idx + n
        if end >= self.capacity:
            diff = self.capacity - self._idx
            self._copy(data, self.storage, slice(self._idx, self.capacity), 0, diff)
            self._copy(data, self.storage, slice(0, n - diff), diff, n - diff)
            self._is_full = True
        else:
            self._copy(data, self.storage, slice(self._idx, end), 0, n)
        self._idx = end % self.capacity
        self._recompute = True

    def _compact_depth(self, data):
        observation = dict(data["observation"])
        depth = observation["depth_image"]
        if depth.ndim != 5 or depth.shape[2] != len(self.depth_history_offsets) or depth.dtype != torch.uint8:
            raise ValueError("Compact replay requires uint8 depth [time, env, frames, height, width]")
        observation["depth_image"] = depth[:, :, -1]
        # Delay is fixed within an episode, so the newest delayed frame at t-k
        # is exactly the k-old delayed frame at t. Reset padding repeats frame 0.
        starts = data.get("depth_episode_start")
        if starts is None:
            if self.transition_outcomes:
                raise ValueError("Transition-outcome replay requires explicit depth episode starts")
            starts = data.get("transition_truncated", data["truncated"]).bool().clone()
            terminated = data.get("transition_terminated", data.get("terminated"))
            if terminated is not None:
                starts |= terminated.bool()
        else:
            starts = starts.bool().clone()
        if starts.shape != data["truncated"].shape:
            raise ValueError("Depth episode boundaries must match truncated shape")
        if self._depth_new_segment:
            starts[0] = True
        self._depth_new_segment = False
        return {**data, "observation": observation, "depth_episode_start": starts}

    def _restore_depth(self, observation, idxs):
        time_idx, env_idx = idxs
        offsets = self._depth_offsets
        lookback = self._depth_lookback
        times = (time_idx[:, None] - lookback) % self.capacity
        starts = self.storage["depth_episode_start"][times, env_idx[:, None]].reshape(len(time_idx), -1)
        nearest_start = torch.where(starts, lookback, offsets[0]).amin(dim=1)
        available = time_idx if not self._is_full else (time_idx - self._idx) % self.capacity
        available = torch.minimum(available, nearest_start)
        source = (time_idx[:, None] - torch.minimum(offsets, available[:, None])) % self.capacity
        return {**observation, "depth_image": self.storage["observation"]["depth_image"][source, env_idx[:, None]]}

    def _init_storage(self, data, storage, capacity, device):
        for k, v in data.items():
            if isinstance(v, Mapping):
                storage[k] = {}
                self._init_storage(v, storage[k], capacity, device)
            else:
                if v.ndim < 2 or v.shape[1] != self.num_envs:
                    raise ValueError(
                        f"replay leaves must have shape (time, {self.num_envs}, ...), got {tuple(v.shape)}"
                    )
                shape = (capacity, *v.shape[1:])
                storage[k] = torch.zeros(shape, dtype=v.dtype if torch.is_tensor(v) else torch.float32, device=device)

    def _copy(self, data, storage, dest_slice, src_start, n):
        for k, v in data.items():
            if isinstance(v, Mapping):
                self._copy(v, storage[k], dest_slice, src_start, n)
            else:
                if not torch.is_tensor(v):
                    v = _to_torch(v, device=self.device)
                storage[k][dest_slice] = v[src_start : src_start + n]

    def _refresh_traj(self):
        if not self._recompute:
            return
        # Appends, trajectory closure and checkpoint loads all invalidate through _recompute.
        self._sampling_cache.clear()
        done = self.storage[self.end_key]
        if done.shape[-1] == 1:
            done = done.squeeze(-1)
        done = done if self._is_full else done[: self._idx]
        self.start_idx, self.stop_idx, self.lengths = _find_start_stop_traj(
            done, at_capacity=self._is_full, cursor=self._idx - 1
        )
        if self.compact_depth_history and self._is_full:
            # Do not fabricate history after the ring overwrites older frames.
            age = (self.start_idx[:, 0] - self._idx) % self.capacity
            skip = (self.depth_history_offsets[0] - age).clamp_min(0)
            skip = torch.minimum(skip, self.lengths)
            self.start_idx[:, 0] = (self.start_idx[:, 0] + skip) % self.capacity
            self.lengths -= skip
        self._recompute = False

    def _valid_windows(self, seq_length):
        offset = int(bool(self.output_key_tp1))
        windows = self.lengths - seq_length - offset + 1
        if self.transition_outcomes and offset:
            # A saved terminal outcome makes the final action sampleable, even
            # if no subsequent reset record has been appended yet.
            windows += self.storage["terminal_valid"][tuple(self.stop_idx.T)].reshape(-1).long()
        return windows.clamp_min(0)

    def _sampling_metadata(self, seq_length):
        self._refresh_traj()
        key = (seq_length, int(bool(self.output_key_tp1)))
        if key not in self._sampling_cache:
            windows = self._valid_windows(seq_length)
            eligible = windows > 0
            lengths = self.lengths[eligible]
            valid_windows = windows[eligible]
            self._sampling_cache[key] = (
                lengths, self.start_idx[eligible], valid_windows, valid_windows.to(torch.float32),
            )
        return self._sampling_cache[key]

    def _restore_terminal_observations(self, output, idxs):
        times, envs = idxs
        terminal_rows = self.storage["terminal_valid"][idxs].reshape(-1).nonzero().flatten()
        for time in times[terminal_rows].unique().tolist():
            terminal = self.terminal_observations.get(time)
            if terminal is None:
                raise RuntimeError("Replay terminal outcome is missing")
            candidates = terminal_rows[times[terminal_rows] == time]
            matches = envs[candidates, None] == terminal["env_ids"][None, :]
            local_rows, sources = matches.nonzero(as_tuple=True)
            rows = candidates[local_rows]
            if rows.numel():
                def replace(dest, source):
                    for key in dest:
                        if isinstance(dest[key], Mapping):
                            replace(dest[key], source[key])
                        else:
                            dest[key][rows] = source[key][sources]
                replace(output, terminal["observation"])
        return output

    def sample(self, batch_size: int, seq_length: int = 1):
        self._refresh_traj()
        assert batch_size % seq_length == 0
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        lengths, starts, valid_windows, priorities = self._sampling_metadata(seq_length)
        if lengths.numel() == 0:
            raise ValueError("no trajectory long enough to sample")
        idxs, sampled_traj_idx = _get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=lengths,
            start_idx=starts,
            storage_length=self.capacity,
            priorities=priorities,
            return_traj_idx=True,
            valid_windows=valid_windows,
        )
        self._last_sampled_lengths = lengths[sampled_traj_idx]
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = _tree_map(lambda x: x[idxs], self.storage[k])
        if self.compact_depth_history and "observation" in output:
            output["observation"] = self._restore_depth(output["observation"], idxs)
        current_idxs = idxs
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            source_idxs = current_idxs if self.transition_outcomes and k != "observation" else idxs
            output["next"][k] = _tree_map(lambda x: x[source_idxs], self.storage[k])
        if self.compact_depth_history and "observation" in output.get("next", {}):
            output["next"]["observation"] = self._restore_depth(output["next"]["observation"], idxs)
        if self.transition_outcomes and "observation" in output.get("next", {}):
            self._restore_terminal_observations(output["next"]["observation"], current_idxs)
        return output

    def diagnostics(self, seq_length: int = 1) -> Dict[str, torch.Tensor]:
        """Return sampling diagnostics without changing replay contents."""
        self._refresh_traj()
        offset = int(bool(self.output_key_tp1))
        valid_windows = self._valid_windows(seq_length)
        eligible = valid_windows > 0
        lengths = self.lengths[eligible].to(torch.float32)
        windows = valid_windows[eligible].to(torch.float32)
        sampled = getattr(self, "_last_sampled_lengths", None)
        if lengths.numel() == 0:
            zero = torch.zeros((), device=self.lengths.device)
            return {
                "replay/trajectory_count": zero,
                "replay/trajectory_length_mean": zero,
                "replay/trajectory_length_min": zero,
                "replay/valid_window_count": zero,
                "replay/sampled_trajectory_length_mean": zero,
                "replay/sampled_trajectory_length_min": zero,
            }
        sampled = lengths if sampled is None or sampled.numel() == 0 else sampled.to(torch.float32)
        return {
            "replay/trajectory_count": torch.tensor(float(lengths.numel()), device=lengths.device),
            "replay/trajectory_length_mean": lengths.mean(),
            "replay/trajectory_length_min": lengths.min(),
            "replay/valid_window_count": windows.sum(),
            "replay/sampled_trajectory_length_mean": sampled.mean(),
            "replay/sampled_trajectory_length_min": sampled.min(),
        }


def _get_idxs(
    seq_length, num_slices, lengths, start_idx, storage_length, priorities=None,
    return_traj_idx=False, valid_windows=None,
):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    if valid_windows is None:
        valid_windows = lengths - seq_length
    end_point = valid_windows[traj_idx]
    relative_starts = (torch.rand(num_slices, device=lengths.device) * end_point).floor().to(start_idx.dtype)
    starts = torch.cat([(start_idx[traj_idx, 0] + relative_starts).unsqueeze(1), start_idx[traj_idx, 1:]], 1)
    idxs = _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)
    return (idxs, traj_idx) if return_traj_idx else idxs


def _find_start_stop_traj(end: torch.Tensor, at_capacity=True, cursor=None):
    """Return trajectory boundaries for a time-major one- or two-dimensional mask."""
    length = end.shape[0]
    if length == 0:
        empty = torch.empty((0, end.ndim), dtype=torch.long, device=end.device)
        return empty, empty, torch.empty(0, dtype=torch.long, device=end.device)
    end = end.clone()
    if not at_capacity:
        end[-1] = True
    elif cursor is not None:
        end[cursor % length] = True
    else:
        end[-1] |= ~end.any(dim=0)
    return _end_to_start_stop(end, length)


def _end_to_start_stop(end: torch.Tensor, length: int):
    stop_idx = end.transpose(0, -1).nonzero()
    stop_idx[:, [0, -1]] = stop_idx[:, [-1, 0]].clone()
    start_idx = stop_idx.clone()
    start_idx[:, 0] = (start_idx[:, 0] + 1) % length
    if start_idx.shape[0] > 1:
        same_stream = (start_idx[1:, 1:] == start_idx[:-1, 1:]).all(-1)
        dst_mask = torch.cat([torch.zeros_like(same_stream[:1]), same_stream])
        src_mask = torch.cat([same_stream, torch.zeros_like(same_stream[:1])])
        shifted = torch.empty_like(start_idx)
        shifted[dst_mask] = start_idx[src_mask]
        shifted[~dst_mask] = start_idx[~src_mask]
        start_idx = shifted
    lengths = stop_idx[:, 0] - start_idx[:, 0] + 1
    lengths[lengths <= 0] += length
    return start_idx, stop_idx, lengths


def _tensor_slices_from_startend(seq_length, starts, storage_length):
    arange = torch.arange(seq_length, device=starts.device, dtype=starts.dtype)
    ndims = starts.shape[-1] - 1 if starts.ndim > 1 else 0
    if ndims:
        delta = torch.zeros((seq_length, ndims + 1), device=starts.device, dtype=starts.dtype)
        delta[:, 0] = arange
    else:
        delta = arange.unsqueeze(-1)
    delta = delta.unsqueeze(0).expand(starts.shape[0], -1, -1)
    result = (starts.unsqueeze(1) + delta).flatten(0, 1)
    result[:, 0] = result[:, 0] % storage_length
    return result
