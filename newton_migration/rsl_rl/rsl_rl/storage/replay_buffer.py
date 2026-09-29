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
        self.file_names = []

    @property
    def capacity(self):
        return self.storage[self.end_key].shape[0]

    def __len__(self):
        return self.capacity

    def sample(self, batch_size: int, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        assert batch_size % seq_length == 0
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        idxs = _get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=self.lengths[traj_idx],
            start_idx=self.start_idx[traj_idx],
            storage_length=self.capacity,
            priorities=None,
        )
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = _tree_map(lambda x: x[idxs], self.storage[k])
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            output["next"][k] = _tree_map(lambda x: x[idxs], self.storage[k])
        return output


class ReplayBuffer:
    """Time-major ring-buffer matching UFO's TrajectoryDictBufferMultiDim."""

    def __init__(self, capacity: int, num_envs: int, device="cpu",
                 output_key_t=("observation", "action", "z", "aux_rewards"),
                 output_key_tp1=("observation", "terminated"), end_key="truncated"):
        if capacity < 2 * num_envs:
            raise ValueError("replay capacity must hold at least two vector-environment steps")
        self.transition_capacity = capacity
        self.capacity = capacity // num_envs
        self.num_envs = num_envs
        self.device = device
        self.output_key_t = list(output_key_t)
        self.output_key_tp1 = list(output_key_tp1)
        self.end_key = end_key
        self.storage = None
        self._idx = 0
        self._is_full = False
        self._recompute = True

    def __len__(self):
        return self.capacity if self._is_full else self._idx

    def size(self):
        return len(self) * self.num_envs

    def empty(self):
        return len(self) == 0

    @torch.no_grad()
    def extend(self, data: Dict) -> None:
        """Append one or more time-major vector-environment observations.

        Every tensor leaf must have shape ``(time, num_envs, ...)``.  The
        runner deliberately stores the previous step's termination flags with
        the current observation, just like the original UFO workspace.
        """
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
        done = self.storage[self.end_key]
        if done.shape[-1] == 1:
            done = done.squeeze(-1)
        done = done if self._is_full else done[: self._idx]
        self.start_idx, self.stop_idx, self.lengths = _find_start_stop_traj(
            done, at_capacity=self._is_full, cursor=self._idx - 1
        )
        self._recompute = False

    def sample(self, batch_size: int, seq_length: int = 1):
        self._refresh_traj()
        assert batch_size % seq_length == 0
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        lengths = self.lengths[traj_idx]
        if lengths.numel() == 0:
            raise ValueError("no trajectory long enough to sample")
        idxs = _get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=lengths,
            start_idx=self.start_idx[traj_idx],
            storage_length=self.capacity,
            priorities=None,
        )
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = _tree_map(lambda x: x[idxs], self.storage[k])
        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            output["next"][k] = _tree_map(lambda x: x[idxs], self.storage[k])
        return output


def _get_idxs(seq_length, num_slices, lengths, start_idx, storage_length, priorities=None):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    end_point = lengths[traj_idx] - seq_length - 1
    relative_starts = (torch.rand(num_slices, device=lengths.device) * end_point).floor().to(start_idx.dtype)
    starts = torch.cat([(start_idx[traj_idx, 0] + relative_starts).unsqueeze(1), start_idx[traj_idx, 1:]], 1)
    return _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)


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
