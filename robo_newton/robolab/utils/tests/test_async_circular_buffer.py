"""Direct history indexing preserves the asynchronous camera ring contract."""

import pytest
import torch

from robolab.utils.buffers import AsyncCircularBuffer


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def old_history(buffer, batch_ids):
    ids = buffer._ALL_INDICES if batch_ids is None else torch.as_tensor(batch_ids, device=buffer.device)
    selected = buffer._buffer[:, ids].clone()
    shifts = buffer.max_length - buffer._pointer[ids] - 1
    index = (torch.arange(buffer.max_length, device=buffer.device)[:, None] - shifts[None, :]) % buffer.max_length
    index = index.reshape(buffer.max_length, len(ids), *([1] * (selected.ndim - 2))).expand_as(selected)
    return torch.gather(selected, 0, index).transpose(0, 1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_history_matches_old_gather_after_async_resets_and_wrap(device, dtype):
    buffer = AsyncCircularBuffer(37, 4, device)
    all_ids = torch.arange(4, device=device)
    buffer.append(torch.zeros(4, 2, 3, 1, device=device, dtype=dtype), all_ids)
    offsets = torch.tensor([35, 30, 25, 20, 15, 10, 5, 0], device=device)
    delays = torch.tensor([0, 1, 1, 0], device=device)
    for step in range(80):
        if step in (9, 40, 61):
            buffer.reset([1, 3])
            delays[[1, 3]] = 1 - delays[[1, 3]]
        ids = all_ids if step % 2 == 0 else all_ids[[0, 2]]
        values = (step + ids).to(dtype).reshape(-1, 1, 1, 1).expand(-1, 2, 3, 1)
        buffer.append(values, ids)
        for selection in (None, [3, 1, 0], all_ids[:0], [2, 2]):
            actual = buffer.get_by_batch_ids(selection)
            torch.testing.assert_close(actual, old_history(buffer, selection), rtol=0, atol=0)
        history = buffer.get_by_batch_ids()
        frame_indices = 36 - offsets[None, :] - delays[:, None]
        torch.testing.assert_close(history[all_ids[:, None], frame_indices],
                                   old_history(buffer, None)[all_ids[:, None], frame_indices], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_history_is_independent_snapshot_and_first_frame_is_repeated(device):
    buffer = AsyncCircularBuffer(4, 3, device)
    ids = torch.arange(3, device=device)
    first = torch.arange(6, device=device).reshape(3, 2)
    buffer.append(first, ids)
    snapshot = buffer.get_by_batch_ids()
    torch.testing.assert_close(snapshot, first[:, None].expand(-1, 4, -1))
    buffer.append(first + 10, ids)
    torch.testing.assert_close(snapshot, first[:, None].expand(-1, 4, -1))
    snapshot.fill_(-1)
    assert (buffer.get_by_batch_ids() >= 0).all()
    buffer.reset([1])
    buffer.append(first[1:2] + 20, ids[1:2])
    torch.testing.assert_close(buffer.get_by_batch_ids([1]), (first[1:2] + 20)[:, None].expand(-1, 4, -1))
