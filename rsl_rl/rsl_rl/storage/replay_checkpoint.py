"""Bounded-memory, rank-local replay snapshots; model publication is the commit."""

import json
import os
from pathlib import Path
import shutil

import torch


def save_replay(buffer, directory, *, rank, world_size, env_iterations, chunk_bytes=64 * 1024**2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "version": 3 if buffer.transition_outcomes else (2 if buffer.compact_depth_history else 1), "rank": rank, "world_size": world_size,
        "env_iterations": env_iterations, "capacity": buffer.transition_capacity,
        "num_envs": buffer.num_envs, "idx": buffer._idx, "is_full": buffer._is_full,
        "length": len(buffer), "output_key_t": buffer.output_key_t,
        "output_key_tp1": buffer.output_key_tp1, "end_key": buffer.end_key, "leaves": [],
        "compact_depth_history": buffer.compact_depth_history,
        "depth_history_offsets": list(buffer.depth_history_offsets),
        "depth_new_segment": buffer._depth_new_segment,
        "transition_outcomes": buffer.transition_outcomes,
        "terminal_chunks": [],
    }

    def write(tree, keys=()):
        for key, value in tree.items():
            path = (*keys, key)
            if isinstance(value, dict):
                write(value, path)
                continue
            leaf = {"keys": path, "shape": list(value.shape[1:]), "dtype": str(value.dtype), "chunks": []}
            leaf_id = len(metadata["leaves"])
            rows = max(1, chunk_bytes // max(1, value[0].numel() * value.element_size()))
            for start in range(0, len(buffer), rows):
                end = min(len(buffer), start + rows)
                filename = f"tensor_{leaf_id}_{start}.pt"
                # Clone only a bounded slice: torch.save(view) writes the entire backing storage.
                torch.save(value[start:end].detach().to("cpu").clone(), directory / filename)
                leaf["chunks"].append([start, end, filename])
            metadata["leaves"].append(leaf)

    try:
        write(buffer.storage or {})
        pending, pending_bytes = {}, 0
        def tensor_bytes(tree):
            if isinstance(tree, dict):
                return sum(tensor_bytes(value) for value in tree.values())
            return tree.numel() * tree.element_size()
        def flush_terminal_chunk():
            filename = f"terminal_{len(metadata['terminal_chunks'])}.pt"
            torch.save(pending, directory / filename)
            metadata["terminal_chunks"].append(filename)
        for time, terminal in buffer.terminal_observations.items():
            pending[time] = terminal
            pending_bytes += tensor_bytes(terminal)
            if pending_bytes >= chunk_bytes:
                flush_terminal_chunk()
                pending, pending_bytes = {}, 0
        if pending:
            flush_terminal_chunk()
        (directory / "manifest.json").write_text(json.dumps(metadata))
    except BaseException:
        shutil.rmtree(directory)
        raise


def load_replay(buffer, directory, *, rank, world_size, env_iterations):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    expected = {
        "version": 3 if buffer.transition_outcomes else (2 if buffer.compact_depth_history else 1),
        "rank": rank, "world_size": world_size, "env_iterations": env_iterations,
        "capacity": buffer.transition_capacity, "num_envs": buffer.num_envs,
        "output_key_t": buffer.output_key_t, "output_key_tp1": buffer.output_key_tp1,
        "end_key": buffer.end_key,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Replay checkpoint mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    for key, value, legacy in (
        ("compact_depth_history", buffer.compact_depth_history, False),
        ("depth_history_offsets", list(buffer.depth_history_offsets), []),
        ("transition_outcomes", buffer.transition_outcomes, False),
    ):
        if metadata.get(key, legacy) != value:
            raise ValueError(f"Replay checkpoint mismatch for {key}: incompatible storage")
    idx, full, length = metadata["idx"], metadata["is_full"], metadata["length"]
    if type(full) is not bool or not 0 <= idx < buffer.capacity or length != (buffer.capacity if full else idx):
        raise ValueError("Invalid replay cursor/length")
    new_segment = metadata.get("depth_new_segment", not bool(length))
    if type(new_segment) is not bool:
        raise ValueError("Invalid replay depth segment marker")
    storage = {}
    for leaf in metadata["leaves"]:
        keys = leaf["keys"]
        dtype = getattr(torch, leaf["dtype"].removeprefix("torch."), None)
        if not keys or not isinstance(dtype, torch.dtype) or leaf["shape"][0] != buffer.num_envs:
            raise ValueError("Invalid replay tensor metadata")
        dest = torch.zeros((buffer.capacity, *leaf["shape"]), dtype=dtype, device=buffer.device)
        cursor = 0
        for start, end, filename in leaf["chunks"]:
            if os.path.basename(filename) != filename or start != cursor or not start < end <= length:
                raise ValueError("Invalid replay chunk coverage")
            chunk = torch.load(directory / filename, map_location="cpu", weights_only=True)
            if chunk.dtype != dtype or list(chunk.shape) != [end - start, *leaf["shape"]]:
                raise ValueError("Replay chunk tensor shape/dtype mismatch")
            dest[start:end].copy_(chunk)
            cursor = end
        if cursor != length:
            raise ValueError("Incomplete replay tensor")
        tree = storage
        for key in keys[:-1]:
            tree = tree.setdefault(key, {})
        if keys[-1] in tree:
            raise ValueError("Duplicate replay tensor")
        tree[keys[-1]] = dest
    if length and buffer.end_key not in storage:
        raise ValueError("Replay has no episode boundaries")
    if length and buffer.compact_depth_history:
        depth = storage.get("observation", {}).get("depth_image")
        starts = storage.get("depth_episode_start")
        if (depth is None or depth.ndim != 4 or depth.dtype != torch.uint8
                or starts is None or starts.shape != (buffer.capacity, buffer.num_envs, 1)
                or starts.dtype != torch.bool):
            raise ValueError("Invalid compact depth replay tensors")
    terminal_observations = {}
    for filename in metadata.get("terminal_chunks", []):
        if not isinstance(filename, str) or os.path.basename(filename) != filename:
            raise ValueError("Invalid terminal replay chunk filename")
        chunk = torch.load(directory / filename, map_location=buffer.device, weights_only=True)
        for time, terminal in chunk.items():
            if type(time) is not int or not 0 <= time < length or time in terminal_observations:
                raise ValueError("Invalid terminal replay index")
            ids = terminal["env_ids"]
            if ids.dtype != torch.long or ids.ndim != 1 or not ids.numel() or bool(((ids < 0) | (ids >= buffer.num_envs)).any()) or ids.unique().numel() != ids.numel():
                raise ValueError("Invalid terminal replay environments")
            def validate_observation(source, reference, keys=()):
                if set(source) != set(reference):
                    raise ValueError("Terminal replay observation keys mismatch")
                for key, value in source.items():
                    if isinstance(value, dict):
                        validate_observation(value, reference[key], (*keys, key))
                    else:
                        expected_shape = (ids.numel(), *reference[key].shape[2:])
                        if buffer.compact_depth_history and (*keys, key) == ("depth_image",):
                            expected_shape = (ids.numel(), len(buffer.depth_history_offsets), *reference[key].shape[2:])
                        if value.shape != expected_shape or value.dtype != reference[key].dtype:
                            raise ValueError("Terminal replay observation shape/dtype mismatch")
            validate_observation(terminal["observation"], storage["observation"])
            terminal_observations[time] = terminal
    if length and buffer.transition_outcomes:
        valid = storage.get("terminal_valid")
        if valid is None or valid.shape != (buffer.capacity, buffer.num_envs, 1) or valid.dtype != torch.bool:
            raise ValueError("Invalid terminal replay validity mask")
        restored_valid = torch.zeros_like(valid)
        for time, terminal in terminal_observations.items():
            restored_valid[time, terminal["env_ids"]] = True
        if not torch.equal(valid[:length], restored_valid[:length]):
            raise ValueError("Incomplete terminal replay observations")
    buffer.terminal_observations = terminal_observations
    buffer.storage = storage or None
    buffer._idx, buffer._is_full, buffer._recompute = idx, full, True
    buffer._depth_new_segment = new_segment
