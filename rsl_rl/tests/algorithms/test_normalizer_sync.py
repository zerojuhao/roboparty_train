"""Distributed equivalence and bounded NCCL benchmark for normalizer sync."""

import argparse
import copy
from datetime import timedelta
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent
from rsl_rl.algorithms.fb_networks import EMA, ObsNormalizer


def make_agent(device):
    agent = object.__new__(FbCprAuxAgent)
    agent._obs_normalizer = ObsNormalizer({
        "state": 54, "privileged_state": 373, "last_action": 24,
        "history_actor": 312, "terrain_priv": 273,
    }).to(device)
    agent._aux_reward_normalizer = EMA().to(device)
    return agent


def buffers(agent):
    return [*agent._obs_normalizer.buffers(), *agent._aux_reward_normalizer.buffers()]


def legacy_sync(agent):
    for value in buffers(agent):
        if value.is_floating_point():
            torch.distributed.all_reduce(value)
            value.div_(torch.distributed.get_world_size())


def equivalence_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        agent = make_agent("cpu")
        agent._aux_reward_normalizer.register_buffer("double_probe", torch.zeros(3, dtype=torch.float64))
        pointers = [value.data_ptr() for value in buffers(agent)]
        for iteration in range(3):
            for index, value in enumerate(buffers(agent)):
                value.fill_(index + rank * 2 + iteration)
            reference = copy.deepcopy(agent)
            legacy_sync(reference)
            with patch("torch.distributed.all_reduce", wraps=torch.distributed.all_reduce) as reduce:
                agent._sync_floating_buffers()
                assert reduce.call_count == 2  # fp32 and fp64; no integer collective.
            for actual, expected in zip(buffers(agent), buffers(reference)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert pointers == [value.data_ptr() for value in buffers(agent)]
        # Saved normalizers retain the original keys/shapes after coalescing.
        restored = make_agent("cpu")
        restored._obs_normalizer.load_state_dict(agent._obs_normalizer.state_dict())
    finally:
        torch.distributed.destroy_process_group()


class TestNormalizerSync(unittest.TestCase):
    def test_distributed_values_counters_and_storage_match_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(equivalence_worker, args=(os.path.join(directory, "rendezvous"),),
                                        nprocs=2, join=True)

    def test_single_process_is_unchanged(self):
        agent = make_agent("cpu")
        before = [value.clone() for value in buffers(agent)]
        agent._sync_floating_buffers()
        for actual, expected in zip(buffers(agent), before):
            torch.testing.assert_close(actual, expected)


def benchmark(iterations, rounds):
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
    agent = make_agent(f"cuda:{rank}")
    try:
        original = copy.deepcopy(agent)
        for index, value in enumerate(buffers(agent)):
            value.fill_(index + rank)
        for index, value in enumerate(buffers(original)):
            value.fill_(index + rank)
        legacy_sync(original)
        agent._sync_floating_buffers()
        for actual, expected in zip(buffers(agent), buffers(original)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        phases = (("legacy", legacy_sync), ("coalesced", FbCprAuxAgent._sync_floating_buffers))
        schedule = [(round_index, name, sync) for round_index in range(rounds)
                    for name, sync in (phases if round_index % 2 == 0 else phases[::-1])]
        for round_index, name, sync in schedule:
            for _ in range(10):
                sync(agent)
            torch.distributed.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                sync(agent)
            torch.cuda.synchronize()
            ms = torch.tensor([(time.perf_counter() - start) * 1000 / iterations], device=f"cuda:{rank}")
            samples = [torch.zeros_like(ms) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(samples, ms)
            if rank == 0:
                print(json.dumps({"phase": name, "round": round_index, "iterations": iterations,
                                  "ms_per_update_by_rank": [value.item() for value in samples]}), flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    args, rest = parser.parse_known_args()
    if args.benchmark:
        benchmark(args.iterations, args.rounds)
    else:
        unittest.main(argv=[__file__, *rest])
