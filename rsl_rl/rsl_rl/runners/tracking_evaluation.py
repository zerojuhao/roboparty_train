"""Persistent, isolated tracking evaluation and exact motion-shard aggregation."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import torch


def isolated_evaluator_device(device, environment):
    """Map the parent's logical CUDA device to a one-GPU child namespace."""
    device = torch.device(device)
    environment = dict(environment)
    if device.type != "cuda":
        return str(device), environment
    index = device.index if device.index is not None else torch.cuda.current_device()
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(index)
    else:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if index >= len(devices):
            raise ValueError(f"Evaluator device {device} is outside CUDA_VISIBLE_DEVICES")
        selected = devices[index]
    environment["CUDA_VISIBLE_DEVICES"] = selected
    return "cuda:0", environment


def motion_shard(num_motions, rank, world_size):
    if num_motions < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid tracking evaluation shard dimensions")
    return list(range(rank, num_motions, world_size))


def merge_tracking_shards(results, motion_keys, num_motions):
    errors = [result["error"] for result in results if "error" in result]
    if errors:
        raise RuntimeError("Tracking evaluation failed: " + "; ".join(errors))
    sums, counts, clips = {}, {}, {}
    for rank, result in enumerate(results):
        if result["motion_keys"] != motion_keys:
            raise ValueError("Tracking evaluation motion ordering differs from training")
        expected = motion_shard(num_motions, rank, len(results))
        if result["motion_ids"] != expected or sorted(result["clips"]) != expected:
            raise ValueError(f"Tracking evaluation shard {rank} has missing or duplicate clips")
        if set(result["sums"]) != set(result["counts"]):
            raise ValueError("Tracking evaluation metric sums/counts differ")
        if expected and (not result["sums"] or any(
            count != len(expected) for count in result["counts"].values()
        )):
            raise ValueError("Tracking evaluation did not measure every clip")
        for key, value in result["sums"].items():
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite tracking metric: {key}")
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + result["counts"][key]
        for clip_id, values in result["clips"].items():
            if clip_id in clips or not all(math.isfinite(v) for v in values.values()):
                raise ValueError("Duplicate or non-finite tracking clip result")
            clips[clip_id] = values
    if sorted(clips) != list(range(num_motions)) or not sums:
        raise ValueError("Tracking evaluation coverage is incomplete")
    if any(count != num_motions for count in counts.values()):
        raise ValueError("Tracking evaluation metric coverage is incomplete")
    return {"metrics": {key: value / counts[key] for key, value in sums.items()}, "clips": clips}


def gather_tracking_shards_via_files(
    directory, global_step, eval_mode, rank, world_size, result, group, *, timeout_seconds=1860
):
    """Exchange evaluation shards through shared storage instead of object collectives.

    Tracking results contain per-clip dictionaries and optional phase-error arrays. Sending
    those objects through a Gloo object collective made the training ranks sit at 0% GPU
    utilization after the evaluator children had already completed. The shared filesystem
    keeps the collective payload to three small barriers and leaves the serialized evidence
    available when an aggregation failure needs diagnosis.
    """
    if world_size < 2:
        return [result]
    mode = eval_mode or "legacy"
    round_dir = os.path.join(directory, f".shards_{global_step}_{mode}")
    if rank == 0:
        shutil.rmtree(round_dir, ignore_errors=True)
        os.makedirs(round_dir, exist_ok=True)
    timeout = timedelta(seconds=max(30.0, float(timeout_seconds)))
    torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=True)

    compact = {
        key: result[key]
        for key in ("motion_keys", "motion_ids", "sums", "counts", "clips", "phase_errors", "eval_mode")
        if key in result
    }
    temporary = os.path.join(round_dir, f"rank{rank}.pt.tmp.{os.getpid()}")
    destination = os.path.join(round_dir, f"rank{rank}.pt")
    torch.save(compact, temporary)
    os.replace(temporary, destination)
    torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=True)

    merged_path = os.path.join(round_dir, "merged.pt")
    if rank == 0:
        try:
            results = [torch.load(os.path.join(round_dir, f"rank{index}.pt"),
                                  map_location="cpu", weights_only=False)
                       for index in range(world_size)]
            payload = {"results": results}
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        temporary = merged_path + f".tmp.{os.getpid()}"
        torch.save(payload, temporary)
        os.replace(temporary, merged_path)
    torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=True)
    payload = torch.load(merged_path, map_location="cpu", weights_only=False)
    if "error" in payload:
        raise RuntimeError("Tracking shard aggregation failed: " + payload["error"])
    return payload["results"]


class PersistentTrackingEvaluator:
    """One simulator child per rank; stdin requests and atomic result files."""

    def __init__(self, *, entrypoint, request, device, environment, log_dir, rank, timeout):
        self.timeout = timeout
        self.log_path = os.path.join(log_dir, f"tracking_eval_rank{rank}.log")
        self.directory = tempfile.TemporaryDirectory(prefix=f"tracking_eval_rank{rank}_", dir=log_dir)
        self.process = None
        self.output = None
        try:
            initial_path = os.path.join(self.directory.name, "initial.pt")
            request = dict(request, persistent=True, log_dir=self.directory.name)
            torch.save(request, initial_path)
            self.output = open(self.log_path, "a")
            self.process = subprocess.Popen(
                [sys.executable, entrypoint, "--tracking_eval_request", initial_path,
                 "--device", str(device), "--visualizer", "none"],
                env=environment, stdin=subprocess.PIPE, stdout=self.output,
                stderr=subprocess.STDOUT, text=True,
            )
        except BaseException:
            self.close()
            raise

    def evaluate(self, model_state, global_step, motion_ids):
        request_path = os.path.join(self.directory.name, "round.pt")
        result_path = os.path.join(self.directory.name, f"result_{global_step}.pt")
        torch.save({"model_state": model_state, "global_step": global_step,
                    "motion_ids": motion_ids, "result_path": result_path}, request_path)
        self.process.stdin.write(json.dumps({"request_path": request_path}) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while not os.path.isfile(result_path):
            if self.process.poll() is not None:
                raise RuntimeError(f"Evaluator exited with code {self.process.returncode}; see {self.log_path}")
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(f"Tracking evaluation timed out; see {self.log_path}")
            time.sleep(0.05)
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        os.unlink(result_path)
        if result.get("global_step") != global_step:
            raise ValueError("Tracking evaluator returned a stale result")
        return result

    def close(self):
        if self.process is not None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.output is not None:
            self.output.close()
            self.output = None
        self.directory.cleanup()
