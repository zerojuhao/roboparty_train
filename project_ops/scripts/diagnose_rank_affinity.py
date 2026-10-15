"""Bounded A/B/A CPU-affinity experiment on an existing torchrun job."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def cpu_set(value):
    result = set()
    for part in value.strip().split(","):
        bounds = part.split("-")
        result.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def task_stats(pid):
    stats = {}
    for path in Path(f"/proc/{pid}/task").glob("*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            stats[int(path.parent.name)] = {
                "name": raw[raw.index("(") + 1:raw.rindex(")")],
                "ticks": int(fields[11]) + int(fields[12]),
                "cpu": int(fields[36]),
                "start": int(fields[19]),
            }
        except FileNotFoundError:
            pass
    return stats


def discover(parent):
    gpus = {}
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader",
    ], text=True)
    for index, bus in csv.reader(io.StringIO(output)):
        domain, device, function = bus.strip().lower().split(":")
        path = Path(f"/sys/bus/pci/devices/{int(domain, 16):04x}:{device}:{function}/local_cpulist")
        gpus[int(index)] = cpu_set(path.read_text())
    ranks = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            status = dict(line.split(":", 1) for line in path.read_text().splitlines())
            if int(status["PPid"]) != parent:
                continue
            env = dict(entry.split("=", 1) for entry in (path.parent / "environ").read_text().split("\0") if "=" in entry)
            if "LOCAL_RANK" not in env:
                continue
            rank = int(env["LOCAL_RANK"])
            visible = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, sorted(gpus)))).split(",")
            gpu = int(visible[rank])
            pid = int(path.parent.name)
            tasks = task_stats(pid)
            affinity = {tid: sorted(os.sched_getaffinity(tid)) for tid in tasks}
            local = gpus[gpu] & set(affinity[pid])
            if not local:
                raise ValueError(f"rank {rank} has no allowed local CPUs")
            ranks.append(dict(rank=rank, pid=pid, gpu=gpu, start=tasks[pid]["start"],
                              original=affinity, local=sorted(local)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not ranks:
        raise ValueError(f"No live training ranks under PID {parent}")
    return sorted(ranks, key=lambda row: row["rank"])


def set_affinity(ranks, local):
    for rank in ranks:
        tasks = task_stats(rank["pid"])
        main = tasks.get(rank["pid"])
        if main is None or main["start"] != rank["start"]:
            if local:
                raise RuntimeError(f"rank {rank['rank']} exited")
            continue
        for tid in tasks:
            original = rank["original"].get(tid, rank["original"][rank["pid"]])
            cpus = sorted(set(original) & set(rank["local"])) if local else original
            # Preserve pre-existing thread restrictions that do not overlap.
            if not cpus:
                continue
            try:
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                pass


def sample_phase(name, ranks, seconds, log, output):
    before = {rank["pid"]: task_stats(rank["pid"]) for rank in ranks}
    offset = log.stat().st_size
    start = time.monotonic()
    gpu_rows = []
    while time.monotonic() - start < seconds:
        values = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=10)
        gpu_rows.extend([[time.time(), *map(float, row)] for row in csv.reader(io.StringIO(values))])
        time.sleep(1)
    elapsed = time.monotonic() - start
    with log.open() as stream:
        stream.seek(offset)
        lines = stream.read()
    points = [(int(step.replace(",", "")), int(fps.replace(",", "")))
              for step, fps in re.findall(r"\[FbCpr\] env_steps=([\d,]+).*?fps=([\d,]+)", lines)]
    rank_stats = []
    for rank in ranks:
        after = task_stats(rank["pid"])
        ticks = os.sysconf("SC_CLK_TCK")
        threads = []
        for tid, value in after.items():
            prior = before[rank["pid"]].get(tid)
            if prior and prior["start"] == value["start"]:
                threads.append(dict(tid=tid, name=value["name"], cpu=value["cpu"],
                                    cpu_percent=100 * (value["ticks"] - prior["ticks"]) / ticks / elapsed))
        rank_stats.append(dict(rank=rank["rank"], threads=sorted(threads, key=lambda x: -x["cpu_percent"])[:8]))
    gpu_means = {gpu: statistics.mean(row[2] for row in gpu_rows if row[1] == gpu)
                 for gpu in sorted({row[1] for row in gpu_rows})}
    result = dict(phase=name, elapsed=elapsed, points=points, ranks=rank_stats,
                  gpu_means=gpu_means, evaluation_overlap="tracking evaluation" in lines)
    with (output / f"{name}_gpu.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "gpu", "utilization", "power_w", "memory_mib"])
        writer.writerows(gpu_rows)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2))
    fps = [point[1] for point in points]
    print(json.dumps(dict(phase=name, updates=len(fps), fps_mean=statistics.mean(fps) if fps else None,
                          fps_min=min(fps) if fps else None, gpu_means=gpu_means,
                          evaluation_overlap=result["evaluation_overlap"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ranks = discover(args.parent)
    (args.output / "original_affinity.json").write_text(json.dumps(ranks, indent=2))
    print(json.dumps([{k: row[k] for k in ("rank", "pid", "gpu", "local")} for row in ranks]), flush=True)
    sample_phase("baseline", ranks, args.seconds, args.log, args.output)
    try:
        set_affinity(ranks, local=True)
        sample_phase("local_numa", ranks, args.seconds, args.log, args.output)
    finally:
        set_affinity(ranks, local=False)
        print("Original CPU affinities restored", flush=True)
    sample_phase("restored", ranks, args.seconds, args.log, args.output)


if __name__ == "__main__":
    main()
