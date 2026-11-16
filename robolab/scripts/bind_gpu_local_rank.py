"""Bind a torchrun worker before imports/allocations, then exec its training script."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def rank_placement(environ):
    rank = int(environ["LOCAL_RANK"])
    visible = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not 0 <= rank < len(visible) or not visible[rank].strip():
        raise ValueError("An explicit CUDA_VISIBLE_DEVICES entry is required for LOCAL_RANK")
    gpu = visible[rank].strip()
    output = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=pci.bus_id", "--format=csv,noheader",
    ], text=True, timeout=10)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("Expected one PCI bus ID for the selected GPU")
    match = re.fullmatch(r"([0-9a-f]{4,8}):([0-9a-f]{2}):([0-9a-f]{2}\.[0-7])",
                         rows[0][0].strip().lower())
    if match is None:
        raise ValueError("Invalid GPU PCI bus ID")
    domain, bus, device = match.groups()
    pci = f"{int(domain, 16):04x}:{bus}:{device}"
    local = set()
    for part in Path(f"/sys/bus/pci/devices/{pci}/local_cpulist").read_text().strip().split(","):
        bounds = part.split("-")
        local.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    allowed = set(os.sched_getaffinity(0))
    cpus = sorted(local & allowed)
    if not cpus:
        raise ValueError(f"rank {rank} has no allowed CPUs local to GPU {gpu} ({pci})")
    return {"rank": rank, "gpu": gpu, "pci": pci, "cpus": cpus,
            "original_cpus": sorted(allowed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    placement = rank_placement(os.environ)
    print("[NUMA] " + json.dumps(placement), flush=True)
    if args.dry_run:
        return
    # All subsequently created threads inherit this affinity. Anonymous replay
    # allocations can then first-touch locally; actual placement must be checked.
    os.sched_setaffinity(0, placement["cpus"])
    os.execv(sys.executable, [sys.executable, args.script, *args.args])


if __name__ == "__main__":
    main()
