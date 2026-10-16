"""Portable paths for the repository-local Newton data package."""

from __future__ import annotations

import os
from pathlib import Path

from robolab.assets import ISAAC_DATA_DIR


def _motion_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("ROBO_TRAIN_DATA_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(Path(ISAAC_DATA_DIR))
    return roots


def motion_data_dir(dataset: str) -> str:
    """Return the first available ``motions/<dataset>`` directory."""
    candidates = [root / "motions" / dataset for root in _motion_roots()]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])


def motion_file(dataset: str, filename: str) -> str:
    """Resolve one file within a unified Newton motion dataset."""
    return str(Path(motion_data_dir(dataset)) / filename)


__all__ = ["motion_data_dir", "motion_file"]
