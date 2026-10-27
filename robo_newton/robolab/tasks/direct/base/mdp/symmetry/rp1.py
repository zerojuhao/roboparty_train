# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Name-based symmetry for RP1 Direct observations and actions."""

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def joint_mirror_spec(joint_names):
    indices_by_name = {name: index for index, name in enumerate(joint_names)}
    if len(indices_by_name) != 24 or len(joint_names) != 24:
        raise ValueError("RP1 symmetry requires 24 unique joint names.")
    indices, signs = [], []
    for name in joint_names:
        if name.startswith("left_"):
            other = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            other = "left_" + name.removeprefix("right_")
        elif name in ("waist_roll_joint", "waist_yaw_joint"):
            other = name
        else:
            raise ValueError(f"Unsupported RP1 joint: {name!r}.")
        if other not in indices_by_name:
            raise ValueError(f"Missing RP1 mirrored joint: {other!r}.")
        indices.append(indices_by_name[other])
        # RP1's ankle-pitch axes share a sign; the remaining axes reverse.
        signs.append(1.0 if name.endswith("ankle_pitch_joint") else -1.0)
    return tuple(indices), tuple(signs)


def mirror_joints(values, joint_names):
    if values.shape[-1] != 24:
        raise ValueError(f"RP1 joint tensor has {values.shape[-1]} columns, expected 24.")
    indices, signs = joint_mirror_spec(tuple(joint_names))
    return values[..., list(indices)] * values.new_tensor(signs)
