"""Termination terms for the UFO task.

The reference run disables all task terminations (time-out only), so
`motion_far` is provided but off by default.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def motion_far(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    motion_res = command.get_motion_state()
    root_pos_ref = motion_res["rg_pos_t"][:, 0, :]
    root_pos = env.scene["robot"].data.root_pos_w
    dist = torch.norm(root_pos[:, :2] - root_pos_ref[:, :2], dim=-1)
    return dist > threshold

from isaaclab.envs.mdp.terminations import time_out  # noqa: E402,F401
