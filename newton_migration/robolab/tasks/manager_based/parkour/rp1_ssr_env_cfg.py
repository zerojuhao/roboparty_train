# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RP1 SSR environment configurations."""

from isaaclab.utils import configclass

from robolab.tasks.manager_based.parkour.mdp.foothold_prediction import FootholdSupportCfg
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg, RP1ParkourEnvCfg_PLAY
from robolab.tasks.manager_based.parkour.ssr_env_cfg import enable_ssr_foothold_guidance


def _disable_virtual_stair_edges(cfg):
    cfg.scene.terrain.virtual_obstacles = {}
    cfg.rewards.locomotion.volume_points_penetration_feet = None
    cfg.rewards.locomotion.volume_points_penetration_knee = None
    cfg.events.register_virtual_obstacles = None
    cfg.events.register_virtual_obstacles_knee = None
    cfg.curriculum.volume_points_penetration_weight_feet = None
    cfg.curriculum.volume_points_penetration_weight_knee = None


@configclass
class RP1SSREnvCfg(RP1ParkourEnvCfg):
    foothold_support: FootholdSupportCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        enable_ssr_foothold_guidance(self)
        _disable_virtual_stair_edges(self)


@configclass
class RP1SSREnvCfg_PLAY(RP1ParkourEnvCfg_PLAY):
    foothold_support: FootholdSupportCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        enable_ssr_foothold_guidance(self)
        _disable_virtual_stair_edges(self)
