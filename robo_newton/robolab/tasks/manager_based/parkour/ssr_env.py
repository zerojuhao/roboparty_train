# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SSR environment with imagined foothold guidance."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab.envs import VecEnvStepReturn

from robolab.tasks.manager_based.parkour.mdp.foothold_imagination import (
    FootholdImaginationManager,
)
from robolab.tasks.manager_based.parkour.mdp.observations.foothold_observations import (
    build_foothold_predictor_state,
    resolve_foothold_predictor_obs,
)
from robolab.tasks.manager_based.parkour.parkour_env import ParkourEnv


class SSREnv(ParkourEnv):
    """AMP env with imagined-foothold predictor hooks."""

    def load_managers(self):
        super().load_managers()
        self.foothold_guidance = None

    def configure_foothold_imagination(self, predictor_cfg: dict) -> None:
        if self.foothold_guidance is not None:
            raise RuntimeError("Foothold imagination has already been configured.")
        self.foothold_guidance = FootholdImaginationManager(
            self, predictor_cfg, self.cfg.foothold_support
        )
        print("[INFO] SSR imagined foothold guidance enabled (critic obs + current action).")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0 and self.foothold_guidance is not None:
            self.foothold_guidance.reset(reset_env_ids)
        return obs, rew, terminated, truncated, extras

    def prepare_foothold_prediction_step(
        self,
        obs: TensorDict,
        action: torch.Tensor,
        enable_inference_guidance: bool = False,
    ) -> None:
        """Predict footholds from the critic observation group plus action."""
        if self.foothold_guidance is None:
            return
        if enable_inference_guidance:
            self.foothold_guidance.enable_inference_guidance()
        privileged = resolve_foothold_predictor_obs(obs)
        self.foothold_guidance.prepare_step(build_foothold_predictor_state(privileged), action)

    def update_foothold_predictor(self) -> dict[str, float]:
        if self.foothold_guidance is None:
            return {}
        return self.foothold_guidance.update_predictor()
