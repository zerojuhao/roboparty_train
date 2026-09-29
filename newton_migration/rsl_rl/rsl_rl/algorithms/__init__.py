# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .distillation import Distillation
from .fb_cpr_aux import FbCprAuxAgent
from .multi_reward_ppo_amp import MultiRewardPPOAMP
from .ppo import PPO
from .ppo_amp import PPOAMP

__all__ = ["PPO", "PPOAMP", "MultiRewardPPOAMP", "Distillation", "FbCprAuxAgent"]
