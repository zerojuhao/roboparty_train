# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from robolab.tasks.manager_based.amp.mdp.symmetry import rpo

from .amp_model_cfg import AMPModelCfg


@configclass
class RslRlAmpCfg:
    """Configuration class for AMP (Adversarial Motion Priors) training."""

    disc_obs_buffer_size: int = 1000
    """Size of the replay buffer for storing discriminator observations"""
    grad_penalty_scale: float = 10.0
    """Scale for the gradient penalty in AMP training"""
    disc_trunk_weight_decay: float = 1.0e-4
    """Weight decay for the discriminator trunk network"""
    disc_linear_weight_decay: float = 1.0e-2
    """Weight decay for the discriminator linear network"""
    disc_learning_rate: float = 1.0e-5
    """Learning rate for the discriminator networks"""
    disc_max_grad_norm: float = 1.0
    """Maximum gradient norm for the discriminator networks"""
    disc_num_updates: int | None = None
    """Discriminator updates per policy iteration. None preserves the PPO epoch/minibatch product."""

    @configclass
    class AMPDiscriminatorCfg:
        """Configuration for the AMP discriminator network."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the AMP discriminator network."""

        activation: str = "elu"
        """The activation function for the AMP discriminator network."""

        style_reward_scale: float = 1.0
        """Scale for the style reward in the training"""
        task_style_lerp: float = 0.0
        """Linear interpolation factor for the task style reward in the AMP training."""
    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()
    """Configuration for the AMP discriminator network."""
    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
    """Type of loss function used for the AMP discriminator (e.g., 'GAN', 'LSGAN', 'WGAN')"""

    discriminators: dict[str, dict] = {
        "general": {
            "route_value": 0,
            "policy_obs_set": "discriminator",
            "demonstration_obs_set": "discriminator_demonstration",
        }
    }
    """Discriminator routes, each with its own observations and dataset."""

    route_obs_group: str | None = None
    """Observation group selecting the discriminator route per environment."""


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm."""

    class_name: str = "PPOAMP"
    """The algorithm class name. Default is PPOAmp."""

    enable_aux_loss: bool = False
    """Whether to optimize auxiliary policy losses exposed by the actor-critic."""

    aux_loss_coef: float = 0.0
    """Weight applied to auxiliary policy losses."""

    use_mixed_precision: bool = False
    """Whether to run policy and value forward/loss computation in BF16 autocast."""

    grad_reduce_bucket_mb: float = 25.0
    """Maximum packed gradient bucket size used by distributed PPO training."""

    resume_learning_rate: float | None = None
    """Explicit rate override after restoring Adam state; None keeps the checkpoint rate."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


@configclass
class RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "rpo_amp"
    wandb_project = "rpo_amp"
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    actor = AMPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        enable_aux_loss=True,
        aux_loss_coef=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-3,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=1.5,
                task_style_lerp=0.6,
            ),
            loss_type="LSGAN",
        ),
    )
