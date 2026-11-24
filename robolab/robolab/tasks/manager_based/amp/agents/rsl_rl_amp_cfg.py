# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared AMP algorithm config used by AMP and parkour runners."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


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

    actor_target_loss_coef: float = 0.0
    """Training-only actor-mean interval supervision, separate from estimators."""

    actor_target_warmup_updates: int = 0
    """Number of PPO updates to reach the configured actor target coefficient."""

    use_mixed_precision: bool = False
    """Whether to run policy and value forward/loss computation in BF16 autocast."""

    grad_reduce_bucket_mb: float = 25.0
    """Maximum packed gradient bucket size used by distributed PPO training."""

    resume_learning_rate: float | None = None
    """Explicit rate override after restoring Adam state; None keeps the checkpoint rate."""

    adaptive_learning_rate_max: float = 1.0e-2
    """Upper bound for adaptive PPO rates, including restored or overridden rates."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""

