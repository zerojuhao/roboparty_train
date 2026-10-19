# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from ..mdp.symmetry.rp1 import compute_symmetric_states

from .amp_model_cfg import AMPModelCfg
from .rpo_amp_agent_cfg import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg


@configclass
class RP1AmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "rp1_amp"
    logger = "wandb"
    wandb_project = "rp1_amp"
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
        distribution_cfg=AMPModelCfg.GaussianDistributionCfg(
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
            data_augmentation_func=compute_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            disc_num_updates=4,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[256, 128],
                activation="elu",
                style_reward_scale=4.0,
                task_style_lerp=0.7,
            ),
            discriminators={
                "general": {
                    "all_envs": True,
                    "policy_obs_set": "discriminator",
                    "demonstration_obs_set": "discriminator_demonstration",
                    "amp_discriminator": {"hidden_dims": [256, 128]},
                },
            },
            loss_type="LSGAN",
        ),
    )
