"""RP1 AMP Soft runner: force-aware actor, critic, and dual AMP discriminators."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from ..mdp.symmetry.rp1 import compute_symmetric_states
from .amp_model_cfg import AMPModelCfg
from .rsl_rl_amp_cfg import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg


@configclass
class RP1AmpSoftRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'OnPolicyRunner'
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = 'rp1_amp_soft'
    wandb_project = 'rp1_amp_soft'
    obs_groups = {
        'actor': ['policy'],
        'critic': ['critic'],
        'discriminator': ['disc'],
        'discriminator_demonstration': ['disc_demo'],
        'discriminator_lower': ['disc_lower'],
        'discriminator_lower_demonstration': ['disc_lower_demo'],
    }
    actor = AMPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation='elu',
        distribution_cfg=AMPModelCfg.GaussianDistributionCfg(init_std=1.0),
        enable_force_estimation=True,
        force_estimation_history_length=8,
        force_estimation_with_torque=True,
        force_estimation_multi_contact=True,
        force_estimation_compact=True,
        enable_force_vel_target_estimation=True,
        # Penalize false load detections 5x (removes ln 5 of the balanced-loss logit bias).
        force_load_negative_weight=5.0,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation='elu',
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name='PPOAMP',
        enable_aux_loss=True,
        aux_loss_coef=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        adaptive_learning_rate_max=5.0e-4,
        schedule='adaptive',
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
            route_obs_group='amp_route',
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256],
                activation='elu',
                style_reward_scale=2.0,
                task_style_lerp=0.4,
            ),
            discriminators={
                'general': {
                    'route_value': 0,
                    'policy_obs_set': 'discriminator',
                    'demonstration_obs_set': 'discriminator_demonstration',
                    'amp_discriminator': {'hidden_dims': [512, 256]},
                },
                'compliance': {
                    'route_value': 1,
                    'policy_obs_set': 'discriminator_lower',
                    'demonstration_obs_set': 'discriminator_lower_demonstration',
                    'amp_discriminator': {'hidden_dims': [512, 256]},
                },
            },
            loss_type='LSGAN',
        ),
    )


def mixed_runner_cfg():
    """Keep existing Mixed play commands looking in their original log folder."""
    return RP1AmpSoftRunnerCfg(experiment_name='rp1_amp_mixed', wandb_project='rp1_amp_mixed')


def interaction_runner_cfg():
    return RP1AmpSoftRunnerCfg(experiment_name='rp1_amp_interaction', wandb_project='rp1_amp_interaction')
