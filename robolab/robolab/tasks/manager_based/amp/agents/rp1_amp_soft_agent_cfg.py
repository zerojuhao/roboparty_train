"""Full-body AMP for commanded locomotion; legs and waist AMP at zero command."""

from isaaclab.utils import configclass

from .rp1_amp_agent_cfg import RP1AmpRunnerCfg


@configclass
class RP1AmpSoftRunnerCfg(RP1AmpRunnerCfg):
    experiment_name = 'rp1_amp_soft'
    wandb_project = 'rp1_amp_soft'
    save_interval = 200

    def __post_init__(self):
        self.actor.enable_force_estimation = True
        self.actor.force_estimation_history_length = 8
        self.actor.force_estimation_with_torque = True
        self.actor.force_estimation_multi_contact = True
        self.actor.force_estimation_compact = True
        self.actor.enable_force_vel_target_estimation = True
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.adaptive_learning_rate_max = 5.0e-4
        self.algorithm.desired_kl = 0.01
        self.algorithm.symmetry_cfg.use_data_augmentation = False
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.obs_groups['critic'] = ['critic', 'compact_interaction_critic', 'force_vel_target']
        self.obs_groups['discriminator_lower'] = ['disc_lower']
        self.obs_groups['discriminator_lower_demonstration'] = ['disc_lower_demo']
        self.algorithm.force_style_reward_scale = 1.0
        self.algorithm.amp_cfg.amp_discriminator.style_reward_scale = 2.0
        self.algorithm.amp_cfg.amp_discriminator.task_style_lerp = 0.4
        self.algorithm.amp_cfg.amp_discriminator.hidden_dims = [512, 256]
        self.algorithm.amp_cfg.route_obs_group = 'amp_route'
        self.algorithm.amp_cfg.discriminators = {
            'general': {
                'route_value': 0, 'policy_obs_set': 'discriminator',
                'demonstration_obs_set': 'discriminator_demonstration',
                'amp_discriminator': {'hidden_dims': [512, 256]},
            },
            'compliance': {
                'route_value': 1, 'policy_obs_set': 'discriminator_lower',
                'demonstration_obs_set': 'discriminator_lower_demonstration',
                'amp_discriminator': {'hidden_dims': [512, 256]},
            },
        }


def mixed_runner_cfg():
    """Keep existing Mixed play commands looking in their original log folder."""
    return RP1AmpSoftRunnerCfg(experiment_name='rp1_amp_mixed', wandb_project='rp1_amp_mixed')


def interaction_runner_cfg():
    return RP1AmpSoftRunnerCfg(experiment_name='rp1_amp_interaction', wandb_project='rp1_amp_interaction')
