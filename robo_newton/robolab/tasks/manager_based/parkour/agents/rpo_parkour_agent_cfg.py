from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.amp.agents.rpo_amp_agent_cfg import (
    RslRlAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from robolab.tasks.manager_based.parkour.mdp.symmetry import rpo
from .parkour_model_cfg import ParkourModelCfg


@configclass
class RPOParkourAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "rpo_parkour"
    wandb_project = "rpo_parkour"
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    actor = ParkourModelCfg(
        hidden_dims=[256, 128, 64],
        obs_normalization=True,
        distribution_cfg=ParkourModelCfg.GaussianDistributionCfg(init_std=1.0),
        encoder_cfg={
            "channels": [4], "kernel_sizes": [3], "strides": [1],
            "hidden_sizes": [256, 256], "output_size": 128, "paddings": [1],
            "nonlinearity": "ReLU", "use_maxpool": True, "last_activation": "ReLU",
        },
        enable_velocity_estimation=True,
    )
    critic = ParkourModelCfg(
        hidden_dims=[256, 128, 64],
        obs_normalization=True,
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        enable_aux_loss=True,
        aux_loss_coef=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=rpo.compute_symmetric_states
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=5.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-2,
            disc_learning_rate=1.0e-4,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="ReLU",
                style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            loss_type="LSGAN",
        ),
    )
