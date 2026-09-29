from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.amp.agents.rpo_amp_agent_cfg import RslRlAmpCfg, RslRlPpoAmpAlgorithmCfg
from robolab.tasks.manager_based.parkour.mdp.foothold_prediction import FootholdGridCfg, FootholdPredictorCfg
from robolab.tasks.manager_based.parkour.mdp.symmetry import rp1
from .parkour_model_cfg import ParkourModelCfg


@configclass
class RslRlMultiRewardPpoAmpAlgorithmCfg(RslRlPpoAmpAlgorithmCfg):
    class_name: str = "MultiRewardPPOAMP"
    num_reward_heads: int = 3
    advantage_weights: list[float] = [1.0, 0.5, 0.5]
    reward_head_names: list[str] = ["locomotion", "foothold", "style"]
    enable_aux_loss: bool = True
    aux_loss_coef: float = 1.0


@configclass
class RP1SSRAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rp1_ssr"
    wandb_project = "rp1_ssr"
    logger = "wandb"
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
            "channels": [16, 32, 64], "kernel_sizes": [3, 3, 3],
            "strides": [2, 2, 1], "hidden_sizes": [256, 256],
            "output_size": 256, "paddings": [1, 1, 1], "nonlinearity": "ReLU",
            "use_maxpool": True, "last_activation": "ReLU",
        },
        num_moe_experts=3,
        enable_ssr_estimation=True,
    )
    critic = ParkourModelCfg(hidden_dims=[256, 128, 64], obs_normalization=True, num_moe_experts=3)
    foothold_imagination: FootholdPredictorCfg = FootholdPredictorCfg(
        enabled=True,
        hidden_dims=[512, 256, 128],
        learning_rate=5.0e-4,
        weight_decay=1.0e-5,
        ema_decay=0.99,
        grid=FootholdGridCfg(
            sigma_min=0.01,
            sigma_max=0.25,
            yaw_sigma_min=0.05,
            yaw_sigma_max=0.5,
            expectation_grid_size=5,
            expectation_std_range=2.0,
            expectation_eval_chunk_size=64,
            expectation_weighting="max",
            reward_sigma_min=0.03,
        ),
        nll_loss_coef=1.0,
        xy_mean_loss_coef=2.0,
        yaw_mean_loss_coef=0.5,
        mean_loss_beta=0.02,
        nll_horizon_tau=32.0,
        nll_horizon_weight_min=0.25,
        guidance_horizon_tau=8.0,
        guidance_horizon_weight_min=0.25,
        max_pending_steps=48,
        pending_sample_stride=1,
        train_pending_tail_steps=0,
        train_buffer_capacity=250_000,
        batch_size=4096,
        updates_per_iteration=8,
        balanced_sampling=True,
        balanced_candidate_multiplier=8,
        min_train_samples=25_000,
        curriculum_level_threshold=1.0,
        enable_xy_rmse_threshold=0.05,
        enable_yaw_rmse_threshold=0.15,
        clear_train_buffer_on_resume=True,
    )
    algorithm = RslRlMultiRewardPpoAmpAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=3,
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
            data_augmentation_func=rp1.compute_symmetric_states,
        ),
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100,
            grad_penalty_scale=10.0,
            disc_trunk_weight_decay=1.0e-5,
            disc_linear_weight_decay=1.0e-3,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[512, 256], activation="ReLU", style_reward_scale=2.0,
                task_style_lerp=0.3,
            ),
            loss_type="LSGAN",
        ),
    )
