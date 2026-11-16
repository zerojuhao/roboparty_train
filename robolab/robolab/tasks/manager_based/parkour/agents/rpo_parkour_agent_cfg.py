from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from robolab.tasks.manager_based.amp.agents.rsl_rl_amp_cfg import (
    RslRlAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from robolab.tasks.manager_based.parkour.mdp.symmetry import rpo

@configclass
class RslRlEncoderMoEModelCfg(RslRlMLPModelCfg):
    class_name: str = "rsl_rl.models.encoder_model:EncoderMoEModel"
    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = []
    hidden_dims: list[int] = [256, 128, 64]
    obs_normalization: bool = False
    activation: str = "elu"
    encoder_obs_groups: list[str] = ["depth"]
    encoder_cfg: dict = {
        "channels": [16, 32, 64],
        "kernel_sizes": [3, 3, 3],
        "strides": [1, 2, 1],
        "hidden_sizes": [256],
        "output_size": 256,
        "paddings": [1, 1, 1],
        "nonlinearity": "ReLU",
        "use_maxpool": True,
        "last_activation": "ReLU",
        "input_dtype": "uint8",
        "input_range": (0.0, 255.0),
    }


@configclass
class RPOParkourAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    export_policy_func: str = "robolab.tasks.manager_based.parkour.agents.exporter:export_policy"
    num_steps_per_env = 24
    compact_depth_history = True
    depth_history_offsets = (35, 30, 25, 20, 15, 10, 5, 0)
    check_for_nan_interval = 100
    torch_compile_mode: str | None = "default"
    max_iterations = 30000
    save_interval = 500
    experiment_name = "rpo_parkour"
    wandb_project = "rpo_parkour"
    obs_groups = {
        "actor": ["policy", "depth"],
        "critic": ["critic"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }
    actor = RslRlEncoderMoEModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="rsl_rl.algorithms.ppo_amp:PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
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
