"""Runner configuration for the H200 PUFO visual training branch."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlBaseRunnerCfg

from .fb_cpr_cfg import (
    FbCprDepthEncoderCfg,
    FbCprDiscriminatorCfg,
    FbCprMlpModelCfg,
    FbCprResidualModelCfg,
)


@configclass
class PufoAlgorithmCfg:
    class_name: str = "FbCprAuxAgent"


@configclass
class RP1PUFORunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "FbCprRunner"
    algorithm: PufoAlgorithmCfg = PufoAlgorithmCfg()

    experiment_name: str = "rp1_pufo"
    wandb_project: str = "rp1_pufo"
    logger: str = "wandb"
    num_steps_per_env: int = 1
    max_iterations: int = 1_000_000
    num_env_steps: int = 1_000_000_000
    checkpoint_interval_steps: int = 9_600_000
    checkpoint_replay: bool = False
    replay_checkpoint_keep: int = 1
    resume_replay_warmup_steps: int = 10_240
    seed: int = 42

    visual_enabled: bool = True
    estimated_velocity_enabled: bool = True
    velocity_estimator_updates_encoder: bool = True
    actor_obs_keys: tuple[str, ...] = (
        "state",
        "last_action",
        "history_actor",
        "heading",
        "velocity_command",
    )
    privileged_obs_keys: tuple[str, ...] = ("state", "privileged_state")
    full_obs_keys: tuple[str, ...] = (
        "state",
        "privileged_state",
        "last_action",
        "history_actor",
        "terrain_priv",
        "heading",
        "velocity_command",
        "velocity_target_yaw",
    )

    z_dim: int = 256
    norm_z: bool = True
    actor = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=2,
        activation="mish",
        layer_norm=True,
    )
    depth_encoder = FbCprDepthEncoderCfg(
        input_shape=(8, 18, 36),
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 1],
        paddings=[1, 1, 1],
        pool_shape=(2, 4),
        hidden_dims=[256],
        activation="relu",
        output_size=256,
        gru_layers=1,
    )
    # Estimate the robot's current yaw-frame vx/vy and vertical velocity.
    velocity_estimator = FbCprMlpModelCfg(
        hidden_dims=[512, 256],
        activation="elu",
        first_activation="elu",
        layer_norm=False,
    )
    forward_map = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=6,
        activation="mish",
        layer_norm=True,
    )
    backward_map = FbCprMlpModelCfg(
        hidden_dims=[256],
        activation="relu",
        first_activation="tanh",
        layer_norm=True,
    )
    critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=6,
        activation="mish",
        layer_norm=True,
    )
    # Used only when shared_value_net=False; otherwise critic defines the shared trunk.
    aux_critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=6,
        embedding_layers=6,
        activation="mish",
        layer_norm=True,
    )
    heading_critic_enabled: bool = True
    # Share heading/velocity trunks when both are enabled; heading config/lr owns the trunk.
    shared_context_value_net: bool = True
    heading_critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=3,
        embedding_layers=3,
        activation="mish",
        layer_norm=True,
    )
    velocity_critic_enabled: bool = True
    # Used only with separate context critics (or velocity alone).
    velocity_critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=3,
        embedding_layers=3,
        activation="mish",
        layer_norm=True,
    )
    discriminator: FbCprDiscriminatorCfg = FbCprDiscriminatorCfg(
        hidden_dims=[1024, 1024, 1024],
        activation="relu",
        first_activation="tanh",
        layer_norm=True,
        loss_type="bce",
        reward_type="log_odds",
    )
    critic_ensemble_members: int = 2
    forward_ensemble_members: int = 2
    # Share the discriminator-reward and auxiliary Q trunk, separate from the context trunk.
    shared_value_net: bool = True
    batch_size: int = 1024
    discount: float = 0.98
    buffer_size: int = 5_120_000
    num_seed_steps: int = 10_240
    num_agent_updates: int = 16
    update_agent_every_samples: int = 1024
    update_z_every_step: int = 100
    z_buffer_size: int = 8192
    actor_std: float = 0.05
    action_rescale: float = 8.0
    seq_length: int = 8
    use_mix_rollout: bool = True
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5

    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    lr_discriminator: float = 5e-6
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
    lr_heading_critic: float = 3e-4  # Also the shared context critic learning rate.
    lr_velocity_critic: float = 3e-4  # Used only with separate context critics.
    weight_decay: float = 0.0
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0

    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    heading_critic_pessimism_penalty: float = 0.5
    velocity_critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 0.01
    reg_coeff_aux: float = 0.02
    reg_coeff_heading: float = 0.002
    reg_coeff_velocity: float = 0.002
    velocity_error_scale_mps: float = 0.5
    velocity_estimator_loss_coef: float = 1.0
    scale_reg: bool = True
    ortho_coef: float = 100.0
    stddev_clip: float = 0.3
    relabel_ratio: float | None = 0.8
    expert_asm_ratio: float = 0.6
    train_goal_ratio: float = 0.2
    clip_grad_norm: float = 0.0
    grad_bucket_size_mb: int = 128
    use_fused_adam: bool = True
    torch_compile_mode: str | None = "default"
    use_bf16_amp: bool = True
    matmul_precision: str = "high"

    eval_interval_steps: int = 3_200_000
    # Keep train/* and Reward/* at one point per update phase.
    log_every_update: bool = True
    # Controls sparse depth-gradient diagnostics, not the main train/reward cadence.
    log_interval_steps: int = 384_000
    # One minibatch in this relative update phase; zero disables profiling.
    profile_update_phase: int = 0
    # -1 captures the same minibatch on every rank.
    profile_update_rank: int = 0
    eval_num_motions: int = 0
    eval_num_envs: int = 256
    eval_reference_chunk: int = 128
    eval_timeout_s: float = 1800.0
    prioritization: bool = True
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 2.0
    prioritization_scale: float = 2.0
    prioritization_mode: str = "exp"
