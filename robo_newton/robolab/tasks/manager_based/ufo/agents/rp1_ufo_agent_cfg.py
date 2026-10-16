"""Runner/agent configuration for the UFO FB-CPR-Aux task (RP1)."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlBaseRunnerCfg

from .fb_cpr_cfg import (
    FbCprDiscriminatorCfg,
    FbCprMlpModelCfg,
    FbCprResidualModelCfg,
)


@configclass
class FbCprAlgorithmCfg:
    """Placeholder algorithm config for the unified entrypoints.

    The FB-CPR runner builds its own agent; only ``class_name`` is inspected by
    the entrypoint helpers (e.g. the Distillation checkpoint selector).
    """

    class_name: str = "FbCprAuxAgent"


@configclass
class RP1UFORunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "FbCprRunner"
    algorithm: FbCprAlgorithmCfg = FbCprAlgorithmCfg()

    experiment_name: str = "rp1_ufo"
    wandb_project: str = "rp1_ufo"
    logger: str = "wandb"

    num_steps_per_env: int = 1
    max_iterations: int = 1_000_000
    num_env_steps: int = 1_000_000_000
    checkpoint_interval_steps: int = 9_600_000
    checkpoint_replay: bool = True
    replay_checkpoint_keep: int = 1
    resume_replay_warmup_steps: int = 10_240
    seed: int = 52

    # Estimate yaw-frame root velocity from the proprioceptive history.  UFO
    # has no depth encoder, but uses the same supervised target as PUFO.
    estimated_velocity_enabled: bool = True
    velocity_estimator = FbCprMlpModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        first_activation="elu",
        layer_norm=False,
    )
    velocity_estimator_loss_coef: float = 1.0
    actor_obs_keys: tuple[str, ...] = (
        "state", "last_action", "history_actor", "heading", "velocity_command",
    )
    privileged_obs_keys: tuple[str, ...] = ("state", "privileged_state")
    full_obs_keys: tuple[str, ...] = (
        "state", "privileged_state", "last_action", "history_actor",
        "heading", "velocity_command", "velocity_target_yaw",
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
    discriminator: FbCprDiscriminatorCfg = FbCprDiscriminatorCfg(
        hidden_dims=[1024, 1024, 1024],
        activation="relu",
        first_activation="tanh",
        layer_norm=True,
        loss_type="bce",
        reward_type="log_odds",
    )
    # Share one trunk between the discriminator and auxiliary critic heads.
    # Both losses are optimized together in one backward/optimizer step.
    # Match H200 FB-CPR: discriminator and auxiliary critics have independent trunks.
    shared_value_net: bool = False
    heading_critic_enabled: bool = True
    velocity_critic_enabled: bool = True
    shared_context_value_net: bool = True
    heading_critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=3,
        embedding_layers=3,
        activation="mish",
        layer_norm=True,
    )
    velocity_critic = FbCprResidualModelCfg(
        hidden_dim=1024,
        hidden_layers=3,
        embedding_layers=3,
        activation="mish",
        layer_norm=True,
    )
    lr_heading_critic: float = 3e-4
    lr_velocity_critic: float = 3e-4
    heading_critic_pessimism_penalty: float = 0.5
    velocity_critic_pessimism_penalty: float = 0.5
    reg_coeff_heading: float = 0.01
    reg_coeff_velocity: float = 0.01
    velocity_error_scale_mps: float = 0.5
    # Keep two independently parameterized members for pessimistic estimates.
    critic_ensemble_members: int = 2
    forward_ensemble_members: int = 2
    batch_size: int = 1024
    discount: float = 0.98
    buffer_size: int = 5_120_000
    num_seed_steps: int = 10_240
    num_agent_updates: int = 16
    # Per-rank environment samples collected between agent-update phases.
    # Matches the original UFO FB runtime (update_agent_every = 1024 samples).
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
    lr_discriminator: float = 2e-6
    lr_critic: float = 3e-4
    lr_aux_critic: float = 3e-4
    weight_decay: float = 0.0
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0

    fb_target_tau: float = 0.01
    critic_target_tau: float = 0.005
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    aux_critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 0.01
    reg_coeff_aux: float = 0.02
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
    # Match WBC precision handling: fp32 master weights and optimizer state,
    # with update-network forwards under bf16 autocast.
    use_bf16_amp: bool = True
    # "high" enables TF32 for fp32 matrix multiplications; use "highest" for pure fp32.
    matmul_precision: str = "high"

    # Original 384M run evaluated every 3.2M env steps; 400 iterations x 8192
    # env steps = 3.28M.
    eval_interval_steps: int = 3_200_000
    log_interval_steps: int = 384_000
    eval_num_envs: int = 256
    eval_timeout_s: float = 1800.0
    eval_num_motions: int = 0

    # Original-UFO EMD prioritization (workspace.py): per-clip joint-space EMD
    # from the tracking eval feeds clip sampling weights.
    prioritization: bool = True
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 2.0
    prioritization_scale: float = 1.5
    prioritization_mode: str = "exp"  # lin | exp | bin
