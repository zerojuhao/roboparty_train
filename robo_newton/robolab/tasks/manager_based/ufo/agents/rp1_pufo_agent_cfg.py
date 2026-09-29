"""Runner configuration for the H200 PUFO visual training branch."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlBaseRunnerCfg

from .fb_cpr_cfg import FbCprDiscriminatorCfg


@configclass
class PufoAlgorithmCfg:
    class_name: str = "FbCprAux"


@configclass
class RP1PUFORunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "FbCprRunner"
    algorithm: PufoAlgorithmCfg = PufoAlgorithmCfg()

    experiment_name: str = "rp1_pufo"
    wandb_project: str = "rp1_pufo"
    num_steps_per_env: int = 1
    max_iterations: int = 375000
    save_interval: int = 2000
    seed: int = 42

    visual_enabled: bool = True
    depth_shape: tuple[int, int, int] = (8, 18, 32)
    estimated_velocity_enabled: bool = True
    actor_obs_keys: tuple[str, ...] = ("state", "last_action", "history_actor", "heading", "velocity_command")
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
    actor_hidden_dim: int = 1024
    actor_hidden_layers: int = 6
    forward_hidden_dim: int = 1024
    forward_hidden_layers: int = 6
    critic_hidden_dim: int = 1024
    critic_hidden_layers: int = 6
    backward_hidden_dim: int = 256
    backward_hidden_layers: int = 1
    net_embedding_layers: int = 2
    discriminator: FbCprDiscriminatorCfg = FbCprDiscriminatorCfg(
        hidden_dims=[1024, 1024, 1024],
        loss_type="bce",
        reward_type="log_odds",
    )
    critic_ensemble_members: int = 2
    shared_value_net: bool = True
    context_critic_hidden_dim: int = 1024
    context_critic_layers: int = 3
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
    lr_heading_critic: float = 3e-4
    lr_velocity_critic: float = 3e-4
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

    eval_interval_iterations: int = 400
    eval_num_motions: int = 0
    prioritization: bool = False
    aux_rewards: list[str] = [
        "penalty_action_rate",
        "limits_dof_pos",
        "penalty_body_impact",
        "penalty_slippage",
        "penalty_ankle_roll",
        "feet_stumble",
    ]
    aux_rewards_scaling: dict[str, float] = {
        "penalty_action_rate": -0.1,
        "limits_dof_pos": -10.0,
        "penalty_body_impact": -1.0,
        "penalty_slippage": -2.0,
        "penalty_ankle_roll": -1.0,
        "feet_stumble": -2.0,
    }
