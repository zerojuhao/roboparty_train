"""Runner/agent configuration for the UFO FB-CPR-Aux task (RP1)."""

from __future__ import annotations

import os

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlBaseRunnerCfg

from .fb_cpr_cfg import FbCprDiscriminatorCfg

AUX_REWARDS = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "height_gated_undesired_contacts",
    "penalty_slippage",
    # Regularization terms shared with the WBC task.
    "joint_acc_l2",
    "joint_velocity_l2",
    "joint_energy",
    "joint_rated_velocity",
    "joint_rated_effort",
    "action_jerk_l2",
    "angular_momentum_rate_l2",
    "yaw_angular_momentum_l2",
)

# Aux reward aggregation scalings. The eight original UFO terms use the
# original FB agent preset values (humanoidverse/agents/presets/fb.py,
# cartwheel_aux_safe=False), except penalty_torques which follows the RP1
# WbcRewardsCfg joint_torques_l2 weight (-1e-5) instead of the original 0.0,
# and undesired contacts which use BeyondMimic's per-body count semantics
# with a stronger UFO-specific coefficient.
# The WBC-shared regularization terms keep the RP1 WbcRewardsCfg weights.
AUX_REWARDS_SCALING = {
    "penalty_action_rate": -0.2,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "height_gated_undesired_contacts": -0.5,
    "penalty_torques": -1e-5,
    "joint_acc_l2": -5.0e-7,
    "joint_velocity_l2": -2e-4,
    "joint_energy": -1e-4,
    "joint_rated_velocity": -0.05,
    "joint_rated_effort": -0.04,
    "action_jerk_l2": -0.02,
    "angular_momentum_rate_l2": -5e-5,
    "yaw_angular_momentum_l2": -1e-3,
}


@configclass
class FbCprAlgorithmCfg:
    """Placeholder algorithm config for the unified entrypoints.

    The FB-CPR runner builds its own agent; only ``class_name`` is inspected by
    the entrypoint helpers (e.g. the Distillation checkpoint selector).
    """

    class_name: str = "FbCprAux"


@configclass
class RP1UFORunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "FbCprRunner"
    algorithm: FbCprAlgorithmCfg = FbCprAlgorithmCfg()

    experiment_name: str = "rp1_ufo"
    wandb_project: str = "rp1_ufo"

    num_steps_per_env: int = 1
    max_iterations: int = 375000
    num_env_steps: int = 384_000_000
    save_interval: int = 2000
    seed: int = 52

    z_dim: int = 256
    norm_z: bool = True
    # Network sizes are explicit so each branch can be tuned without changing
    # the FB agent implementation.
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
    # Share one trunk between the discriminator and auxiliary critic heads.
    # Both losses are optimized together in one backward/optimizer step.
    shared_value_net: bool = True
    # Keep two independently parameterized members for pessimistic estimates.
    critic_ensemble_members: int = 2
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
    eval_interval_iterations: int = 400
    eval_num_motions: int = 0

    # Original-UFO EMD prioritization (workspace.py): per-clip joint-space EMD
    # from the tracking eval feeds clip sampling weights.
    prioritization: bool = True
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 2.0
    prioritization_scale: float = 1.5
    prioritization_mode: str = "exp"  # lin | exp | bin

    aux_rewards: list[str] = list(AUX_REWARDS)
    aux_rewards_scaling: dict[str, float] = dict(AUX_REWARDS_SCALING)

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        bf16 = os.environ.get("UFO_BF16_AMP")
        if bf16 is not None:
            self.use_bf16_amp = bf16.strip().lower() in ("1", "true", "yes")
            print(f"[UFO_PREC] use_bf16_amp -> {self.use_bf16_amp}", flush=True)
        precision = os.environ.get("UFO_MATMUL_PRECISION")
        if precision:
            self.matmul_precision = precision.strip().lower()
            print(f"[UFO_PREC] matmul_precision -> {self.matmul_precision}", flush=True)
        fused_adam = os.environ.get("UFO_FUSED_ADAM")
        if fused_adam is not None:
            self.use_fused_adam = fused_adam.strip().lower() in ("1", "true", "yes")
            print(f"[UFO_PERF] use_fused_adam -> {self.use_fused_adam}", flush=True)
        compile_mode = os.environ.get("UFO_TORCH_COMPILE_MODE")
        if compile_mode is not None:
            compile_mode = compile_mode.strip().lower()
            self.torch_compile_mode = None if compile_mode in ("", "0", "none", "off") else compile_mode
            print(f"[UFO_PERF] torch_compile_mode -> {self.torch_compile_mode}", flush=True)
