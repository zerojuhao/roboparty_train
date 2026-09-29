"""Native RSL-RL model configuration for depth-based Parkour policies."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class ParkourModelCfg(RslRlMLPModelCfg):
    class_name: str = "ParkourModel"
    activation: str = "elu"
    encoder_cfg: dict | None = None
    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = []
    enable_velocity_estimation: bool = False
    velocity_estimation_history_length: int = 8
    velocity_estimation_estimator_hidden_dims: list[int] = [512, 256]
    aux_velocity_coef: float = 1.0
    enable_ssr_estimation: bool = False
    proprio_history_length: int = 8
    estimation_estimator_hidden_dims: list[int] = [512, 256]
    estimation_decoder_hidden_dims: list[int] = [256, 128]
    enable_actor_foothold: bool = True
    enable_actor_foot_height: bool = True
    estimation_foot_height_actor_dim: int = 16
    gate_foothold_estimation_on_predictor_ready: bool = True
    aux_foot_height_coef: float = 1.0
    aux_foothold_coef: float = 2.0
