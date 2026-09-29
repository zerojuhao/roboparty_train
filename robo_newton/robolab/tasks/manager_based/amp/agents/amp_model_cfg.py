"""Model configuration for the AMP velocity estimator."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class AMPModelCfg(RslRlMLPModelCfg):
    class_name: str = "rsl_rl.models.amp_model:AMPModel"
    enable_velocity_estimation: bool = True
    velocity_estimation_hidden_dims: list[int] = [128, 64]
    critic_history_length: int = 3
