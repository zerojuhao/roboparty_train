"""Model configuration for the AMP velocity estimator."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class AMPModelCfg(RslRlMLPModelCfg):
    @configclass
    class GaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
        # Match RSL-RL defaults; experiments can limit sampling without changing the mean policy.
        std_range: tuple[float, float] = (1.0e-6, 1.0e6)

    class_name: str = "rsl_rl.models.amp_model:AMPModel"
    enable_velocity_estimation: bool = True
    velocity_estimation_hidden_dims: list[int] = [512, 256]
    # Filled from the resolved critic observations when constructing PPOAMP.
    velocity_target_indices: tuple[int, int, int] | None = None
    enable_force_estimation: bool = False
    force_estimation_hidden_dims: list[int] = [512, 256]
    force_estimation_history_length: int = 8
    force_estimation_with_torque: bool = False
    force_estimation_multi_contact: bool = False
    force_estimation_compact: bool = False
    enable_force_vel_target_estimation: bool = False
    # Weight of unloaded samples in the compact load-detection loss (1.0 = balanced, legacy).
    force_load_negative_weight: float = 1.0
