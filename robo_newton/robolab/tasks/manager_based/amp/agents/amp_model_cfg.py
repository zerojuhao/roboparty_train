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
    velocity_estimation_hidden_dims: list[int] = [128, 64]
    # Filled from the resolved critic observations when constructing PPOAMP.
    velocity_target_indices: tuple[int, int] | None = None
