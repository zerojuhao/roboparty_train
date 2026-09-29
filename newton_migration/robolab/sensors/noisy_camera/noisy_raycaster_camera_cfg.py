from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, RayCasterCfg
from isaaclab.utils import configclass

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_raycaster_camera import NoisyRayCasterCamera


@configclass
class NoisyRayCasterCameraCfg(NoisyCameraCfgMixin, RayCasterCameraCfg):
    """
    Configuration class for the NoisyRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type = NoisyRayCasterCamera
