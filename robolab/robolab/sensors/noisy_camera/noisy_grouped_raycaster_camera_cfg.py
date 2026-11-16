from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg
from isaaclab.utils import configclass

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@configclass
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, MultiMeshRayCasterCameraCfg):
    """
    Configuration class for the NoisyGroupedRayCasterCamera sensor and manages image transforms and their parameters.
    """

    class_type: type = NoisyGroupedRayCasterCamera
