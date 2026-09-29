"""Newton multi-mesh depth camera with RoboLab noise and frame history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyMultiMeshRayCasterCamera

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
    from .noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


class NoisyGroupedRayCasterCamera(NoisyCameraMixin, LegacyMultiMeshRayCasterCamera):
    cfg: NoisyGroupedRayCasterCameraCfg

    def _initialize_impl(self):
        super()._initialize_impl()
        self.build_noise_pipeline()
        self.build_history_buffers()

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        super().reset(env_ids=env_ids, env_mask=env_mask)
        if env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    def _update_buffers_impl(self, env_mask: wp.array):
        super()._update_buffers_impl(env_mask)
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)
