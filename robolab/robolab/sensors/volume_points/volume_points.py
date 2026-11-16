from __future__ import annotations

import re
import torch
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab_newton.physics.newton_manager import NewtonManager

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    from isaaclab.markers import VisualizationMarkersCfg

    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(SensorBase):
    """Volume Points sensor for detecting volume points in a simulation."""

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)

        # Initialize the volume points
        self._volume_points = None
        self._newton_body_indices: torch.Tensor | None = None

    """
    Properties
    """

    @property
    def data(self) -> VolumePointsData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with volume points sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with volume points sensors attached."""
        return self._newton_body_names

    """
    Operations
    """

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        """Record the edges buffer to the sensor so that the penetration data can be updated.

        NOTE: typically this is called by a startup event.

        """
        self._virtual_obstacles.update(virtual_obstacles)

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        ...

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Implementation
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        model = NewtonManager.get_model()
        leaf_pattern = re.compile(f"^{self.cfg.prim_path.rsplit('/', 1)[-1]}$")
        env_body_indices = []
        for env_idx in range(self._num_envs):
            matches = [
                (idx, label.rsplit("/", 1)[-1])
                for idx, label in enumerate(model.body_label)
                if f"/env_{env_idx}/" in label and leaf_pattern.match(label.rsplit("/", 1)[-1])
            ]
            if not matches or (env_body_indices and len(matches) != len(env_body_indices[0])):
                raise RuntimeError(f"VolumePoints could not resolve consistent Newton bodies: {self.cfg.prim_path}")
            env_body_indices.append([index for index, _ in matches])
            if env_idx == 0:
                self._newton_body_names = [name for _, name in matches]
        self._newton_body_indices = torch.tensor(env_body_indices, dtype=torch.long, device=self.device)
        self._num_bodies = len(self._newton_body_names)

        # initialize the volume points data
        self._volume_points_pattern: torch.Tensor = self.cfg.points_generator.func(self.cfg.points_generator).to(
            self.device
        )  # (P, 3)
        self._data: VolumePointsData = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )

        # initialize handlers to access virtual obstacles
        self._virtual_obstacles: dict = dict()

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        # update the volume points data
        self._refresh_volume_points(env_ids)
        # update the penetration depth data
        self._refresh_penetration_offset(env_ids)

    def _refresh_volume_points(self, env_ids: Sequence[int] | None = None) -> None:
        """Refresh the volume points data. If env_ids is None, refresh all environments."""

        state = NewtonManager.get_state_0()
        body_indices = self._newton_body_indices[env_ids]
        body_poses = wp.to_torch(state.body_q)[body_indices]
        body_vels = wp.to_torch(state.body_qd)[body_indices]
        self._data.pos_w[env_ids] = body_poses[..., :3]  # (N_, B, 3)
        self._data.quat_w[env_ids] = body_poses[..., 3:]
        self._data.vel_w[env_ids] = body_vels[..., :3]  # (N_, B, 3)
        self._data.ang_vel_w[env_ids] = body_vels[..., 3:]  # (N_, B, 3)

        # calculate the volume points positions and velocities in world frame
        N_B = self._data.pos_w[env_ids].shape[0] * self._data.pos_w[env_ids].shape[1]  # (N_*B)
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(N_B, -1, -1),  # (N_*B, P, 3)
            self._data.pos_w[env_ids].flatten(0, 1),  # (N_*B, 3)
            self._data.quat_w[env_ids].flatten(0, 1),  # (N_*B, 4)
        ).reshape(
            *self._data.pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3
        )  # (N_, B, P, 3)
        self._data.points_pos_w[env_ids] = points_pos_w
        points_vel_w = self._data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()  # (N_, B, P, 3)
        points_vel_w += torch.linalg.cross(
            self._data.ang_vel_w[env_ids].unsqueeze(-2),
            (points_pos_w - self._data.pos_w[env_ids].unsqueeze(-2)),
            dim=-1,
        )  # (N_, B, P, 3)
        self._data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: Sequence[int]) -> None:
        """Refresh the penetration depth data. If env_ids is None, refresh all environments."""

        penetration_offset_buf: torch.Tensor = self._data.penetration_offset[env_ids]
        penetration_offset_buf[:] = 0.0
        penetration_depth_buf = torch.zeros_like(penetration_offset_buf[..., 0])  # (N_, B, P)

        for virtual_obstacle in self._virtual_obstacles.values():
            # get the penetration offset for the virtual obstacle
            penetration_offset = virtual_obstacle.get_points_penetration_offset(
                self._data.points_pos_w[env_ids].flatten(0, 2)
            )  # (N_*B*P, 3)
            penetration_offset = penetration_offset.reshape(self._data.points_pos_w[env_ids].shape)  # (N_, B, P, 3)
            penetration_depth = torch.norm(penetration_offset, dim=-1)  # (N_, B, P)
            # update the penetration offset if the depth is greater than the current depth
            mask = penetration_depth > penetration_depth_buf  # (N_, B, P)
            penetration_depth_buf[mask] = penetration_depth[mask]
            penetration_offset_buf[mask] = penetration_offset[mask]

        self._data.penetration_offset[env_ids] = penetration_offset_buf

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.points_visualizer.set_visibility(True)
        else:
            if hasattr(self, "points_visualizer"):
                self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self._newton_body_indices is None:
            return

        points = self._data.points_pos_w.view(-1, 3)  # (N_*B*P, 3)
        penetrated = torch.norm(self._data.penetration_offset.view(-1, 3), dim=-1) > 0.0  # (N_*B*P,)

        # add penetrated points if none
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=self.device)], dim=0)

        self.points_visualizer.visualize(
            translations=points,
            marker_indices=penetrated.long(),
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        if hasattr(self, "points_visualizer"):
            delattr(self, "points_visualizer")
        self._newton_body_indices = None
