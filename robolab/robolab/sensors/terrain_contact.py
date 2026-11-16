"""Body contact reporting plus unmerged, all-contact stumble statistics."""

import math
from dataclasses import fields

import numpy as np
import warp as wp

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.contact_sensor.contact_sensor import ContactSensor


def _base_contact_cfg(cfg):
    # Backend from_base_cfg forwards every dataclass field to its constructor.
    # Keep project-only settings on our instance, not in the backend config.
    return ContactSensorCfg(**{field.name: getattr(cfg, field.name) for field in fields(ContactSensorCfg)})


@wp.kernel
def reduce_contact_excess(
    count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int],
    forces: wp.array[wp.spatial_vector], shape_rows: wp.array[int],
    env_mask: wp.array[bool], bodies_per_env: int, horizontal_ratio: float,
    excess: wp.array[float],
):
    contact = wp.tid()
    if contact >= count[0]:
        return
    force = wp.spatial_top(forces[contact])
    value = wp.max(wp.length(wp.vec2(force[0], force[1])) - horizontal_ratio * wp.abs(force[2]), 0.0)
    # Atomic max must not silently discard NaN contacts as zero penalty.
    if not wp.isfinite(force[0]) or not wp.isfinite(force[1]) or not wp.isfinite(force[2]):
        value = wp.inf
    for side in range(2):
        shape = shape0[contact]
        if side == 1:
            shape = shape1[contact]
        if shape >= 0:
            row = shape_rows[shape]
            if row >= 0 and env_mask[row // bodies_per_env]:
                wp.atomic_max(excess, row, value)


@wp.kernel
def append_excess_history(
    excess: wp.array[float], env_mask: wp.array[bool],
    bodies_per_env: int, history_length: int, history: wp.array3d[float],
):
    env, body = wp.tid()
    if env_mask[env]:
        for offset in range(history_length - 1):
            t = history_length - 1 - offset
            history[env, t, body] = history[env, t - 1, body]
        history[env, 0, body] = excess[env * bodies_per_env + body]


@wp.kernel
def reset_excess_history(env_mask: wp.array[bool], history: wp.array3d[float]):
    env, t, body = wp.tid()
    if env_mask[env]:
        history[env, t, body] = 0.0


class NewtonTerrainContactSensor(ContactSensor):
    """Preserve standard body forces/timers; reduce individual contacts separately.

    History has the base sensor's body ordering. Storing horizontal excess
    permits reward thresholds/scales to change without retaining force vectors.
    """

    def __init__(self, cfg):
        self.stumble_horizontal_ratio = getattr(cfg, "stumble_horizontal_ratio", 1.0)
        if not math.isfinite(self.stumble_horizontal_ratio) or self.stumble_horizontal_ratio < 0:
            raise ValueError("stumble_horizontal_ratio must be finite and nonnegative")
        super().__init__(_base_contact_cfg(cfg))

    def _create_buffers(self):
        super()._create_buffers()
        if self.contact_view.sensing_type != "body":
            raise ValueError("Terrain stumble statistics require body-level contact sensing")
        model = NewtonManager.backend.model
        body_rows = np.full(model.body_count, -1, dtype=np.int32)
        sensing = np.asarray(self.contact_view.sensing_indices, dtype=np.int32)
        body_rows[sensing] = np.arange(len(sensing), dtype=np.int32)
        shape_bodies = model.shape_body.numpy()
        rows = np.full(len(shape_bodies), -1, dtype=np.int32)
        valid = shape_bodies >= 0
        rows[valid] = body_rows[shape_bodies[valid]]
        self._contact_shape_rows = wp.array(rows, dtype=int, device=self._device)
        self._contact_excess = wp.zeros(self._num_envs * self._num_sensors, device=self._device)
        self._contact_excess_history = wp.zeros(
            (self._num_envs, self._history_length, self._num_sensors), device=self._device,
        )
        self._data.contact_excess_history = wp.to_torch(self._contact_excess_history)

    def _update_buffers_impl(self, env_mask):
        super()._update_buffers_impl(env_mask)
        contacts = NewtonManager._contacts
        if contacts is None or contacts.force is None:
            raise RuntimeError("Terrain stumble requires Newton per-contact force reporting")
        self._contact_excess.zero_()
        wp.launch(reduce_contact_excess, dim=contacts.rigid_contact_max,
                  inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0,
                          contacts.rigid_contact_shape1, contacts.force, self._contact_shape_rows,
                          env_mask, self._num_sensors, self.stumble_horizontal_ratio, self._contact_excess],
                  device=self._device)
        wp.launch(append_excess_history, dim=(self._num_envs, self._num_sensors),
                  inputs=[self._contact_excess, env_mask, self._num_sensors,
                          self._history_length, self._contact_excess_history], device=self._device)

    def reset(self, env_ids=None, env_mask=None):
        mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(env_mask=mask)
        wp.launch(reset_excess_history, dim=self._contact_excess_history.shape,
                  inputs=[mask, self._contact_excess_history], device=self._device)


class TerrainContactSensor:
    """Use the extra statistics on Newton without replacing PhysX sensing."""

    def __new__(cls, cfg):
        import isaaclab.sim as sim_utils
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab.sensors import ContactSensor as BackendContactSensor

        if isinstance(sim_utils.SimulationContext.instance().cfg.physics, NewtonCfg):
            return NewtonTerrainContactSensor(cfg)
        return BackendContactSensor(_base_contact_cfg(cfg))


@configclass
class TerrainContactSensorCfg(ContactSensorCfg):
    class_type: type = TerrainContactSensor
    stumble_horizontal_ratio: float = 1.0
