"""RP1 foot slip from individual Newton contact points, not ankle motion."""

import numpy as np
import torch
import warp as wp

from isaaclab.managers import ManagerTermBase
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.contact_sensor.contact_sensor import ContactSensor

from robolab.utils.tensors import as_torch_tensor


@wp.func
def point_velocity(
    body: int, point: wp.vec3, body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
):
    if body < 0:
        return wp.vec3(0.0)
    arm = wp.transform_vector(body_q[body], point - body_com[body])
    return wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), arm)


@wp.kernel
def accumulate_contact_slip(
    count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int],
    point0: wp.array[wp.vec3], point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3], offset1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3], forces: wp.array[wp.spatial_vector],
    shape_bodies: wp.array[int], shape_rows: wp.array[int],
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3], normal_load: wp.array[float], weighted_speed: wp.array[float],
):
    contact = wp.tid()
    if contact >= count[0]:
        return
    a, b = shape0[contact], shape1[contact]
    row_a, row_b = shape_rows[a], shape_rows[b]
    # -1 is an external surface; -2 is another robot body. Ignore self-contact.
    row = int(-1)
    if row_a >= 0 and row_b == -1:
        row = row_a
    elif row_b >= 0 and row_a == -1:
        row = row_b
    if row < 0:
        return

    normal = wp.normalize(normals[contact])
    load = wp.abs(wp.dot(wp.spatial_top(forces[contact]), normal))
    if load <= 0.0:
        return
    velocity_a = point_velocity(
        shape_bodies[a], point0[contact] + offset0[contact], body_q, body_qd, body_com,
    )
    velocity_b = point_velocity(
        shape_bodies[b], point1[contact] + offset1[contact], body_q, body_qd, body_com,
    )
    relative = velocity_a - velocity_b
    tangent = relative - wp.dot(relative, normal) * normal
    wp.atomic_add(normal_load, row, load)
    # Average magnitudes, not vectors: opposite sliding at two sole edges must not cancel.
    wp.atomic_add(weighted_speed, row, load * wp.length(tangent))


def loaded_slip_penalty(normal_load, weighted_speed, body_mass, gravity, force_threshold_ratio):
    """Sum each loaded foot's force-weighted mean tangential speed (m/s)."""
    weight = body_mass.sum(dim=-1, keepdim=True) * gravity
    loaded = normal_load > force_threshold_ratio * weight
    speed = weighted_speed / normal_load.clamp_min(1.0e-6)
    return torch.where(loaded, speed, 0.0).sum(dim=-1)


class feet_contact_slip(ManagerTermBase):
    """Instantaneous contact-point slip; no stance anchors or displacement clipping.

    Uses the existing all-robot contact sensor to resolve global body indices.
    Contact forces and point kinematics come directly from Newton. The reward
    keeps the previous last-physics-substep sampling and sum-over-feet reduction.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        sensor = env.scene.sensors[sensor_cfg.name]
        if not isinstance(sensor, ContactSensor):
            raise NotImplementedError("RP1 contact-point slip requires the Newton contact backend.")
        model = NewtonManager.get_model()
        sensed = np.asarray(sensor.contact_view.sensing_indices).reshape(env.num_envs, -1)
        feet = sensed[:, sensor_cfg.body_ids]
        body_rows = np.full(model.body_count, -1, dtype=np.int32)
        body_rows[sensed.ravel()] = -2
        body_rows[feet.ravel()] = np.arange(feet.size, dtype=np.int32)
        shape_bodies = model.shape_body.numpy()
        shape_rows = np.full(model.shape_count, -1, dtype=np.int32)
        attached = shape_bodies >= 0
        shape_rows[attached] = body_rows[shape_bodies[attached]]
        self.shape_rows = wp.array(shape_rows, dtype=int, device=env.device)
        self.normal_load = wp.zeros(feet.size, device=env.device)
        self.weighted_speed = wp.zeros(feet.size, device=env.device)
        self.foot_shape = feet.shape
        self.gravity = float(np.linalg.norm(env.cfg.sim.gravity))

    def __call__(self, env, sensor_cfg, asset_cfg, force_threshold_ratio=0.05):
        model = NewtonManager.get_model()
        state = NewtonManager.get_state()
        contacts = NewtonManager.get_contacts()
        self.normal_load.zero_()
        self.weighted_speed.zero_()
        wp.launch(
            accumulate_contact_slip, dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0, contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0, contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal, contacts.force, model.shape_body, self.shape_rows,
                state.body_q, state.body_qd, model.body_com, self.normal_load, self.weighted_speed,
            ],
            device=env.device,
        )
        return loaded_slip_penalty(
            wp.to_torch(self.normal_load).view(self.foot_shape),
            wp.to_torch(self.weighted_speed).view(self.foot_shape),
            as_torch_tensor(env.scene[asset_cfg.name].data.body_mass),
            self.gravity, force_threshold_ratio,
        )
