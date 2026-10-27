"""Differentiable reconstruction of UFO's full-body privileged observation."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from pathlib import Path
import xml.etree.ElementTree as ET

import pytorch_kinematics as pk
import torch
from torch import nn


@contextmanager
def _geometry_precision(device_type):
    previous = torch.get_float32_matmul_precision()
    try:
        if previous != "highest":
            torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type, enabled=False):
            yield
    finally:
        if previous != "highest":
            torch.set_float32_matmul_precision(previous)


class _GeometryEvaluation(torch.autograd.Function):
    """Recompute geometry backward under the same FP32 contract as forward."""

    @staticmethod
    def forward(ctx, module, state, bodies):
        ctx.module = module
        ctx.bodies = bodies
        ctx.save_for_backward(state)
        return module._evaluate(state, bodies)

    @staticmethod
    def backward(ctx, *output_gradients):
        state, = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        with torch.enable_grad(), _geometry_precision(state.device.type):
            outputs = ctx.module._evaluate(state, ctx.bodies, analytical_grad=not higher_order)
            gradient, = torch.autograd.grad(outputs, state, output_gradients, create_graph=higher_order)
        return None, gradient, None


class UFOPrivilegedKinematics(nn.Module):
    """Map root-link state and joint coordinates to UFO's 373D local-self input.

    FK is provided by pytorch-kinematics. Link velocities use its joint axes and
    ancestry with the geometric Jacobian, including the floating root twist.
    Neither inertial parameters nor body COM positions enter link kinematics.
    """

    observation_dim = 373
    _body_fields = ("position", "rotation", "linear_velocity", "angular_velocity")

    def __init__(self, urdf_path, joint_names, body_names, *, device="cpu",
                 dtype=torch.float32, root_height_offset=0.0):
        super().__init__()
        description = Path(urdf_path).read_bytes()
        self.urdf_sha256 = hashlib.sha256(description).hexdigest()
        self.joint_names = tuple(joint_names)
        self.body_names = tuple(body_names)
        self.root_height_offset = float(root_height_offset)
        if not math.isfinite(self.root_height_offset):
            raise ValueError("Root height offset must be finite")
        urdf = ET.fromstring(description)
        # RP1 adds actuator armature to URDF dynamics; it does not affect FK.
        for dynamics in urdf.findall("joint/dynamics"):
            dynamics.attrib.pop("armature", None)
        self.chain = pk.build_chain_from_urdf(ET.tostring(urdf))
        chain_joint_names = self.chain.get_joint_parameter_names()
        if (len(self.joint_names) != 24 or len(set(self.joint_names)) != 24
                or set(self.joint_names) != set(chain_joint_names)):
            raise ValueError("UFO kinematics requires the URDF's 24 named joints")
        if (len(self.body_names) != 25 or len(set(self.body_names)) != 25
                or set(self.body_names) != set(self.chain.frame_to_idx)
                or self.chain.frame_to_idx[self.body_names[0]] != 0):
            raise ValueError("UFO kinematics requires all 25 bodies with the root first")
        if any(joint.joint_type != "revolute" for joint in self.chain.get_joints()):
            raise ValueError("UFO kinematics currently supports revolute joints only")

        joint_order = [self.joint_names.index(name) for name in chain_joint_names]
        body_order = [self.chain.frame_to_idx[name] for name in self.body_names]
        joint_frames = [int((self.chain.joint_indices == j).nonzero()[0, 0]) for j in range(24)]
        ancestors = torch.tensor([
            [frame in self.chain.parents_indices[body].tolist() for body in body_order]
            for frame in joint_frames
        ], dtype=torch.float32)
        self.register_buffer("joint_order", torch.tensor(joint_order, dtype=torch.long))
        self.register_buffer("body_order", torch.tensor(body_order, dtype=torch.long))
        self.register_buffer("joint_frames", torch.tensor(joint_frames, dtype=torch.long))
        self.register_buffer("ancestors", ancestors)
        self.register_buffer("axes", self.chain.axes.detach().clone())
        self.to(device=device, dtype=dtype)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.chain.to(device=self.axes.device, dtype=self.axes.dtype)
        return result

    def checkpoint_contract(self):
        return {
            "version": 1,
            "library": "pytorch-kinematics==0.10.0",
            "urdf_sha256": self.urdf_sha256,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "root_height_offset": self.root_height_offset,
            "velocity_convention": "world_frame_link_origin",
        }

    def body_kinematics(self, state):
        """Return world-frame link poses and twists, with arbitrary root x=y=0."""
        return dict(zip(self._body_fields, self._run(state, bodies=True)))

    def _run(self, state, *, bodies):
        if torch.is_grad_enabled() and state.requires_grad:
            return _GeometryEvaluation.apply(self, state, bodies)
        return self._evaluate(state, bodies)

    def _evaluate(self, state, bodies, *, analytical_grad=True):
        with _geometry_precision(state.device.type):
            state = state.to(dtype=self.axes.dtype)
            if bodies:
                result = self._body_kinematics(state, analytical_grad=analytical_grad)
                return tuple(result[key] for key in self._body_fields)
            return (self._observation(state, analytical_grad=analytical_grad),)

    def _body_kinematics(self, state, *, analytical_grad=True):
        if state.ndim != 2 or state.shape[-1] != 59:
            raise ValueError("UFO physical state must have shape (batch, 59)")
        joints = state[:, 11:35].index_select(-1, self.joint_order)
        speeds = state[:, 35:59].index_select(-1, self.joint_order)
        transforms = self.chain.forward_kinematics_tensor(joints, analytical_grad=analytical_grad)
        bodies = transforms.index_select(0, self.body_order).transpose(0, 1)
        joint_transforms = transforms.index_select(0, self.joint_frames).transpose(0, 1)
        relative_positions = bodies[:, :, :3, 3]
        axes = (joint_transforms[:, :, :3, :3] @ self.axes[None, :, :, None]).squeeze(-1)
        angular_contributions = axes * speeds[:, :, None]
        relative_angular = torch.einsum("jb,njc->nbc", self.ancestors, angular_contributions)
        # Sum_j axis_j*dq_j x (link_position - joint_origin) over ancestor joints.
        origin_contributions = torch.linalg.cross(angular_contributions, joint_transforms[:, :, :3, 3])
        relative_linear = torch.linalg.cross(relative_angular, relative_positions)
        relative_linear = relative_linear - torch.einsum("jb,njc->nbc", self.ancestors, origin_contributions)

        quaternion_wxyz = torch.cat((state[:, 4:5], state[:, 1:4]), dim=-1)
        root_rotation = pk.quaternion_to_matrix(quaternion_wxyz)
        rotation = root_rotation[:, None] @ bodies[:, :, :3, :3]
        rotate = lambda value: (root_rotation[:, None] @ value[..., None]).squeeze(-1)
        displacement = rotate(relative_positions)
        root_position = torch.cat((torch.zeros_like(state[:, :2]), state[:, :1] + self.root_height_offset), -1)
        position = displacement + root_position[:, None]
        root_angular = state[:, None, 8:11]
        linear_velocity = state[:, None, 5:8] + torch.linalg.cross(root_angular.expand_as(displacement), displacement)
        linear_velocity = linear_velocity + rotate(relative_linear)
        angular_velocity = root_angular + rotate(relative_angular)
        return {
            "position": position, "rotation": rotation,
            "linear_velocity": linear_velocity, "angular_velocity": angular_velocity,
        }

    def forward(self, state):
        return self._run(state, bodies=False)[0]

    def _observation(self, state, *, analytical_grad=True):
        bodies = self._body_kinematics(state, analytical_grad=analytical_grad)
        rotation = bodies["rotation"]
        heading = torch.atan2(rotation[:, 0, 1, 0], rotation[:, 0, 0, 0])
        cosine, sine = heading.cos(), heading.sin()
        zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
        inverse_heading = torch.stack((cosine, sine, zero, -sine, cosine, zero, zero, zero, one), -1)
        inverse_heading = inverse_heading.reshape(-1, 3, 3)
        rotate = lambda value: (inverse_heading[:, None] @ value[..., None]).squeeze(-1)
        local_positions = rotate(bodies["position"] - bodies["position"][:, :1])[:, 1:]
        local_rotations = inverse_heading[:, None] @ rotation
        tangent_normal = torch.cat((local_rotations[..., 0], local_rotations[..., 2]), -1)
        return torch.cat((
            bodies["position"][:, 0, 2:3], local_positions.flatten(1), tangent_normal.flatten(1),
            rotate(bodies["linear_velocity"]).flatten(1), rotate(bodies["angular_velocity"]).flatten(1),
        ), -1)
