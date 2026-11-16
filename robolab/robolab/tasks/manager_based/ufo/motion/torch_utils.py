"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Trimmed to the functions still used by the UFO task after the NPZ motion
refactor: the observation math (motion_observations), the expert loader and
the runner's tracking eval. The former FK/motion-library helpers were removed
together with the vendored motion stack.
"""

import torch
from torch import Tensor


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    """Convert angle-axis to quaternion."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def my_quat_rotate(q, v):
    """Rotate vector by quaternion (assumes w_last=True format)."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def calc_heading(q):
    """Calculate heading direction from quaternion."""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat_inv(q, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Calculate inverse heading rotation from quaternion."""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_to_tan_norm(q: Tensor, w_last: bool) -> Tensor:
    """Represent rotation using tangent and normal vectors."""
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    if w_last:
        tan = my_quat_rotate(q, ref_tan)
    else:
        raise NotImplementedError

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    if w_last:
        norm = my_quat_rotate(q, ref_norm)
    else:
        raise NotImplementedError

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Multiply two quaternions."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    """Rotate vector by inverse quaternion."""
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return a - b + c


@torch.jit.script
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    q0, q1 = quat_unit(q0), quat_unit(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    cosine = dot.abs().clamp(max=1.0)
    near = cosine > 0.9995
    # Keep the unused spherical branch finite, including its backward at q0 == q1.
    angle = torch.acos(torch.where(near, torch.zeros_like(cosine), cosine))
    sine = torch.sin(angle)
    spherical = (torch.sin((1 - t) * angle) * q0 + torch.sin(t * angle) * q1) / sine
    linear = (1 - t) * q0 + t * q1
    return quat_unit(torch.where(near, linear, spherical))
