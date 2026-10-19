"""Offline URDF forward kinematics for motion tracks (numpy only).

Computes world-frame body poses from root pose + joint angles using the same
URDF declaration order as ``motion_npz.urdf_names``. Used at load/train time so
Desktop ``joint_q`` clips do not need on-disk ``body_*`` tracks.

This is real kinematic FK from the URDF tree — not fabricated tracks.
Velocities are finite-differenced from the resulting poses (and joint angles).
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY (roll X, pitch Y, yaw Z) → 3x3 rotation."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues: axis (3,) + angles (N,) → rotations (N,3,3)."""
    axis = np.asarray(axis, dtype=np.float64)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return np.broadcast_to(np.eye(3), (angles.shape[0], 3, 3)).copy()
    x, y, z = axis / nrm
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1.0 - c
    R = np.empty((angles.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s
    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s
    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C
    return R


def _quat_xyzw_to_matrix_batch(q: np.ndarray) -> np.ndarray:
    """(N,4) XYZW → (N,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(n, 1e-12, None)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _matrix_to_quat_xyzw_batch(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) XYZW unit quaternions."""
    m = np.asarray(R, dtype=np.float64)
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    for i in range(n):
        mi = m[i]
        ti = float(t[i])
        if ti > 0.0:
            s = 0.5 / np.sqrt(ti + 1.0)
            w = 0.25 / s
            x = (mi[2, 1] - mi[1, 2]) * s
            y = (mi[0, 2] - mi[2, 0]) * s
            z = (mi[1, 0] - mi[0, 1]) * s
        elif mi[0, 0] > mi[1, 1] and mi[0, 0] > mi[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mi[0, 0] - mi[1, 1] - mi[2, 2])
            w = (mi[2, 1] - mi[1, 2]) / s
            x = 0.25 * s
            y = (mi[0, 1] + mi[1, 0]) / s
            z = (mi[0, 2] + mi[2, 0]) / s
        elif mi[1, 1] > mi[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mi[1, 1] - mi[0, 0] - mi[2, 2])
            w = (mi[0, 2] - mi[2, 0]) / s
            x = (mi[0, 1] + mi[1, 0]) / s
            y = 0.25 * s
            z = (mi[1, 2] + mi[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + mi[2, 2] - mi[0, 0] - mi[1, 1])
            w = (mi[1, 0] - mi[0, 1]) / s
            x = (mi[0, 2] + mi[2, 0]) / s
            y = (mi[1, 2] + mi[2, 1]) / s
            z = 0.25 * s
        q[i] = (x, y, z, w)
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    return q.astype(np.float32)


class UrdfKinematicTree:
    """Minimal URDF kinematic tree for revolute / continuous / fixed joints."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.joint_names: list[str] = []
        self.body_names: list[str] = [b.attrib["name"] for b in root.findall("link")]
        if not self.body_names or self.body_names[0] != "base_link":
            raise ValueError(
                f"URDF {self.urdf_path} must declare base_link first; got {self.body_names[:3]!r}"
            )
        self._body_index = {n: i for i, n in enumerate(self.body_names)}

        # parent, child, xyz, R_origin, axis_or_None, joint_col_or_-1
        self._edges: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None, int]] = []
        joint_col = 0
        for joint in root.findall("joint"):
            jtype = joint.attrib.get("type", "fixed")
            name = joint.attrib["name"]
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
            rpy = _parse_xyz(origin.attrib.get("rpy") if origin is not None else None)
            R0 = _rpy_to_matrix(rpy)
            axis_el = joint.find("axis")
            axis = _parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else "1 0 0")
            if parent not in self._body_index or child not in self._body_index:
                raise ValueError(f"Joint {name} references unknown link ({parent}->{child})")
            if jtype == "fixed":
                self._edges.append(
                    (self._body_index[parent], self._body_index[child], xyz, R0, None, -1)
                )
            elif jtype in ("revolute", "continuous"):
                self.joint_names.append(name)
                self._edges.append(
                    (self._body_index[parent], self._body_index[child], xyz, R0, axis, joint_col)
                )
                joint_col += 1
            else:
                raise ValueError(f"Unsupported joint type {jtype!r} for {name} in {self.urdf_path}")

        seen = {0}
        for p, c, *_ in self._edges:
            if p not in seen:
                raise ValueError(
                    f"URDF edge parent {self.body_names[p]} appears before its pose is known"
                )
            seen.add(c)

    def forward_kinematics(
        self,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``body_pos_w`` (N,B,3) and ``body_quat_w`` (N,B,4) XYZW."""
        root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
        root_quat_w = np.asarray(root_quat_w, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
            raise ValueError(f"root_pos_w must be (N,3), got {root_pos_w.shape}")
        if root_quat_w.shape != (root_pos_w.shape[0], 4):
            raise ValueError(f"root_quat_w must be (N,4), got {root_quat_w.shape}")
        n = root_pos_w.shape[0]
        if joint_pos.shape != (n, len(self.joint_names)):
            raise ValueError(
                f"joint_pos must be (N,{len(self.joint_names)}), got {joint_pos.shape}"
            )

        n_bodies = len(self.body_names)
        pos = np.zeros((n, n_bodies, 3), dtype=np.float64)
        rot = np.zeros((n, n_bodies, 3, 3), dtype=np.float64)
        pos[:, 0] = root_pos_w
        rot[:, 0] = _quat_xyzw_to_matrix_batch(root_quat_w)

        for parent, child, xyz, R0, axis, jcol in self._edges:
            Rp = rot[:, parent]  # (N,3,3)
            pp = pos[:, parent]  # (N,3)
            if axis is None:
                Rj = np.broadcast_to(R0, (n, 3, 3))
            else:
                Rj = np.einsum("ij,njk->nik", R0, _axis_angle_matrix_batch(axis, joint_pos[:, jcol]))
            rot[:, child] = np.einsum("nij,njk->nik", Rp, Rj)
            pos[:, child] = pp + np.einsum("nij,j->ni", Rp, xyz)

        body_pos = pos.astype(np.float32)
        # Flatten bodies for batch quat convert then reshape.
        body_quat = _matrix_to_quat_xyzw_batch(rot.reshape(-1, 3, 3)).reshape(n, n_bodies, 4)
        for i in range(1, n):
            dots = np.sum(body_quat[i] * body_quat[i - 1], axis=-1, keepdims=True)
            body_quat[i] = np.where(dots < 0.0, -body_quat[i], body_quat[i])
        return body_pos, body_quat


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape[0] < 2:
        raise ValueError("Need >= 2 frames to finite-difference velocities")
    v = np.empty_like(x, dtype=np.float32)
    v[:-1] = ((x[1:] - x[:-1]) / dt).astype(np.float32)
    v[-1] = v[-2]
    return v


def _ang_vel_world_from_quat(quat: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from consecutive XYZW quaternions (N,B,4)."""
    q = np.asarray(quat, dtype=np.float64)
    n, b, _ = q.shape
    out = np.zeros((n, b, 3), dtype=np.float32)
    for i in range(n - 1):
        q0 = q[i]
        q1 = q[i + 1]
        # p = conj(q0) = (-x,-y,-z,w); r = p * q1
        px, py, pz, pw = -q0[:, 0], -q0[:, 1], -q0[:, 2], q0[:, 3]
        qx, qy, qz, qw = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        dx = pw * qx + px * qw + py * qz - pz * qy
        dy = pw * qy - px * qz + py * qw + pz * qx
        dz = pw * qz + px * qy - py * qx + pz * qw
        dw = pw * qw - px * qx - py * qy - pz * qz
        # Prefer the shorter arc.
        neg = dw < 0.0
        dx = np.where(neg, -dx, dx)
        dy = np.where(neg, -dy, dy)
        dz = np.where(neg, -dz, dz)
        dw = np.where(neg, -dw, dw)
        vnorm = np.sqrt(dx * dx + dy * dy + dz * dz)
        angle = 2.0 * np.arctan2(vnorm, dw)
        axis = np.zeros((b, 3), dtype=np.float64)
        mask = vnorm > 1e-12
        axis[mask, 0] = dx[mask] / vnorm[mask]
        axis[mask, 1] = dy[mask] / vnorm[mask]
        axis[mask, 2] = dz[mask] / vnorm[mask]
        body_aa = axis * angle[:, None]
        R0 = _quat_xyzw_to_matrix_batch(q0)
        out[i] = (np.einsum("bij,bj->bi", R0, body_aa) / dt).astype(np.float32)
    out[-1] = out[-2]
    return out


def compute_full_body_tracks(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    joint_pos: np.ndarray,
    *,
    urdf_path: str | Path,
    fps: float,
    joint_names: list[str] | None = None,
) -> dict:
    """FK full-body + finite-diff velocities for tracking loaders.

    Returns body_names, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
    joint_vel. When ``joint_names`` is given and differs from the URDF order,
    columns are reordered before FK.
    """
    tree = UrdfKinematicTree(urdf_path)
    jp = np.asarray(joint_pos, dtype=np.float32)
    if joint_names is not None:
        names = [str(n) for n in joint_names]
        if names != tree.joint_names:
            lookup = {n: i for i, n in enumerate(names)}
            missing = [n for n in tree.joint_names if n not in lookup]
            if missing:
                raise ValueError(f"Motion missing URDF joints for FK: {missing}")
            jp = jp[:, [lookup[n] for n in tree.joint_names]]
    elif jp.shape[1] != len(tree.joint_names):
        raise ValueError(
            f"joint_pos width {jp.shape[1]} != URDF DoF {len(tree.joint_names)}; pass joint_names"
        )

    body_pos_w, body_quat_w = tree.forward_kinematics(root_pos_w, root_quat_w, jp)
    dt = 1.0 / float(fps)
    return {
        "body_names": np.asarray(tree.body_names, dtype=str),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": _finite_diff(body_pos_w, dt),
        "body_ang_vel_w": _ang_vel_world_from_quat(body_quat_w, dt),
        "joint_vel": _finite_diff(jp, dt),
    }
