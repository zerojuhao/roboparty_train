"""Newton-backed full-body motion FK for BeyondMimic / UFO (and reusable by AMP).

Training path: replay root + joint_pos into the live Newton articulation, read
``body_pos_w`` / ``body_quat_w``, then finite-diff velocities. Matches policy
body-frame conventions (same model as observations).

Offline / pytest without a Newton scene falls back to URDF FK via
``motion_npz.ensure_full_body_tracks``.
"""

from __future__ import annotations

import numpy as np
import torch

from robolab.utils.tensors import as_torch_tensor
from robolab.utils import motion_fk as _motion_fk


def newton_backend_available(robot) -> bool:
    """True when ``robot.data`` is the Isaac Lab Newton articulation backend."""
    return getattr(getattr(robot, "data", None), "__backend_name__", None) == "newton"


def compute_full_body_tracks_with_newton(
    robot,
    env,
    *,
    root_pos_w,
    root_quat_w,
    joint_pos,
    fps: float,
    device=None,
    motion_file: str = "<motion>",
) -> dict:
    """Compute ``body_*`` + ``joint_vel`` by replaying a clip on the live Newton model.

    Temporarily writes poses into the first batch of environments, reads link
    poses (world, origin-subtracted), restores the scene, then finite-diffs
    velocities with the same helpers as the offline URDF path.
    """
    if not newton_backend_available(robot):
        raise RuntimeError(
            f"{motion_file}: Newton FK requires the Newton backend; "
            f"got {getattr(robot.data, '__backend_name__', None)!r}."
        )

    body_names = list(robot.body_names)
    if not body_names or body_names[0] != "base_link":
        raise RuntimeError(
            f"{motion_file}: Newton FK requires body index 0 to be base_link, got {body_names[:3]}"
        )

    if device is None:
        device = root_pos_w.device if isinstance(root_pos_w, torch.Tensor) else torch.device("cpu")

    root_pos = torch.as_tensor(root_pos_w, device=device, dtype=torch.float32)
    root_quat = torch.as_tensor(root_quat_w, device=device, dtype=torch.float32)
    joints = torch.as_tensor(joint_pos, device=device, dtype=torch.float32)
    # Remap motion joints to articulation order when names differ.
    motion_joint_width = int(joints.shape[1])
    art_joint_names = list(robot.joint_names)
    if motion_joint_width != len(art_joint_names):
        raise ValueError(
            f"{motion_file}: joint_pos width {motion_joint_width} != "
            f"articulation DoF {len(art_joint_names)}"
        )

    num_frames = int(root_pos.shape[0])
    num_envs = int(env.num_envs)
    if num_envs < 1:
        raise RuntimeError("Cannot compute motion FK without at least one environment")

    env_origins = as_torch_tensor(env.scene.env_origins).to(device=device, dtype=torch.float32)[:, :3]
    if env_origins.shape[0] < min(num_envs, num_frames):
        raise RuntimeError("Scene environment origins do not cover the FK batch")

    saved_root_state = as_torch_tensor(robot.data.root_link_state_w).clone()
    saved_joint_pos = as_torch_tensor(robot.data.joint_pos).clone()
    saved_joint_vel = as_torch_tensor(robot.data.joint_vel).clone()

    body_pos_w = torch.empty((num_frames, len(body_names), 3), device=device, dtype=torch.float32)
    body_quat_w = torch.empty((num_frames, len(body_names), 4), device=device, dtype=torch.float32)

    try:
        with torch.no_grad():
            batch_size = min(num_envs, num_frames)
            for start in range(0, num_frames, batch_size):
                end = min(start + batch_size, num_frames)
                env_ids = torch.arange(end - start, device=device, dtype=torch.long)
                origins = env_origins[env_ids]
                root_pose = torch.cat((root_pos[start:end] + origins, root_quat[start:end]), dim=-1)
                robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
                robot.write_joint_position_to_sim_index(position=joints[start:end], env_ids=env_ids)
                # Newton refreshes FK on this read after the manual pose writes.
                pos = as_torch_tensor(robot.data.body_pos_w)
                quat = as_torch_tensor(robot.data.body_quat_w)
                body_pos_w[start:end] = pos[env_ids] - origins.unsqueeze(1)
                body_quat_w[start:end] = quat[env_ids]
    finally:
        with torch.no_grad():
            restore_env_ids = torch.arange(saved_root_state.shape[0], device=device, dtype=torch.long)
            robot.write_root_link_pose_to_sim_index(
                root_pose=saved_root_state[:, :7], env_ids=restore_env_ids
            )
            robot.write_root_link_velocity_to_sim_index(
                root_velocity=saved_root_state[:, 7:], env_ids=restore_env_ids
            )
            robot.write_joint_state_to_sim_index(
                position=saved_joint_pos,
                velocity=saved_joint_vel,
                env_ids=restore_env_ids,
            )
            _ = as_torch_tensor(robot.data.body_pos_w)

    # Finite-diff velocities (numpy helpers shared with URDF offline path).
    dt = 1.0 / float(fps)
    pos_np = body_pos_w.detach().cpu().numpy().astype(np.float32)
    quat_np = body_quat_w.detach().cpu().numpy().astype(np.float32)
    # Normalize quaternions before ang-vel.
    qn = np.linalg.norm(quat_np, axis=-1, keepdims=True)
    quat_np = quat_np / np.clip(qn, 1e-8, None)
    jp_np = joints.detach().cpu().numpy().astype(np.float32)

    return {
        "body_names": np.asarray(body_names, dtype=str),
        "body_pos_w": pos_np,
        "body_quat_w": quat_np,
        "body_lin_vel_w": _motion_fk._finite_diff(pos_np, dt),
        "body_ang_vel_w": _motion_fk._ang_vel_world_from_quat(quat_np, dt),
        "joint_vel": _motion_fk._finite_diff(jp_np, dt),
        "body_tracks_source": np.asarray("newton_fk"),
    }


def fill_full_body_tracks(data: dict, *, robot=None, env=None, motion_file: str = "<motion>", device=None) -> dict:
    """Fill missing ``body_*`` / ``joint_vel``: Newton when scene available, else URDF.

    Does not invent tracks without FK. No-op when full-body fields already present.
    """
    from robolab.utils.motion_npz import FULL_BODY_FIELDS, ensure_full_body_tracks

    needed = (*FULL_BODY_FIELDS, "body_names", "joint_vel")
    if all(k in data for k in needed):
        return data
    if robot is not None and env is not None and newton_backend_available(robot):
        tracks = compute_full_body_tracks_with_newton(
            robot,
            env,
            root_pos_w=data["root_pos_w"],
            root_quat_w=data["root_quat_w"],
            joint_pos=data["joint_pos"],
            fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
            device=device,
            motion_file=motion_file,
        )
        for key, value in tracks.items():
            if key not in data:
                data[key] = value
        return data
    return ensure_full_body_tracks(data)
