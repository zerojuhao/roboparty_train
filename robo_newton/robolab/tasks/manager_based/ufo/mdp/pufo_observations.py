"""PUFO visual and motion-context observations, independent of terrain task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from robolab.tasks.manager_based.ufo.mdp.motion_observations import compute_humanoid_observations_max


TERRAIN_PRIV_X_MIN = -0.4
TERRAIN_PRIV_X_MAX = 1.6
TERRAIN_PRIV_Y_MIN = -0.6
TERRAIN_PRIV_Y_MAX = 0.6
TERRAIN_PRIV_RESOLUTION = 0.1


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def local_self_obs(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat(
        [
            value
            for value in compute_humanoid_observations_max(
                _tensor(robot.data.body_pos_w),
                _tensor(robot.data.body_quat_w),
                _tensor(robot.data.body_link_lin_vel_w),
                _tensor(robot.data.body_link_ang_vel_w),
                local_root_obs=True,
                root_height_obs=True,
            ).values()
        ],
        dim=-1,
    )


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "distance_to_image_plane_noised_history",
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    image = _tensor(sensor.data.output[data_type])
    if image.ndim == 5 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    if image.ndim != 4:
        raise ValueError(f"PUFO depth must be [env, frames, height, width], got {tuple(image.shape)}")
    return image.clamp(0.0, 2.5).div(2.5)


def terrain_priv(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_priv"),
    clip: float = 0.5,
) -> torch.Tensor:
    """Return H200-style ground-height deltas on the robot-heading ray grid."""
    sensor = env.scene[sensor_cfg.name]
    hit_heights = _tensor(sensor.data.ray_hits_w)[..., 2]
    if hit_heights.ndim != 2 or hit_heights.shape[-1] % 2 == 0:
        raise ValueError(f"PUFO terrain_priv expects an odd flattened ray grid, got {tuple(hit_heights.shape)}")
    num_x = round((TERRAIN_PRIV_X_MAX - TERRAIN_PRIV_X_MIN) / TERRAIN_PRIV_RESOLUTION) + 1
    num_y = round((TERRAIN_PRIV_Y_MAX - TERRAIN_PRIV_Y_MIN) / TERRAIN_PRIV_RESOLUTION) + 1
    if hit_heights.shape[-1] != num_x * num_y:
        raise ValueError(
            "PUFO terrain_priv ray count does not match the H200 grid: "
            f"expected {num_x * num_y}, got {hit_heights.shape[-1]}"
        )
    # GridPatternCfg(ordering="xy") flattens x fastest; (0, 0) is x index 4, y index 6.
    reference_index = (
        round(-TERRAIN_PRIV_Y_MIN / TERRAIN_PRIV_RESOLUTION) * num_x
        + round(-TERRAIN_PRIV_X_MIN / TERRAIN_PRIV_RESOLUTION)
    )
    reference = hit_heights[:, reference_index : reference_index + 1]
    return torch.nan_to_num(reference - hit_heights, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def _motion_state(env: ManagerBasedEnv):
    return env.command_manager.get_term("motion").get_motion_state()


def _root_state(env: ManagerBasedEnv):
    robot = env.scene["robot"]
    quat = _tensor(robot.data.body_quat_w)[:, 0]
    lin_vel = _tensor(robot.data.body_link_lin_vel_w)[:, 0]
    ang_vel = _tensor(robot.data.body_link_ang_vel_w)[:, 0]
    return quat, lin_vel, ang_vel


def _heading_xy(quat: torch.Tensor) -> torch.Tensor:
    forward = quat_apply(
        quat,
        torch.tensor([1.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype).expand(quat.shape[0], -1),
    )
    return torch.nn.functional.normalize(forward[:, :2], dim=-1, eps=1.0e-6)


def heading_context(env: ManagerBasedEnv) -> torch.Tensor:
    current = _heading_xy(_root_state(env)[0])
    reference = _heading_xy(_motion_state(env)["root_rot"])
    cos = (current * reference).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin = current[:, :1] * reference[:, 1:2] - current[:, 1:2] * reference[:, :1]
    return torch.cat((1.0 - cos, sin), dim=-1)


def _reference_velocity_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    quat, _, _ = _root_state(env)
    reference = _motion_state(env)
    local_xy = quat_apply_inverse(quat, reference["root_vel"])[:, :2]
    return torch.cat((local_xy, reference["root_ang_vel"][:, 2:3]), dim=-1)


def velocity_command(env: ManagerBasedEnv) -> torch.Tensor:
    return _reference_velocity_yaw(env)


def velocity_target_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    return _reference_velocity_yaw(env).detach()


__all__ = [
    "depth_image",
    "heading_context",
    "local_self_obs",
    "terrain_priv",
    "velocity_command",
    "velocity_target_yaw",
]
