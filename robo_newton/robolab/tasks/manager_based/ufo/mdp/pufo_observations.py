"""PUFO visual and motion-context observations, independent of terrain task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from robolab.tasks.manager_based.ufo.mdp.motion_observations import compute_humanoid_observations_max
from robolab.utils.tensors import as_torch_tensor


TERRAIN_PRIV_X_MIN = -0.4
TERRAIN_PRIV_X_MAX = 1.6
TERRAIN_PRIV_Y_MIN = -0.6
TERRAIN_PRIV_Y_MAX = 0.6
TERRAIN_PRIV_RESOLUTION = 0.1


def local_self_obs(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    body_pos = as_torch_tensor(robot.data.body_pos_w)
    obs = compute_humanoid_observations_max(
        body_pos,
        as_torch_tensor(robot.data.body_quat_w),
        as_torch_tensor(robot.data.body_link_lin_vel_w),
        as_torch_tensor(robot.data.body_link_ang_vel_w),
        local_root_obs=True,
        root_height_obs=True,
    )
    _, ground = _terrain_heights(env, SceneEntityCfg("terrain_priv"))
    obs["root_height"] = body_pos[:, 0, 2:3] - ground
    return torch.cat(list(obs.values()), dim=-1)


class depth_image(ManagerTermBase):
    """Sample normalized camera history from oldest to newest with per-episode delay."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        self.sensor = env.scene.sensors[sensor_cfg.name]
        self.data_type = cfg.params["data_type"]
        self.delayed_frame_ranges = cfg.params["delayed_frame_ranges"]
        self.integer_delay = cfg.params.get("integer_delay", False)
        history_skip_frames = cfg.params["history_skip_frames"]
        num_output_frames = cfg.params["num_output_frames"]
        images = as_torch_tensor(self.sensor.data.output[self.data_type])
        if images.ndim != 5 or images.shape[-1] != 1:
            raise ValueError(f"PUFO depth history must be [env, frames, height, width, 1], got {images.shape}")
        self.history_length = images.shape[1]
        if history_skip_frames < 1 or num_output_frames < 1:
            raise ValueError("Depth history stride and output frame count must be positive")
        min_delay, max_delay = self.delayed_frame_ranges
        frames_needed = (num_output_frames - 1) * history_skip_frames + 1 + max_delay
        if min_delay < 0 or max_delay < min_delay or frames_needed > self.history_length:
            raise ValueError("Depth history is too short or the delay range is invalid")
        self.frame_offsets = torch.arange(
            num_output_frames - 1, -1, -1, device=env.device,
        ) * history_skip_frames
        self.delayed_frames = torch.zeros(env.num_envs, device=env.device)
        self.batch_indices = torch.arange(env.num_envs, device=env.device).unsqueeze(1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self.integer_delay:
            lo, hi = self.delayed_frame_ranges
            self.delayed_frames[env_ids] = torch.randint(
                lo, hi + 1, self.delayed_frames[env_ids].shape, device=self.delayed_frames.device
            ).float()
        else:
            self.delayed_frames[env_ids] = torch.empty_like(self.delayed_frames[env_ids]).uniform_(
                *self.delayed_frame_ranges
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        data_type: str,
        history_skip_frames: int,
        num_output_frames: int,
        delayed_frame_ranges: tuple[int, int],
        integer_delay: bool = False,
    ) -> torch.Tensor:
        images = as_torch_tensor(self.sensor.data.output[self.data_type]).squeeze(-1)
        # Match the reference sampler: subtract continuous delay, then floor frame indices.
        frame_indices = (
            self.history_length - 1 - self.frame_offsets.unsqueeze(0) - self.delayed_frames.unsqueeze(1)
        ).long()
        # The camera pipeline already normalizes depth to [0, 1].
        return images[self.batch_indices, frame_indices]


def _terrain_heights(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    """Read the root-centered grid and its local (0, 0) ground reference."""
    sensor = env.scene[sensor_cfg.name]
    hit_heights = as_torch_tensor(sensor.data.ray_hits_w)[..., 2]
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
    if not torch.isfinite(reference).all():
        # IsaacLab probes dimensions before assigning observation_manager or
        # resetting clones from the staging grid onto the finite terrain.
        # Only those discarded constructor observations may use a placeholder.
        if isinstance(env, ManagerBasedEnv) and not hasattr(env, "observation_manager"):
            reference = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
        else:
            ids = (~torch.isfinite(reference[:, 0])).nonzero().flatten()
            root = as_torch_tensor(env.scene["robot"].data.body_pos_w)[ids, 0]
            details = {"env_ids": ids.cpu(), "root_position": root.detach().cpu(),
                       "reference_hits": hit_heights[ids].detach().cpu()}
            for field in ("_ray_starts_w", "_ray_directions_w"):
                if hasattr(sensor, field):
                    import warp as wp

                    details[field] = wp.to_torch(getattr(sensor, field))[ids, reference_index].detach().cpu()
            env._pufo_ground_failure = details
            raise RuntimeError(
                "PUFO root-ground ray missed terrain; "
                f"env_ids={ids[:16].tolist()} root_positions={root[:16].tolist()} "
                f"finite_roots={bool(torch.isfinite(root).all())}"
            )
    return hit_heights, reference


def terrain_priv(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("terrain_priv"),
    clip: float = 0.5,
) -> torch.Tensor:
    """Return terrain_z(sample) - terrain_z(root), positive for raised terrain."""
    hit_heights, reference = _terrain_heights(env, sensor_cfg)
    # A peripheral miss is unknown/deep terrain, never a raised obstacle.
    relative = torch.where(torch.isfinite(hit_heights), hit_heights - reference, -clip)
    return relative.clamp(-clip, clip)


def _motion_state(env: ManagerBasedEnv):
    return env.command_manager.get_term("motion").get_motion_state()


def _root_state(env: ManagerBasedEnv):
    robot = env.scene["robot"]
    quat = as_torch_tensor(robot.data.body_quat_w)[:, 0]
    lin_vel = as_torch_tensor(robot.data.body_link_lin_vel_w)[:, 0]
    ang_vel = as_torch_tensor(robot.data.body_link_ang_vel_w)[:, 0]
    return quat, lin_vel, ang_vel


def heading_xy(quat: torch.Tensor) -> torch.Tensor:
    forward = quat_apply(
        quat,
        torch.tensor([1.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype).expand(
            quat.shape[0], -1
        ),
    )
    return torch.nn.functional.normalize(forward[:, :2], dim=-1, eps=1.0e-6)


def heading_observation(
    current_heading_xy: torch.Tensor,
    target_heading_xy: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Encode a heading target; invalid or degenerate targets become ``[0, 0]``."""
    current_norm = torch.linalg.vector_norm(current_heading_xy, dim=-1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target_heading_xy, dim=-1, keepdim=True)
    finite = torch.isfinite(current_heading_xy).all(dim=-1, keepdim=True) & torch.isfinite(
        target_heading_xy
    ).all(dim=-1, keepdim=True)
    nondegenerate = (current_norm > 1.0e-6) & (target_norm > 1.0e-6)
    current = current_heading_xy / current_norm.clamp_min(1.0e-6)
    target = target_heading_xy / target_norm.clamp_min(1.0e-6)
    valid = valid.reshape(-1, 1).to(device=current.device, dtype=torch.bool)
    valid = valid & finite & nondegenerate
    cos = (current * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin = current[:, :1] * target[:, 1:2] - current[:, 1:2] * target[:, :1]
    encoded = torch.cat((1.0 - cos, sin), dim=-1)
    return torch.where(valid, encoded, torch.zeros_like(encoded))


def world_to_yaw(vector_xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Rotate horizontal world vectors into a yaw-only robot frame."""
    heading = torch.nn.functional.normalize(heading, dim=-1, eps=1.0e-6)
    x, y = vector_xy.unbind(-1)
    cos, sin = heading.unbind(-1)
    return torch.stack((cos * x + sin * y, -sin * x + cos * y), dim=-1)


def yaw_to_world(vector_xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Rotate yaw-frame horizontal vectors into the world frame."""
    heading = torch.nn.functional.normalize(heading, dim=-1, eps=1.0e-6)
    x, y = vector_xy.unbind(-1)
    cos, sin = heading.unbind(-1)
    return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1)


def align_world_vector(
    vector_xy: torch.Tensor,
    reference_heading_xy: torch.Tensor,
    target_heading_xy: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed yaw alignment between two trajectory anchors."""
    return yaw_to_world(
        world_to_yaw(vector_xy, reference_heading_xy), target_heading_xy
    )


def velocity_observation(
    target_world_xy: torch.Tensor,
    current_heading_xy: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Return the PUFO command contract ``[target_vx, target_vy, valid]``."""
    local = world_to_yaw(target_world_xy, current_heading_xy)
    valid = valid.reshape(-1, 1).to(device=local.device, dtype=torch.bool)
    return torch.cat(
        (torch.where(valid, local, torch.zeros_like(local)), valid.to(local.dtype)),
        dim=-1,
    )


def heading_context(env: ManagerBasedEnv) -> torch.Tensor:
    current = heading_xy(_root_state(env)[0])
    reference = heading_xy(_motion_state(env)["root_rot"])
    valid = torch.ones(current.shape[0], 1, device=current.device)
    return heading_observation(current, reference, valid)


def velocity_command(env: ManagerBasedEnv) -> torch.Tensor:
    """Default manager observation; the runner replaces it with the z-paired command."""
    quat, _, _ = _root_state(env)
    reference = _motion_state(env)
    current_heading = heading_xy(quat)
    valid = torch.ones(
        current_heading.shape[0], 1, dtype=torch.bool, device=current_heading.device
    )
    return velocity_observation(reference["root_vel"][:, :2], current_heading, valid)


def velocity_target_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    """Training-only current root velocity in the yaw frame, in physical units."""
    quat, linear_velocity, _ = _root_state(env)
    local_xy = world_to_yaw(linear_velocity[:, :2], heading_xy(quat))
    return torch.cat((local_xy, linear_velocity[:, 2:3]), dim=-1).detach()


def transition_kinematics(env: ManagerBasedEnv) -> torch.Tensor:
    """World heading and horizontal root velocity for pre-reset outcomes."""
    quat, linear_velocity, _ = _root_state(env)
    return torch.cat((heading_xy(quat), linear_velocity[:, :2]), dim=-1)


__all__ = [
    "depth_image",
    "align_world_vector",
    "heading_context",
    "heading_observation",
    "heading_xy",
    "local_self_obs",
    "terrain_priv",
    "transition_kinematics",
    "velocity_command",
    "velocity_observation",
    "velocity_target_yaw",
    "world_to_yaw",
    "yaw_to_world",
]
