"""Shared auxiliary rewards used by UFO and PUFO."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg

from robolab.utils.tensors import as_torch_tensor

# Thresholds shared by the foot-contact regularizers.
UNDESIRED_CONTACT_FORCE_THRESHOLD = 1.0
UNDESIRED_CONTACT_HISTORY_WINDOW = 3


def penalty_action_rate(env: ManagerBasedEnv) -> torch.Tensor:
    actions = env.action_manager.action
    previous_actions = env.action_manager.prev_action
    return torch.sum(torch.square(previous_actions - actions), dim=1)


def joint_vel_l2(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint velocity for the selected articulation joints."""
    robot = env.scene[asset_cfg.name]
    joint_vel = as_torch_tensor(robot.data.joint_vel)[..., asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=-1)


def joint_acc_l2(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint acceleration for the selected articulation joints."""
    robot = env.scene[asset_cfg.name]
    joint_acc = as_torch_tensor(robot.data.joint_acc)[..., asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_acc), dim=-1)


def sound_suppression_acc_per_foot(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Match Parkour's contact-gated vertical foot velocity/acceleration penalty."""
    data = env.scene["robot"].data
    # Both aliases refer to the body center of mass, as in Parkour.
    foot_vel = as_torch_tensor(data.body_vel_w)[:, sensor_cfg.body_ids, 2]
    foot_acc = as_torch_tensor(data.body_acc_w)[:, sensor_cfg.body_ids, 2]
    contact_force = as_torch_tensor(env.scene[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids, 2]
    in_contact = torch.abs(contact_force) > 0.0
    return ((torch.square(foot_vel) + torch.square(foot_acc)) * in_contact).sum(dim=1)


def limits_dof_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    dof_pos = as_torch_tensor(robot.data.joint_pos)[..., asset_cfg.joint_ids]
    limits = as_torch_tensor(robot.data.soft_joint_pos_limits)[..., asset_cfg.joint_ids, :]
    lower, upper = limits[..., 0], limits[..., 1]
    below = -(dof_pos - lower).clip(max=0.0)
    above = (dof_pos - upper).clip(min=0.0)
    return torch.sum(below + above, dim=1)


def penalty_slippage(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    contact_forces = as_torch_tensor(env.scene[sensor_cfg.name].data.net_normal_forces_w)
    foot_velocity = as_torch_tensor(robot.data.body_link_lin_vel_w)[:, asset_cfg.body_ids]
    foot_force = contact_forces[:, sensor_cfg.body_ids]
    force_norm = torch.linalg.vector_norm(foot_force, dim=-1)
    force_direction = foot_force / force_norm.unsqueeze(-1).clamp_min(1.0e-6)
    normal_velocity = (foot_velocity * force_direction).sum(dim=-1, keepdim=True)
    tangent_velocity = foot_velocity - normal_velocity * force_direction
    loaded_contact = force_norm > UNDESIRED_CONTACT_FORCE_THRESHOLD
    return (torch.linalg.vector_norm(tangent_velocity, dim=-1) * loaded_contact).sum(dim=1)


def penalty_ankle_roll(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*ankle_roll.*"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = as_torch_tensor(robot.data.joint_pos)
    if isinstance(asset_cfg.joint_ids, slice):
        ankle_roll_ids = torch.arange(joint_pos.shape[1], device=joint_pos.device)
    else:
        ankle_roll_ids = torch.as_tensor(asset_cfg.joint_ids, device=joint_pos.device)
    if ankle_roll_ids.numel() == 0:
        return torch.zeros(joint_pos.shape[0], device=joint_pos.device, dtype=joint_pos.dtype)
    return torch.sum(torch.square(joint_pos[:, ankle_roll_ids]), dim=1)


def feet_stumble(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
) -> torch.Tensor:
    contact_history = as_torch_tensor(env.scene[sensor_cfg.name].data.net_normal_forces_w_history)
    if contact_history.ndim != 4 or contact_history.shape[1] < UNDESIRED_CONTACT_HISTORY_WINDOW:
        raise ValueError("UFO requires normal-force history shaped [env, history >= 3, body, xyz].")
    foot_forces = contact_history[:, :, sensor_cfg.body_ids, :]
    force_xy = torch.linalg.vector_norm(foot_forces[..., :2], dim=-1)
    force_z = torch.abs(foot_forces[..., 2])
    excess = torch.relu(force_xy - force_z - 20.0)
    return torch.clamp(excess / 100.0, max=1.0).amax(dim=(1, 2))


def pufo_penalty_slippage(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
    force_threshold_ratio: float = 0.05,
) -> torch.Tensor:
    """Attachment PUFO: tangential speed under a body-weight-relative load gate."""
    robot = env.scene[asset_cfg.name]
    forces = as_torch_tensor(env.scene[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids]
    velocity = as_torch_tensor(robot.data.body_link_lin_vel_w)[:, asset_cfg.body_ids]
    mass = as_torch_tensor(robot.data.body_mass).sum(dim=-1, keepdim=True)
    gravity = forces.new_tensor(env.cfg.sim.gravity).norm()
    weight = (mass * gravity).clamp_min(1.0e-6)
    force_norm = torch.linalg.vector_norm(forces, dim=-1)
    direction = forces / (force_norm.unsqueeze(-1) + 1.0e-6)
    tangent_velocity = velocity - (velocity * direction).sum(dim=-1, keepdim=True) * direction
    loaded = force_norm / weight > force_threshold_ratio
    return (torch.linalg.vector_norm(tangent_velocity, dim=-1) * loaded).sum(dim=-1)


def pufo_feet_stumble(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Shared all-contact metric; support cannot hide a separate side impact."""
    from robolab.utils.contact_rewards import feet_stumble
    return feet_stumble(env, sensor_cfg)


class BodyImpactJointPositionAction(JointPositionAction):
    """Snapshot pre-action world body velocities [m/s] for the impact reward."""

    pre_body_lin_vel_w: torch.Tensor | None = None

    def process_actions(self, actions: torch.Tensor):
        self.pre_body_lin_vel_w = as_torch_tensor(self._asset.data.body_link_lin_vel_w).detach().clone()
        super().process_actions(actions)


class JointLimitedBodyImpactPositionAction(BodyImpactJointPositionAction):
    """Clip absolute PD targets while preserving legacy action/history semantics."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        # Physical joint ranges, not the narrower soft-limit reward ranges.
        limits = as_torch_tensor(self._asset.data.joint_pos_limits)[:, self._joint_ids]
        self._processed_actions = torch.clamp(self._processed_actions, min=limits[..., 0], max=limits[..., 1])


def pufo_penalty_body_impact(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    force_threshold_ratio: float = 0.05,
    smooth_l1_beta: float = 0.25,
    impact_velocity_scale: float = 1.0,
    impact_velocity_cap: float = 2.0,
    severity_cap: float = 4.0,
) -> torch.Tensor:
    """P3 impact severity without static-load penalty; velocity scale is in [m/s]."""
    pre_velocity = env.action_manager.get_term(action_name).pre_body_lin_vel_w
    if pre_velocity is None:
        raise RuntimeError("PUFO body impact requires a pre-action body velocity snapshot")
    history = as_torch_tensor(env.scene[sensor_cfg.name].data.net_forces_w_history)
    if history.ndim != 4 or history.shape[1] != env.cfg.decimation or history.shape[-1] != 3:
        raise ValueError("Body impact force history must cover one complete policy step")
    forces = history[:, :, sensor_cfg.body_ids]
    peak_index = torch.linalg.vector_norm(forces, dim=-1).argmax(dim=1)
    peak_force = forces.gather(1, peak_index[:, None, :, None].expand(-1, 1, -1, 3)).squeeze(1)
    pre_velocity = pre_velocity[:, asset_cfg.body_ids]
    if peak_force.shape != pre_velocity.shape:
        raise ValueError("Body impact sensor and asset must select matching bodies")
    mass = as_torch_tensor(env.scene[asset_cfg.name].data.body_mass).sum(dim=-1).reshape(-1, 1)
    weight = (mass * peak_force.new_tensor(env.cfg.sim.gravity).norm()).clamp_min(1.0e-6)
    force_norm = torch.linalg.vector_norm(peak_force, dim=-1)
    # Newton reports force ON the body; P3 uses the opposite, toward the object.
    incoming_direction = -peak_force / (force_norm.unsqueeze(-1) + 1.0e-6)
    incoming_speed = torch.relu((pre_velocity * incoming_direction).sum(dim=-1))
    force_ratio = torch.relu(force_norm / weight - force_threshold_ratio)
    force_severity = F.smooth_l1_loss(
        force_ratio, torch.zeros_like(force_ratio), beta=smooth_l1_beta, reduction="none",
    )
    impact_gain = (incoming_speed / impact_velocity_scale).clamp(min=0., max=impact_velocity_cap).square()
    return (force_severity * impact_gain).sum(dim=-1).clamp(max=severity_cap)
