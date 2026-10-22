from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from robolab.utils.tensors import as_torch_tensor, resolve_env_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_from_ref(
    env: ManagerBasedEnv,
    env_ids,
    animation: str = "animation",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Write clip joints, orientation, and velocity. Keep reset_base xy; z from robot init_state.

    Motion clip root height is not used: clips can float or penetrate. Ground contact
    comes from ``ArticulationCfg.init_state.pos[2]``.
    """
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
    if env_ids.numel() == 0:
        return

    animation_term = env.animation_manager.get_term(animation)
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = animation_term.get_dof_pos(env_ids)[:, 0].clone()
    joint_vel = animation_term.get_dof_vel(env_ids)[:, 0]
    soft_limits = as_torch_tensor(robot.data.soft_joint_pos_limits)[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    root_states = as_torch_tensor(robot.data.default_root_state)[env_ids].clone()
    root_xy = as_torch_tensor(robot.data.root_link_state_w)[env_ids, :2]
    root_states[:, 0:2] = root_xy
    root_states[:, 2] = as_torch_tensor(env.scene.env_origins)[env_ids, 2] + float(robot.cfg.init_state.pos[2])
    root_states[:, 3:7] = animation_term.get_root_quat(env_ids)[:, 0]
    root_states[:, 7:10] = animation_term.get_root_vel_w(env_ids)[:, 0]
    root_states[:, 10:13] = animation_term.get_root_ang_vel_w(env_ids)[:, 0]
    robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_states[:, 7:13], env_ids=env_ids)
