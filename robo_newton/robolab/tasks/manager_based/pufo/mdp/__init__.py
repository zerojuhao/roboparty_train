"""PUFO MDP terms. Visual terms stay local to this task package."""

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.events import (
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
)
from isaaclab.envs.mdp.observations import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)

from robolab.tasks.manager_based.ufo.mdp import events, terminations
from . import rewards
from .observations import depth_image, heading_context, local_self_obs, velocity_command, velocity_target_yaw

__all__ = [
    "JointPositionActionCfg",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
    "randomize_actuator_gains",
    "randomize_joint_parameters",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_material",
    "depth_image",
    "heading_context",
    "local_self_obs",
    "velocity_command",
    "velocity_target_yaw",
    "events",
    "rewards",
    "terminations",
]
