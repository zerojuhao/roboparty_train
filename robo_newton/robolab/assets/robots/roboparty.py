# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR
from robolab.assets.robots.roboparty_actuators import RoboPartyActuatorCfg
from robolab.utils.motion_npz import urdf_names

RPO_JOINT_NAMES, RPO_BODY_NAMES = urdf_names(f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf")
RP1_JOINT_NAMES, RP1_BODY_NAMES = urdf_names(f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/urdf/rp1_24dof.urdf")

RPO_CFG = ArticulationCfg(
    joint_ordering=RPO_JOINT_NAMES,
    body_ordering=RPO_BODY_NAMES,
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


RP1_24DOF_CFG = ArticulationCfg(
    joint_ordering=RP1_JOINT_NAMES,
    body_ordering=RP1_BODY_NAMES,
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/urdf/rp1_24dof.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": -0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "left_elbow_joint": -1.2,
            "right_hip_pitch_joint": 0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_shoulder_pitch_joint": -0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_elbow_joint": 1.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "all_joints": RoboPartyActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim={
                ".*(waist|hip|knee).*": 145.53,
                ".*ankle.*": 56.0,
                ".*(shoulder|elbow|wrist).*": 28.0,
            },
            velocity_limit_sim={
                ".*(waist|hip|knee).*": 13.61357,
                ".*(ankle|shoulder|elbow|wrist).*": 20.94395,
            },
            keep_velocity={
                ".*(waist|hip|knee).*": 8.37758,
                ".*(ankle|shoulder|elbow|wrist).*": 15.70796,
            },
            rated_effort={
                ".*(waist|hip|knee).*": 44.0,
                ".*ankle.*": 19.0,
                ".*(shoulder|elbow|wrist).*": 9.5,
            },
            rated_velocity={
                ".*(waist|hip|knee).*": 8.37758,
                ".*(ankle|shoulder|elbow|wrist).*": 6.28319,
            },
            stiffness={
                ".*waist_roll.*": 300.0,
                ".*waist_yaw.*": 250.0,
                ".*_hip_pitch.*": 120.0,
                ".*_hip_roll.*": 120.0,
                ".*_hip_yaw.*": 90.0,
                ".*_knee_.*": 150.0,
                ".*_ankle_(pitch|roll).*": 40.0,
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*elbow.*": 30.0,
                ".*_wrist_joint": 20.0,
            },
            damping={
                ".*waist_roll.*": 25.0,
                ".*waist_yaw.*": 20.0,
                ".*_hip_pitch.*": 8.0,
                ".*_hip_roll.*": 8.0,
                ".*_hip_yaw.*": 6.0,
                ".*_knee_.*": 10.0,
                ".*_ankle_(pitch|roll).*": 2.0,
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*elbow.*": 1.5,
                ".*_wrist_joint": 1.0,
            },
            armature={
                ".*(waist|hip|knee).*": 0.02,
                ".*(ankle|shoulder|elbow|wrist).*": 0.01,
            },
            min_delay=0,
            max_delay=2,
            friction=0.1,
            dynamic_friction=0.1,
            viscous_friction={
                ".*(waist|hip|knee).*": 0.02,
                ".*(ankle|shoulder|elbow|wrist).*": 0.01,
            },
        ),
    },
)


RP1_UFO_CFG = RP1_24DOF_CFG.replace(
    actuators={
        "all_joints": RP1_24DOF_CFG.actuators["all_joints"].replace(
            stiffness={
                ".*waist_roll.*": 150.0,
                ".*waist_yaw.*": 150.0,
                ".*_hip_pitch.*": 120.0,
                ".*_hip_roll.*": 120.0,
                ".*_hip_yaw.*": 90.0,
                ".*_knee_.*": 120.0,
                ".*_ankle_(pitch|roll).*": 40.0,
                ".*_shoulder_pitch_joint": 30.0,
                ".*_shoulder_roll_joint": 30.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*elbow.*": 30.0,
                ".*_wrist_joint": 20.0,
            },
            damping={
                ".*waist_roll.*": 12.5,
                ".*waist_yaw.*": 12.5,
                ".*_hip_pitch.*": 9.0,
                ".*_hip_roll.*": 9.0,
                ".*_hip_yaw.*": 6.0,
                ".*_knee_.*": 9.0,
                ".*_ankle_(pitch|roll).*": 2.0,
                ".*_shoulder_pitch_joint": 2.5,
                ".*_shoulder_roll_joint": 2.5,
                ".*_shoulder_yaw_joint": 1.5,
                ".*elbow.*": 2.5,
                ".*_wrist_joint": 1.5,
            },
            rfi_lim=0.1,
        ),
    },
)


RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]

PR1_LINKS = [
    'base_link',
    'left_hip_pitch_link',
    'right_hip_pitch_link',
    'waist_roll_link',
    'left_hip_roll_link',
    'right_hip_roll_link',
    'waist_yaw_link',
    'left_hip_yaw_link',
    'right_hip_yaw_link',
    'left_shoulder_pitch_link',
    'right_shoulder_pitch_link',
    'left_knee_link',
    'right_knee_link',
    'left_shoulder_roll_link',
    'right_shoulder_roll_link',
    'left_ankle_pitch_link',
    'right_ankle_pitch_link',
    'left_shoulder_yaw_link',
    'right_shoulder_yaw_link',
    'left_ankle_roll_link',
    'right_ankle_roll_link',
    'left_elbow_link',
    'right_elbow_link',
    'left_wrist_link',
    'right_wrist_link',
]
