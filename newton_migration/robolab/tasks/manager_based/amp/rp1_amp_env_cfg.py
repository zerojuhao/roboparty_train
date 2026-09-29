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

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RP1_24DOF_CFG

from .amp_env_cfg import AmpEnvCfg, CommandsCfg
from .mdp.commands import SingleAxisVelocityCommandCfg
from .mdp.curriculums import VelocityTrackingCurriculum

# Match lab_dof_names and lab_key_body_names in retarget/config/rp1_24dof.yaml.
MOTION_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_yaw_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_joint", "right_wrist_joint",
]
KEY_BODY_NAMES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_knee_link", "right_knee_link",
    "left_wrist_link", "right_wrist_link",
]
AMP_NUM_STEPS = 3


@configclass
class RP1AmpCurriculumCfg:
    velocity_tracking = CurriculumTermCfg(
        func=VelocityTrackingCurriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "min_velocity": 0.5,
            "max_velocity": 4.0,
            "step_size": 0.5,
            "down_threshold": 0.5,
            "up_threshold": 0.8,
        },
    )


@configclass
class RP1CommandsCfg(CommandsCfg):
    base_velocity = None
    single_axis_velocity = SingleAxisVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.5, 15.0),
        rel_standing_envs=0.0,
        heading_command=False,
        ranges=SingleAxisVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class RP1AmpEnvCfg(AmpEnvCfg):
    """RP1 locomotion with the shared AMP rewards."""

    curriculum: RP1AmpCurriculumCfg = RP1AmpCurriculumCfg()
    commands: RP1CommandsCfg = RP1CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.use_newton_actuators = False

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rp1_24dof_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "stand_still": 1.0,
            "walk_arc_cw_loop_001__A048_M": 5.0,
            "walk_arc_cw_loop_001__A048": 5.0,
            "walk_ff_loop_180_R_001__A048": 5.0,
            "run_start_180_R_001__A345": 10.0,
            "run_stop_180_R_001__A345": 10.0,
            "move_l": 1.0,
            "move_r": 1.0,
        }
        for term in (self.observations.disc.joint_pos, self.observations.disc.joint_vel):
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=MOTION_JOINT_NAMES, preserve_order=True
            )
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        upper_joint_names = [
            name for name in MOTION_JOINT_NAMES
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        ]
        upper_joint_indices = [MOTION_JOINT_NAMES.index(name) for name in upper_joint_names]
        self.observations.upper_disc.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=upper_joint_names, preserve_order=True
        )
        self.observations.upper_disc.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=upper_joint_names, preserve_order=True
        )
        self.observations.upper_disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_link", "right_wrist_link"], preserve_order=True
            )
        }
        self.observations.upper_disc_demo.ref_joint_pos.params["joint_indices"] = upper_joint_indices
        self.observations.upper_disc_demo.ref_joint_vel.params["joint_indices"] = upper_joint_indices
        self.observations.upper_disc_demo.ref_key_body_pos_b.params["body_indices"] = [4, 5]

        # Keep the shared reward functions and weights; only resolve RP1 body names.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_hip_yaw_link"
        )
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names="^(?!.*ankle_roll_link$).*$",
        )
        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["base_link", "waist_yaw_link"]
        )
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="waist_yaw_link"
        )

        self.rewards.track_lin_vel_xy_exp.params["command_name"] = "single_axis_velocity"
        self.rewards.track_ang_vel_z_exp.params["command_name"] = "single_axis_velocity"
        self.observations.policy.velocity_commands.params["command_name"] = "single_axis_velocity"
        self.observations.critic.velocity_commands.params["command_name"] = "single_axis_velocity"
        self.curriculum.velocity_tracking.params["command_name"] = "single_axis_velocity"


@configclass
class RP1AmpEnvCfg_PLAY(RP1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.velocity_tracking = None
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
