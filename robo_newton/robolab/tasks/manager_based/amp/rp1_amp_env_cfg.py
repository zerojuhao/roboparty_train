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

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.assets.robots.roboparty import RP1_24DOF_CFG
from robolab.utils.data_paths import motion_data_dir
import robolab.tasks.manager_based.amp.mdp as mdp

from .amp_env_cfg import AmpEnvCfg, CommandsCfg
from .mdp.commands import FilteredUniformVelocityCommandCfg
from .mdp.curriculums import VelocityTrackingCurriculum

# AMP discriminator end-effectors. Full-body key points duplicate joint_pos and
# make the disc too easy to saturate; ankles/knees/wrists match parkour AMP.
KEY_BODY_NAMES = [
    ".*_ankle_roll_link",
    ".*_knee_link",
    ".*_elbow_link",
    ".*_wrist_link",
    ".*_shoulder_pitch_link",
]
AMP_NUM_STEPS = 3


@configclass
class RP1AmpCurriculumCfg:
    velocity_tracking = CurriculumTermCfg(
        func=VelocityTrackingCurriculum,
        params={
            "command_name": "base_velocity",
            "min_velocity": 1.2,
            "max_velocity": 3.0,
            "step_size": 0.1,
            "up_threshold": 0.5,
            "down_threshold": 0.6,
            "up_required_windows": 5,
            "down_required_windows": 2,
        },
    )


@configclass
class RP1CommandsCfg(CommandsCfg):
    base_velocity = FilteredUniformVelocityCommandCfg(
        asset_name="robot",
        min_abs_lin_velocity=0.3,
        min_abs_ang_velocity=0.5,
        single_axis_probability=0.6,
        resampling_time_range=(5.0, 15.0),
        rel_standing_envs=0.10,
        heading_command=False,
        ranges=FilteredUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
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
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.78)
        self.sim.use_newton_actuators = False

        self.motion_data.motion_dataset.motion_data_dir = motion_data_dir("rp1")
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.motion_data_weights = {

            "neutral_walk_ff_180_R_002__A534": 1.0,
            "neutral_walk_ff_180_R_002__A534_M": 1.0,
            # "walk_ff_loop_180_R_001__A048": 1.0,
            # "walk_arc_ccw_loop_L_002__A041": 1.0,
            # "walk_arc_cw_loop_001__A048": 1.0,
            
            # "run_start_180_R_001__A345": 1.0,
            # "run_stop_180_R_001__A345": 1.0,
            # "run_loop_180_R_003__A326": 1.0,

            "jog_forward_A021": 1.0,
            "jog_arc_ccw_loop_003__A041": 1.0,
            "jog_arc_cw_loop_004__A045": 1.0,
            
            "stand_still": 1.0,
            "move_back": 1.0,
            "move_l": 1.0,
            "move_r": 1.0,
            "turn_l": 1.0,
            "turn_r": 1.0,

            # ACCAD male retargets (50Hz, hhtools -> 4-key xyzw)
            "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            # "B9_-__Walk_turn_left_90_stageii": 1.0,
            # "B10_-__Walk_turn_left_45_stageii": 1.0,
            # "B13_-__Walk_turn_right_90_stageii": 1.0,
            # "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            # "B15_-__Walk_turn_around_stageii": 1.0,
            # "Walk_B16_-_Walk_turn_change_stageii": 1.0,

            "C1_-_stand_to_run_stageii": 1.0,
            "C3_-_run_stageii": 1.0,
            "C5_-_walk_to_run_stageii": 1.0,
            "C11_-_run_turn_left_90_stageii": 1.0,
            "C12_-_run_turn_left_45_stageii": 1.0,
            "C14_-_run_turn_right_90_stageii": 1.0,
            "C15_-_run_turn_right_45_stageii": 1.0,

        }
        # self.motion_data.motion_dataset.sampling_mode = "command_match"
        self.motion_data.motion_dataset.command_name = "base_velocity"
        self.motion_data.motion_dataset.command_match_temperature = 0.5
        self.motion_data.motion_dataset.command_match_min_probability = 0.05
        self.motion_data.motion_dataset.command_match_direction_weight = 1.0
        self.motion_data.motion_dataset.command_match_direction_threshold = 0.3
        self.motion_data.motion_dataset.command_match_current_weight = 0.2
        self.motion_data.motion_dataset.command_match_target_weight = 0.8
        self.motion_data.motion_dataset.command_match_static_threshold = 0.2
        self.motion_data.motion_dataset.command_match_static_motion_names = ("stand_still",)
        self.motion_data.motion_dataset.command_match_cruise_speed_threshold = 0.3
        self.motion_data.motion_dataset.command_match_cruise_min_fraction = 0.1
        # Keep commanded/measured velocity in the policy and critic, but prevent
        # AMP from using absolute root speed as a shortcut for motion style.
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["base_link", "waist_yaw_link"]
        )
        for term in vars(self.rewards).values():
            if term is not None and "command_name" in term.params:
                term.params["command_name"] = "base_velocity"
        self.observations.policy.velocity_commands.params["command_name"] = "base_velocity"
        self.observations.critic.velocity_commands.params["command_name"] = "base_velocity"
        self.curriculum.velocity_tracking.params["command_name"] = "base_velocity"


@configclass
class RP1AmpEnvCfg_PLAY(RP1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.velocity_tracking = None
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
