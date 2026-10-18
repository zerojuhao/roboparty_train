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

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_BODY_NAMES, RP1_JOINT_NAMES
from robolab.utils.data_paths import motion_data_dir
from robolab.utils.motion_npz import resolve_name_patterns
import robolab.tasks.manager_based.amp.mdp as mdp

from .amp_env_cfg import AmpEnvCfg, CommandsCfg, RewardsCfg
from .mdp.commands import SingleAxisVelocityCommandCfg
from .mdp.curriculums import VelocityTrackingCurriculum

# Demonstrations and discriminator observations share the explicit URDF order.
MOTION_JOINT_NAMES = RP1_JOINT_NAMES
# Keep the AMP feature order explicit.  These are all RP1 links except the
# floating articulation root (``base_link``), which is the reference frame.
KEY_BODY_NAMES = [
    ".*_hip_pitch_link",
    ".*_hip_roll_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_ankle_pitch_link",
    ".*_ankle_roll_link",
    "waist_roll_link",
    "waist_yaw_link",
    ".*_shoulder_pitch_link",
    ".*_shoulder_roll_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_link",
]
UPPER_KEY_BODY_NAMES = [
    ".*_shoulder_pitch_link",
    ".*_shoulder_roll_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_link",
]
AMP_NUM_STEPS = 1


@configclass
class RP1AmpCurriculumCfg:
    velocity_tracking = CurriculumTermCfg(
        func=VelocityTrackingCurriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "min_velocity": 1.2,
            "max_velocity": 3.0,
            "step_size": 0.1,
            "down_threshold": 0.55,
            # Require sustained tracking before increasing speed, but recover
            # quickly when the current level becomes unstable.
            "up_threshold": 0.8,
            "up_required_windows": 5,
            "down_required_windows": 2,
        },
    )


@configclass
class RP1CommandsCfg(CommandsCfg):
    base_velocity = None
    single_axis_velocity = SingleAxisVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.5, 15.0),
        rel_standing_envs=0.10,
        stop_sequence_probability=0.2,
        stop_start_time_range=(0.5, 1.5),
        stop_peak_speed_probability=0.5,
        stop_hold_time_range=(5.0, 8.0),
        braking_duration_range=None,
        acceleration_duration_range=None,
        heading_command=False,
        ranges=SingleAxisVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.2), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
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

        # Keep the inherited weight; measure sliding at contact points, not at the ankle.
        self.rewards.feet_slide.func = mdp.feet_contact_slip
        self.rewards.feet_slide.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold_ratio": 0.05,
        }

        self.motion_data.motion_dataset.motion_data_dir = motion_data_dir("rp1")
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.motion_data_weights = {
            "stand_still": 1.0,
            "walk_ff_loop_180_R_001__A048": 1.0,
            "move_back": 1.0,
            "run_start_180_R_001__A345": 1.0,
            "run_stop_180_R_001__A345": 1.0,
            "jog_forward_A021": 1.0,
            "jog_forward_A045": 1.0,
            "run_loop_180_R_003__A326": 1.0,
            "turn_l": 1.0,
            "turn_r": 1.0,
        }
        self.motion_data.motion_dataset.sampling_mode = "command_match"
        self.motion_data.motion_dataset.command_name = "single_axis_velocity"
        self.motion_data.motion_dataset.command_match_temperature = 0.3
        self.motion_data.motion_dataset.command_match_min_probability = 0.02
        self.motion_data.motion_dataset.command_match_segment_duration = 0.5
        self.motion_data.motion_dataset.command_match_current_weight = 0.3
        self.motion_data.motion_dataset.command_match_target_weight = 0.7
        # Keep commanded/measured velocity in the policy and critic, but prevent
        # AMP from using absolute root speed as a shortcut for motion style.
        self.observations.disc.base_lin_vel = None
        self.observations.disc_demo.ref_root_lin_vel_b = None
        self.observations.upper_disc.base_lin_vel = None
        self.observations.upper_disc_demo.ref_root_lin_vel_b = None
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.upper_disc.history_length = AMP_NUM_STEPS
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
                "robot", body_names=UPPER_KEY_BODY_NAMES, preserve_order=True
            )
        }
        self.observations.upper_disc_demo.ref_joint_pos.params["joint_indices"] = upper_joint_indices
        self.observations.upper_disc_demo.ref_joint_vel.params["joint_indices"] = upper_joint_indices
        resolved_key_body_names = resolve_name_patterns(RP1_BODY_NAMES, KEY_BODY_NAMES)
        resolved_upper_key_body_names = resolve_name_patterns(RP1_BODY_NAMES, UPPER_KEY_BODY_NAMES)
        self.observations.upper_disc_demo.ref_key_body_pos_b.params["body_indices"] = [
            resolved_key_body_names.index(name) for name in resolved_upper_key_body_names
        ]
        # Both discriminator routes read the same sampled animation frames;
        # the upper route only selects its joint and key-body channels.

        # Resolve RP1 body names and strengthen the RP1-only standing-pose target.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="^(?!.*ankle_roll_link$).*$"
        )
        self.rewards.stand_still_joint_deviation.weight = -0.5

        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["base_link", "waist_yaw_link"]
        )
        for term in vars(self.rewards).values():
            if term is not None and "command_name" in term.params:
                term.params["command_name"] = "single_axis_velocity"
        self.rewards.track_lin_vel_xy_exp.params["use_braking_reference"] = False
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


@configclass
class RP1AmpPathEnvCfg(RP1AmpEnvCfg):
    """Path rewards with proprioceptive actor inputs and privileged critic feedback."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.command_path_state = ObsTerm(
            func=mdp.command_path_state, params={"command_name": "single_axis_velocity"}
        )
        self.rewards.flat_orientation_l2 = None
        for name, weight in (
            ("lean_orientation_l2", -6.0),
            ("shoulder_roll_excess", -1.5),
            ("high_speed_double_flight", 4.0),
        ):
            getattr(self.rewards, name).weight = weight


@configclass
class RP1AmpPathAnkleSlipEnvCfg(RP1AmpPathEnvCfg):
    """A/B control using the original shared AMP ankle-velocity penalty."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide = RewardsCfg().feet_slide


@configclass
class RP1AmpPathEnvCfg_PLAY(RP1AmpPathEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.velocity_tracking = None
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.observations.policy.enable_corruption = False
