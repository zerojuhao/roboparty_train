"""Commanded locomotion with disturbances and zero-command compliance with one actor."""

import math

from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
import robolab.tasks.manager_based.amp.mdp as mdp

from .mdp.arm_compliance import arm_compliance, full_body_return
from .mdp.body_compliance import body_compliance
from .mdp.events import sustained_standing_push
from .mdp.force_estimation import compact_interaction_critic_state
from .mdp.soft import (
    SoftVelocityCommand,
    amp_task_route,
    soft_feet_air_time,
    yaw_compliance,
)
from .rp1_amp_env_cfg import RP1AmpEnvCfg


@configclass
class CompactInteractionCriticCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=compact_interaction_critic_state)
    concatenate_terms = True
    enable_corruption = False


@configclass
class AmpTaskRouteCfg(ObservationGroupCfg):
    task = ObservationTermCfg(func=amp_task_route)
    concatenate_terms = True
    enable_corruption = False


@configclass
class RP1AmpSoftEnvCfg(RP1AmpEnvCfg):
    compliance_env_fraction: float = 0.5
    command_compliance_env_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self._configure_observations()
        self._configure_commands()
        self._configure_disturbances()
        self._configure_motion_data()
        self._configure_rewards()

    def _configure_observations(self):
        self.observations.policy.history_length = 8
        self.observations.compact_interaction_critic = CompactInteractionCriticCfg()
        self.observations.amp_route = AmpTaskRouteCfg()

        # Derive policy/demo indices from the same ordering for the lower-body discriminator.
        joints = self.scene.robot.joint_ordering
        lower_indices = [
            i for i, name in enumerate(joints)
            if any(part in name for part in ('hip_', 'knee_', 'ankle_'))
            or name in ('waist_roll_joint', 'waist_yaw_joint')
        ]
        lower_names = [joints[i] for i in lower_indices]
        key_cfg = self.observations.disc.key_body_pos_b.params['asset_cfg']
        _, key_names = resolve_matching_names(
            key_cfg.body_names, self.scene.robot.body_ordering, preserve_order=key_cfg.preserve_order,
        )
        key_indices = [i for i, name in enumerate(key_names) if 'knee_' in name or 'ankle_' in name]
        assert len(lower_names) == 14 and len(key_indices) == 4

        lower = self.observations.disc.copy()
        lower.joint_pos.params = {
            'asset_cfg': SceneEntityCfg('robot', joint_names=lower_names, preserve_order=True),
        }
        lower.joint_vel.params = lower.joint_pos.params.copy()
        lower.key_body_pos_b.params = {
            'asset_cfg': SceneEntityCfg(
                'robot', body_names=[key_names[i] for i in key_indices], preserve_order=True,
            ),
        }
        self.observations.disc_lower = lower

        lower_demo = self.observations.disc_demo.copy()
        lower_demo.ref_joint_pos.params['joint_indices'] = lower_indices
        lower_demo.ref_joint_vel.params['joint_indices'] = lower_indices
        lower_demo.ref_key_body_pos_b.params['body_indices'] = key_indices
        self.observations.disc_lower_demo = lower_demo

    def _configure_commands(self):
        self.curriculum.velocity_tracking = None
        command = self.commands.base_velocity
        command.class_type = SoftVelocityCommand
        command.ranges.lin_vel_x = (-0.5, 1.0)
        command.ranges.lin_vel_y = (-0.5, 0.5)
        command.ranges.ang_vel_z = (-1.0, 1.0)
        command.max_lin_velocity = 1.0
        command.min_abs_lin_velocity = 0.05
        command.min_abs_ang_velocity = 0.1
        command.rel_standing_envs = 0.05
        command.single_axis_probability = 0.35
        command.resampling_time_range = (3.0, 7.0)

    def _configure_disturbances(self):
        self.events.push_robot.params['velocity_range'] = {
            'x': (-0.3, 0.3),
            'y': (-0.3, 0.3),
            'yaw': (-0.3, 0.3),
        }
        self.events.sustained_standing_push = EventTermCfg(
            func=sustained_standing_push,
            mode='interval',
            interval_range_s=(0.0, 0.0),
            params={
                'asset_cfg': SceneEntityCfg(
                    'robot',
                    body_names=[
                        'left_wrist_link',
                        'right_wrist_link',
                        'left_shoulder_pitch_link',
                        'left_shoulder_roll_link',
                        'left_shoulder_yaw_link',
                        'left_elbow_link',
                        'right_shoulder_pitch_link',
                        'right_shoulder_roll_link',
                        'right_shoulder_yaw_link',
                        'right_elbow_link',
                        'waist_yaw_link',
                    ],
                    preserve_order=True,
                ),
                'command_name': 'base_velocity',
                # Each load below is drawn independently once per event; all four may overlap.
                # Arm point pulls: 1-4 distinct links, uniformly random directions.
                'point_force_probability': 0.5,
                'point_force_range_n': (0.0, 20.0),
                'contact_count_range': (1, 4),
                'point_offset_range': (-0.03, 0.03),
                # Variants of the point pulls: one shared direction, one joint couple
                # (non-coherent multi-point pulls only), or a single horizontal pull.
                'coherent_pull_probability': 0.3,
                'arm_couple_probability': 0.25,
                'small_horizontal_pull_probability': 0.3,
                # Horizontal torso pull, only when the event starts at zero command.
                'torso_horizontal_pull_probability': 0.5,
                'torso_horizontal_force_range_n': (0.0, 20.0),
                # Torso yaw torque, both signs; likelier when the event starts at zero command.
                'torso_yaw_probability': 0.25,
                'zero_command_torso_yaw_probability': 0.5,
                'torso_yaw_torque_range_nm': (0.0, 10.0),
                # One of the ten arm joints, both signs, only when the event starts at zero command.
                'arm_joint_torque_probability': 0.5,
                'arm_joint_torque_range_nm': (0.0, 5.0),
                # Like a hand easing off: outward torque fades to zero at the soft limit.
                'arm_joint_limit_margin_rad': math.radians(25.),
                # Summed point force as a fraction of body weight.
                'max_total_force_ratio': 0.20,
                # Timing.
                'duration_range_s': (1.0, 3.0),
                'long_duration_probability': 0.25,
                'long_duration_range_s': (10.0, 12.0),
                'rest_range_s': (4.0, 8.0),
                'initial_rest_range_s': (0.2, 0.6),
                'ramp_time_range_s': (0.05, 0.4),
                'abrupt_release_probability': 0.25,
                'release_reserve_s': 3.0,
            },
        )

    def _configure_motion_data(self):
        self.animation.animation.random_fetch = True
        dataset = self.motion_data.motion_dataset
        dataset.motion_data_weights = {
            'neutral_walk_ff_180_R_002__A534': 1.0,
            'neutral_walk_ff_180_R_002__A534_M': 1.0,
            'walk_arc_ccw_loop_L_002__A041': 1.0,
            'walk_arc_cw_loop_001__A048': 1.0,
            'stand_still': 1.0,
            'move_back': 1.0,
            'move_l': 1.0,
            'move_r': 1.0,
            'turn_l': 1.0,
            'turn_r': 1.0,
            'B1_-_stand_to_walk_stageii': 1.0,
            'B4_-_Stand_to_Walk_backwards_stageii': 1.0,
        }
        # Commanded walking matches user commands; compliance keeps the motion-weight prior.
        dataset.sampling_mode = 'command_match'
        dataset.command_name = 'base_velocity'
        dataset.command_match_static_threshold = 0.05
        dataset.command_match_static_motion_names = ('stand_still',)

    def _configure_rewards(self):
        # q_target = q_default + 0.25 * action: also penalizes target-position curvature.
        self.rewards.action_second_difference_l2 = RewTerm(
            func=mdp.action_second_difference_l2, weight=-0.01,
        )
        # Local arm yielding and unloaded arm return share the existing reward interface.
        self.rewards.stand_still = None
        # Resolve the same initial joint pose used by the articulation and action offsets.
        defaults = dict.fromkeys(self.scene.robot.joint_ordering, 0.0)
        _, names, values = resolve_matching_names_values(
            self.scene.robot.init_state.joint_pos, self.scene.robot.joint_ordering,
        )
        defaults.update(zip(names, values))
        self.rewards.arm_compliance = RewTerm(
            func=arm_compliance,
            weight=1.0,
            params={
                'asset_cfg': SceneEntityCfg(
                    'robot', body_names=['left_wrist_link', 'right_wrist_link'], preserve_order=True,
                ),
                'force_deadband_bw': 0.015,
                'force_scale_bw': 0.035,
                'force_filter_time_s': 0.05,
                'command_name': 'base_velocity',
                'power_scale_w': 5.0,
                'torque_scale_nm': 2.0,
                'damping_allowance': 8.0,
                'resistance_tolerance_nm': 0.5,
                'limit_margin': 0.25,
                'max_joint_speed': 0.8,
                'overspeed_weight': 0.1,
                'all_arm_links': True,
                'passive_joint_torque_range_nm': (1.0, 2.0),
                'rest_joint_pos': {
                    name: value for name, value in defaults.items()
                    if any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))
                },
            },
        )

        self.rewards.full_body_return = RewTerm(
            func=full_body_return,
            weight=-1.2,
            params={
                'asset_cfg': SceneEntityCfg('robot'),
                'command_name': 'base_velocity',
                'return_std': 0.1,
                'motion_target_threshold': 0.3,
                'posture_weight': 3.0,
                'recovery_time_s': 0.3,
                'arm_velocity_weight': 2.0,
                'return_max_speed': 0.4,
                'arm_fast_return_weight': 8.0,
                'return_speed_threshold': 0.8,
            },
        )

        # Command tracking and force-driven body motion.
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=body_compliance,
            weight=1.5,
            params={
                'command_name': 'base_velocity',
                'std': 0.5,
                # Force-follow speed limits (forward, backward, lateral) match the command ranges.
                'max_speed': (1.0, 0.5, 0.5),
                'target_accel': 2.0,
                'force_deadband_bw': 0.03,
                'force_scale_bw': 0.03,
                'force_filter_time_s': 0.1,
                # 15 N saturates the walk ellipse; 10 N deadband stays above arm teaching.
                'force_range_n': (10.0, 15.0),
                'target_force_range_n': (10.0, 15.0),
                'target_tanh_k': 2.0,
                'target_sustain_s': 0.1,
            },
        )
        self.rewards.track_ang_vel_z_exp.func = yaw_compliance
        self.rewards.track_ang_vel_z_exp.params.update(
            max_yaw_rate=1.0,
            target_accel=2.0,
            reference_length=0.25,
            torque_deadband_bw=0.01,
            torque_scale_bw=0.03,
            filter_time_s=0.1,
            target_torque_range_nm=(2.0, 5.0),
            target_tanh_k=2.0,
            target_sustain_s=0.1,
        )
        # Completed steps toward the command or force target: credit is linear in swing time up to
        # 0.4 s (0.3 * swing per landing) and scales with foot lift up to 5 cm.
        self.rewards.feet_air_time_positive_biped = RewTerm(
            func=soft_feet_air_time,
            weight=1.5,
            params={
                'command_name': 'base_velocity',
                'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll_link'),
                'asset_cfg': SceneEntityCfg('robot', body_names='.*ankle_roll_link'),
                'threshold': 0.3,
                # Slow commanded turns (0.1-0.3 rad/s) also earn stepping credit.
                'command_yaw_threshold': 0.10,
                'target_air_time': 0.4,
                'target_height': 0.05,
            },
        )
        # Landing impact charged once per touchdown event (not per dt-weighted step).
        self.rewards.feet_impact_velocity_l2.params['per_event'] = True
        self.rewards.feet_impact_velocity_l2.weight = -1.0
        self.rewards.feet_impact_velocity_l2.params['downward_speed_threshold'] = 0.10

        # Light foot contacts during command tracking and external-force following.
        self.rewards.feet_contact_force_l2.params.update(
            in_place_turn_weights=None,
            peak_weight=0.7,
            mean_weight=0.2,
            # Impulsive touchdown loading (braking concentrated at contact) is charged like the peak.
            rising_weight=1.0,
            # Soft speeds stay below the 1.2 m/s blend start, so the overload threshold is a constant 1.0 BW.
            low_force_threshold=1.0,
            force_threshold=1.5,
        )

        # Torso balance and hip/waist posture.
        self.rewards.pelvis_orientation_l2.weight = -2.0
        self.rewards.pelvis_orientation_l2.params['asset_cfg'] = SceneEntityCfg(
            'robot', body_names=['base_link', 'waist_yaw_link'],
        )
        self.rewards.pelvis_ang_vel_xy_l2.weight = -1.0
        self.rewards.pelvis_ang_vel_xy_l2.params['asset_cfg'] = SceneEntityCfg(
            'robot', body_names=['base_link', 'waist_yaw_link'],
        )
        self.rewards.feet_orientation_l2 = RewTerm(
            func=mdp.feet_orientation_l2,
            weight=-0.1,
            params={
                'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll_link'),
                'asset_cfg': SceneEntityCfg('robot', body_names='.*ankle_roll_link'),
            },
        )
        self.rewards.feet_close_y_gauss.params['threshold'] = 0.24
        self.rewards.knee_close_y_gauss.params['threshold'] = 0.18
        self.rewards.hip_yaw_deviation_l1.weight = -0.1
        self.rewards.hip_yaw_deviation_l1.params.update(
            asset_cfg=SceneEntityCfg('robot',
                joint_names=['left_hip_yaw_joint', 'right_hip_yaw_joint'], preserve_order=True),
            std=math.radians(15.),
            ang_vel_gate=0.3,
            yaw_reference_term='track_ang_vel_z_exp',
        )


@configclass
class RP1AmpSoftEnvCfg_PLAY(RP1AmpSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.compliance_env_fraction = 1.0
        self.command_compliance_env_fraction = 0.0
        self.observations.policy.enable_corruption = False
