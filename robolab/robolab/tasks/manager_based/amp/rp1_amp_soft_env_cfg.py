"""RP1 AMP Soft: commanded walking plus zero-command compliance.

This task owns its observations, rewards, commands, and events. It uses AmpEnvCfg
only for the shared AMP scene, physics, animation, and motion-data machinery.
"""

import math

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robolab.assets.robots.roboparty import RP1_24DOF_CFG
from robolab.utils.data_paths import motion_data_dir
import robolab.tasks.manager_based.amp.mdp as mdp

from .amp_env_cfg import ANIMATION_TERM_NAME, AmpEnvCfg, EventCfg
from .mdp.arm_compliance import arm_compliance, full_body_return
from .mdp.body_compliance import body_compliance
from .mdp.commands import FilteredUniformVelocityCommandCfg
from .mdp.events import sustained_standing_push
from .mdp.force_estimation import compact_interaction_critic_state, force_vel_target
from .mdp.soft import SoftVelocityCommand, amp_task_route, soft_feet_air_time, yaw_compliance
from .rp1_amp_env_cfg import AMP_NUM_STEPS, KEY_BODY_NAMES

LOWER_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_roll_joint', 'waist_yaw_joint',
]
LOWER_KEY_BODY_NAMES = [
    'left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_link', 'right_knee_link',
]
ARM_REST_JOINT_POS = {
    'left_shoulder_pitch_joint': 0.2,
    'left_shoulder_roll_joint': 0.2,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': -1.2,
    'left_wrist_joint': 0.0,
    'right_shoulder_pitch_joint': -0.2,
    'right_shoulder_roll_joint': -0.2,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 1.2,
    'right_wrist_joint': 0.0,
}


@configclass
class SoftCommandsCfg:
    base_velocity = FilteredUniformVelocityCommandCfg(
        class_type=SoftVelocityCommand,
        asset_name='robot',
        min_abs_lin_velocity=0.05,
        min_abs_ang_velocity=0.1,
        single_axis_probability=0.35,
        resampling_time_range=(3.0, 7.0),
        rel_standing_envs=0.05,
        heading_command=False,
        max_lin_velocity=1.0,
        ranges=FilteredUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class SoftObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={'command_name': 'base_velocity'})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 8
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=3)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=3)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={'command_name': 'base_velocity'}, history_length=3,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel, history_length=3)
        actions = ObsTerm(func=mdp.last_action, history_length=3)
        compact_interaction = ObsTerm(func=compact_interaction_critic_state)
        force_vel_target = ObsTerm(func=force_vel_target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DiscriminatorCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={'asset_cfg': SceneEntityCfg('robot', body_names=KEY_BODY_NAMES, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = AMP_NUM_STEPS
            self.flatten_history_dim = False

    @configclass
    class DiscriminatorLowerCfg(DiscriminatorCfg):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=LOWER_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={'asset_cfg': SceneEntityCfg('robot', joint_names=LOWER_JOINT_NAMES, preserve_order=True)},
        )
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={'asset_cfg': SceneEntityCfg('robot', body_names=LOWER_KEY_BODY_NAMES, preserve_order=True)},
        )

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_lin_vel_b = ObsTerm(
            func=mdp.ref_root_lin_vel_b, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b, params={'animation': ANIMATION_TERM_NAME, 'flatten_steps_dim': False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    @configclass
    class DiscriminatorLowerDemoCfg(DiscriminatorDemoCfg):
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                'animation': ANIMATION_TERM_NAME,
                'flatten_steps_dim': False,
                'joint_indices': list(range(len(LOWER_JOINT_NAMES))),
            },
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                'animation': ANIMATION_TERM_NAME,
                'flatten_steps_dim': False,
                'joint_indices': list(range(len(LOWER_JOINT_NAMES))),
            },
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                'animation': ANIMATION_TERM_NAME,
                'flatten_steps_dim': False,
                'body_indices': list(range(len(LOWER_KEY_BODY_NAMES))),
            },
        )

    @configclass
    class AmpRouteCfg(ObsGroup):
        task = ObsTerm(func=amp_task_route)
        concatenate_terms = True
        enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    disc: DiscriminatorCfg = DiscriminatorCfg()
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()
    disc_lower: DiscriminatorLowerCfg = DiscriminatorLowerCfg()
    disc_lower_demo: DiscriminatorLowerDemoCfg = DiscriminatorLowerDemoCfg()
    amp_route: AmpRouteCfg = AmpRouteCfg()


@configclass
class SoftRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=body_compliance,
        weight=1.5,
        params={
            'command_name': 'base_velocity',
            'std': 0.5,
            'max_speed': (1.0, 0.5, 0.5),
            'target_accel': 2.0,
            'force_deadband_bw': 0.03,
            'force_scale_bw': 0.03,
            'force_filter_time_s': 0.1,
            'force_range_n': (10.0, 15.0),
            'target_force_range_n': (10.0, 15.0),
            'target_tanh_k': 2.0,
            'target_sustain_s': 0.1,
            'force_reward_scale': 2.0,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=yaw_compliance,
        weight=1.5,
        params={
            'command_name': 'base_velocity',
            'std': 0.5,
            'max_yaw_rate': 1.0,
            'target_accel': 2.0,
            'reference_length': 0.25,
            'torque_deadband_bw': 0.01,
            'torque_scale_bw': 0.03,
            'filter_time_s': 0.1,
            'target_torque_range_nm': (2.0, 5.0),
            'target_tanh_k': 2.0,
            'target_sustain_s': 0.1,
        },
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    feet_air_time_positive_biped = RewTerm(
        func=soft_feet_air_time,
        weight=1.5,
        params={
            'command_name': 'base_velocity',
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll_link'),
            'asset_cfg': SceneEntityCfg('robot', body_names='.*ankle_roll_link'),
            'threshold': 0.3,
            'command_yaw_threshold': 0.10,
            'target_air_time': 0.4,
            'target_height': 0.05,
        },
    )
    arm_compliance = RewTerm(
        func=arm_compliance,
        weight=1.5,
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
            'yield_scale_w': 2.0,
            'damping_allowance': 8.0,
            'resistance_tolerance_nm': 0.0,
            'limit_margin': 0.25,
            'max_joint_speed': 1.0,
            'overspeed_weight': 0.05,
            'all_arm_links': True,
            'passive_joint_torque_range_nm': (1.0, 2.0),
            'rest_joint_pos': ARM_REST_JOINT_POS,
        },
    )
    full_body_return = RewTerm(
        func=full_body_return,
        weight=-1.2,
        params={
            'asset_cfg': SceneEntityCfg('robot'),
            'command_name': 'base_velocity',
            'return_std': 0.1,
            'posture_weight': 3.0,
            'recovery_time_s': 0.3,
            'arm_velocity_weight': 2.0,
            'return_max_speed': 0.4,
            'arm_fast_return_weight': 8.0,
            'return_speed_threshold': 0.8,
        },
    )
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation,
        weight=-2.0,
        params={'asset_cfg': SceneEntityCfg('robot', body_names=['base_link', 'waist_yaw_link'])},
    )
    pelvis_ang_vel_xy_l2 = RewTerm(
        func=mdp.link_ang_vel_xy_l2,
        weight=-1.0,
        params={'asset_cfg': SceneEntityCfg('robot', body_names=['base_link', 'waist_yaw_link'])},
    )
    hip_yaw_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            'asset_cfg': SceneEntityCfg(
                'robot', joint_names=['left_hip_yaw_joint', 'right_hip_yaw_joint'], preserve_order=True,
            ),
            'command_name': 'base_velocity',
            'std': math.radians(15.),
            'ang_vel_gate': 0.3,
            'yaw_reference_term': 'track_ang_vel_z_exp',
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.1,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll_link'),
            'asset_cfg': SceneEntityCfg('robot', body_names='.*ankle_roll_link'),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*ankle_roll_link'),
            'asset_cfg': SceneEntityCfg('robot', body_names='.*_ankle_roll_link'),
            'contact_threshold': 1.0,
        },
    )
    feet_close_y_gauss = RewTerm(
        func=mdp.link_close_y_gauss,
        weight=-0.1,
        params={
            'threshold': 0.24,
            'std': 0.1,
            'asset_cfg': SceneEntityCfg(
                'robot', body_names=['left_ankle_roll_link', 'right_ankle_roll_link'], preserve_order=True,
            ),
        },
    )
    knee_close_y_gauss = RewTerm(
        func=mdp.link_close_y_gauss,
        weight=-0.1,
        params={
            'threshold': 0.18,
            'std': 0.1,
            'asset_cfg': SceneEntityCfg(
                'robot', body_names=['left_knee_link', 'right_knee_link'], preserve_order=True,
            ),
        },
    )
    sound_suppression = RewTerm(
        func=mdp.sound_suppression,
        weight=-0.5,
        params={
            'sensor_cfg': SceneEntityCfg(
                'contact_forces',
                body_names=['left_ankle_roll_link', 'right_ankle_roll_link'],
                preserve_order=True,
            ),
            'asset_cfg': SceneEntityCfg(
                'robot',
                body_names=['left_ankle_roll_link', 'right_ankle_roll_link'],
                preserve_order=True,
            ),
            'contact_threshold': 1.0,
        },
    )
    feet_contact_force_l2 = RewTerm(
        func=mdp.feet_contact_force_l2,
        weight=-3.0,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'),
            'asset_cfg': SceneEntityCfg('robot'),
            'command_name': 'base_velocity',
            'force_threshold': 1.5,
            'low_force_threshold': 1.0,
            'low_speed': 1.2,
            'high_speed': 2.0,
            'peak_weight': 0.7,
            'mean_weight': 0.2,
            'rising_weight': 1.0,
            'in_place_turn_weights': None,
        },
    )
    feet_impact_velocity_l2 = RewTerm(
        func=mdp.feet_impact_velocity_l2,
        weight=-1.0,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'),
            'asset_cfg': SceneEntityCfg('robot', body_names='.*_ankle_roll_link'),
            'downward_speed_threshold': 0.10,
            'per_event': True,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            'sensor_cfg': SceneEntityCfg('contact_forces', body_names='^(?!.*ankle_roll_link$).*$'),
            'threshold': 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_second_difference_l2 = RewTerm(func=mdp.action_second_difference_l2, weight=-0.01)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=-1e-5)
    joint_torque_saturation_l2 = RewTerm(
        func=mdp.joint_torque_saturation_l2,
        weight=-0.1,
        params={
            'limit_ratio': 0.95,
            'use_substep_peak': True,
            'asset_cfg': SceneEntityCfg('robot', joint_names=['.*']),
        },
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)


@configclass
class SoftEventCfg(EventCfg):
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode='startup',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names=['base_link', 'waist_yaw_link']),
            'com_range': {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'z': (-0.03, 0.03)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode='interval',
        interval_range_s=(5.0, 10.0),
        params={'velocity_range': {'x': (-0.3, 0.3), 'y': (-0.3, 0.3), 'yaw': (-0.3, 0.3)}},
    )
    sustained_standing_push = EventTerm(
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
            'point_force_probability': 0.7,
            'point_force_range_n': (0.0, 20.0),
            'contact_count_range': (1, 4),
            'point_offset_range': (-0.03, 0.03),
            'coherent_pull_probability': 0.3,
            'arm_couple_probability': 0.25,
            'small_horizontal_pull_probability': 0.3,
            'torso_horizontal_pull_probability': 0.7,
            'torso_horizontal_force_range_n': (0.0, 20.0),
            'torso_yaw_probability': 0.25,
            'zero_command_torso_yaw_probability': 0.7,
            'torso_yaw_torque_range_nm': (0.0, 10.0),
            'arm_joint_torque_probability': 0.7,
            'arm_joint_torque_range_nm': (0.0, 5.0),
            'arm_joint_limit_margin_rad': math.radians(25.),
            'max_total_force_ratio': 0.20,
            'duration_range_s': (1.0, 3.0),
            'long_duration_probability': 0.25,
            'long_duration_range_s': (10.0, 12.0),
            'rest_range_s': (2.0, 4.0),
            'initial_rest_range_s': (0.2, 0.6),
            'ramp_time_range_s': (0.05, 0.4),
            'abrupt_release_probability': 0.25,
            'release_reserve_s': 3.0,
        },
    )


@configclass
class RP1AmpSoftEnvCfg(AmpEnvCfg):
    compliance_env_fraction: float = 0.3
    command_compliance_env_fraction: float = 0.7
    observations: SoftObservationsCfg = SoftObservationsCfg()
    commands: SoftCommandsCfg = SoftCommandsCfg()
    rewards: SoftRewardsCfg = SoftRewardsCfg()
    events: SoftEventCfg = SoftEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = {
            '.*(hip|knee|ankle|waist).*': 0.25,
            '.*(shoulder|elbow|wrist).*': 3.0,
        }
        self.scene.robot = RP1_24DOF_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.78)
        self.sim.use_newton_actuators = False
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.animation.animation.random_fetch = True
        dataset = self.motion_data.motion_dataset
        dataset.motion_data_dir = motion_data_dir('rp1')
        dataset.key_body_names = KEY_BODY_NAMES
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
        dataset.sampling_mode = 'command_match'
        dataset.command_name = 'base_velocity'
        dataset.command_match_static_threshold = 0.05
        dataset.command_match_static_motion_names = ('stand_still',)


@configclass
class RP1AmpSoftEnvCfg_PLAY(RP1AmpSoftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.compliance_env_fraction = 1.0
        self.command_compliance_env_fraction = 0.0
        self.observations.policy.enable_corruption = False
