"""Standalone RP1 PUFO environment with visual and terrain context."""

from __future__ import annotations

import math
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import LegacyRayCaster
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from robolab.assets.robots.roboparty import PR1_LINKS, RP1_UFO_CFG
from robolab.sensors import NoisyGroupedRayCasterCameraCfg, get_link_prim_targets
from robolab.utils.noise import (
    CropAndResizeCfg,
    DepthNormalizationCfg,
    DepthSanitizationCfg,
    GaussianBlurNoiseCfg,
    PixelFailureNoiseCfg,
    RandomConvNoiseCfg,
    RangeBasedGaussianNoiseCfg,
    ScaleRandomizationNoiseCfg,
)

from . import mdp
from .pufo_terrain_cfg import PUFO_TERRAIN_CFG, PUFO_TILE_SIZE
from .mdp.pufo_terrain_importer import PUFOTerrainImporter
from robolab.sensors.terrain_contact import TerrainContactSensorCfg
from .mdp.pufo_physics import PUFOMJWarpSolverCfg
from .lafan_selected import MOTION_DIR, MOTION_NAMES
from .mdp.pufo_reset import reset_pufo_from_motion
from robolab.terrains.terrain_importer_cfg import TerrainImporterCfg as FiledTerrainImporterCfg


@configclass
class PUFOSceneCfg(InteractiveSceneCfg):
    terrain = FiledTerrainImporterCfg(
        class_type=PUFOTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PUFO_TERRAIN_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    contact_forces = TerrainContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_friction_forces=True,
        track_air_time=False,
    )
    terrain_priv = RayCasterCfg(
        # Native Newton BVH rays also hit the robot and ignore mesh_prim_paths.
        class_type=LegacyRayCaster,
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        mesh_prim_paths=["/World/ground/terrain"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(2.0, 1.2),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 2.0)),
        # Keep rays above raised terrain without losing downward reach during jumps.
        max_distance=20.0,
        update_period=0.02,
    )
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/waist_roll_link/waist_yaw_link",
        mesh_prim_paths=["/World/ground/terrain"],
        ray_alignment="base",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=64,
            height=36,
        ),
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.09175, 0.011, 0.3982),
            rot=(0.0, 0.5, 0.0, 0.866),
            convention="world",
        ),
        noise_pipeline={
            # H200 max-crop profile: cap optical depth before blur, then normalize.
            "sanitize_depth": DepthSanitizationCfg(invalid_depth=2.5, max_depth=2.5),
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 14, 14)),
            "gaussian_noise": RangeBasedGaussianNoiseCfg(min_value=0.1, max_value=2.5, noise_std=0.01),
            "scale_randomization": ScaleRandomizationNoiseCfg(
                apply_probability=0.5, scale_min=0.95, scale_max=1.05,
            ),
            "random_conv": RandomConvNoiseCfg(
                apply_probability=0.5, kernel_std=0.01, center_weight=1.0,
            ),
            "pixel_failures": PixelFailureNoiseCfg(
                apply_probability=0.5, dead_pixel_prob=2e-3, saturated_pixel_prob=2e-3,
                dead_value=0.0, saturated_value=2.5,
            ),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.1, 2.5), normalize=True, output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )


@configclass
class PUFOCommandsCfg:
    motion = mdp.UFOMotionCommandCfg(
        asset_name="robot", motion_dir=MOTION_DIR, motion_names=MOTION_NAMES
    )


@configclass
class PUFOActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        class_type=mdp.rewards.BodyImpactJointPositionAction,
        asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True
    )


@configclass
class PUFOObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        local_self = ObsTerm(func=mdp.pufo_local_self_obs)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class VisualCfg(ObsGroup):
        depth_image = ObsTerm(
            func=mdp.pufo_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "distance_to_image_plane_noised_history",
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "integer_delay": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeadingCfg(ObsGroup):
        heading = ObsTerm(func=mdp.pufo_heading_context)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityCommandCfg(ObsGroup):
        velocity_command = ObsTerm(func=mdp.pufo_velocity_command)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityTargetCfg(ObsGroup):
        velocity_target_yaw = ObsTerm(func=mdp.pufo_velocity_target_yaw)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TerrainPrivCfg(ObsGroup):
        terrain_priv = ObsTerm(func=mdp.pufo_terrain_priv)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TransitionKinematicsCfg(ObsGroup):
        kinematics = ObsTerm(func=mdp.pufo_transition_kinematics)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    visual: VisualCfg = VisualCfg()
    heading: HeadingCfg = HeadingCfg()
    velocity_command: VelocityCommandCfg = VelocityCommandCfg()
    velocity_target_yaw: VelocityTargetCfg = VelocityTargetCfg()
    terrain_priv: TerrainPrivCfg = TerrainPrivCfg()
    transition_kinematics: TransitionKinematicsCfg = TransitionKinematicsCfg()


@configclass
class PUFOEventCfg:
    resample_motion = EventTerm(
        func=mdp.events.resample_motion_for_reset,
        mode="reset",
        params={"command_name": "motion"},
    )
    reset_root_state = EventTerm(
        func=reset_pufo_from_motion,
        mode="reset",
        params={
            "command_name": "motion", "lie_down_probability": 0.0,
            "contact_clearance": 0.05, "penetration_scale": 1.25,
        },
    )
    randomize_camera_installation = EventTerm(
        func=mdp.events.randomize_camera_installation,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("camera"),
            "position_error": 0.02,
            "angle_error_rpy": (
                math.radians(3.0),
                math.radians(10.0),
                math.radians(3.0),
            ),
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.7),
            "dynamic_friction_range": (0.3, 1.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.events.push_robot_by_setting_base_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
        },
    )


@configclass
class PUFORewardsCfg:
    penalty_action_rate = RewTerm(
        func=mdp.rewards.penalty_action_rate,
        weight=-0.2,
    )
    # Keep the Parkour foot penalty; strengthen action changes and acceleration
    # while retaining the previous 10x joint-speed penalty.
    sound_suppression_acc_per_foot = RewTerm(
        func=mdp.rewards.sound_suppression_acc_per_foot,
        weight=-5e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.rewards.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.rewards.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    limits_dof_pos = RewTerm(
        func=mdp.rewards.limits_dof_pos,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    penalty_body_impact = RewTerm(
        func=mdp.rewards.pufo_penalty_body_impact,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["waist_yaw_link", ".*_shoulder_.*_link", ".*_hip_.*_link"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["waist_yaw_link", ".*_shoulder_.*_link", ".*_hip_.*_link"],
                preserve_order=True,
            ),
            "force_threshold_ratio": 0.05,
            "smooth_l1_beta": 0.25,
            "impact_velocity_scale": 1.0,
            "impact_velocity_cap": 2.0,
            "severity_cap": 4.0,
        },
    )
    penalty_slippage = RewTerm(
        func=mdp.rewards.pufo_penalty_slippage,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )
    penalty_ankle_roll = RewTerm(
        func=mdp.rewards.penalty_ankle_roll,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle_roll.*")},
    )
    feet_stumble = RewTerm(
        func=mdp.rewards.pufo_feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )


@configclass
class PUFOTerminationsCfg:
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)
    terrain_boundary = DoneTerm(
        func=mdp.terminations.terrain_out_of_bounds, time_out=True,
        params={"safety_margin": 0.5},
    )


@configclass
class PUFOCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.curriculums.terrain_levels_by_global_step,
        params={
            "start_step": 0,
            "interval": 19_200_000,
            "initial_max_level": 2,
            "level_increment": 2,
            "easy_fraction": 0.25,
        },
    )


@configclass
class PUFOPhysicsPresetCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=PUFOMJWarpSolverCfg(
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            iterations=600,
            tolerance=1.0e-2,
            ls_iterations=80,
            update_data_interval=1,
            # Allocation limits, not solver accuracy. The PUFO manager rejects overflows.
            njmax=4096,
            nconmax=512,
        ),
        # Refine contact integration without changing the 5 ms actuator tick.
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class PUFOPlaneSceneCfg(PUFOSceneCfg):
    """Flat scene used exclusively for motion tracking evaluation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


@configclass
class PUFOEnvCfg(ManagerBasedRLEnvCfg):
    scene: PUFOSceneCfg = PUFOSceneCfg()
    commands: PUFOCommandsCfg = PUFOCommandsCfg()
    actions: PUFOActionsCfg = PUFOActionsCfg()
    observations: PUFOObservationsCfg = PUFOObservationsCfg()
    events: PUFOEventCfg = PUFOEventCfg()
    rewards: PUFORewardsCfg = PUFORewardsCfg()
    terminations: PUFOTerminationsCfg = PUFOTerminationsCfg()
    curriculum: PUFOCurriculumCfg = PUFOCurriculumCfg()

    def __post_init__(self):
        self.compute_final_obs = True
        self.scene.env_spacing = PUFO_TILE_SIZE[0]
        self.sim.physics_material = self.scene.terrain.physics_material
        self.decimation = 4
        self.scene.contact_forces.history_length = self.decimation
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PUFOPhysicsPresetCfg().default
        self.sim.use_newton_actuators = False


@configclass
class RP1PUFOEnvCfg(PUFOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(PR1_LINKS))


@configclass
class RP1PUFOEvalEnvCfg(PUFOEnvCfg):
    """PUFO tracking evaluation with the same robot and sensors on a plane."""

    scene: PUFOPlaneSceneCfg = PUFOPlaneSceneCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(PR1_LINKS))
        self.events.reset_root_state.params["lie_down_probability"] = 0.0
