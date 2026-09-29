"""Standalone RP1 PUFO environment with visual and terrain context."""

from __future__ import annotations

import math
import os

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from robolab.assets import UFO_DATA_DIR
from robolab.assets.robots.roboparty import RP1_UFO_CFG
from robolab.sensors import NoisyGroupedRayCasterCameraCfg

from . import mdp
from .pufo_terrain_cfg import PUFO_TERRAIN_CFG


MOTION_MANIFEST = os.environ.get(
    "UFO_MOTION_MANIFEST",
    os.path.join(UFO_DATA_DIR, "motions", "rp1_lafan", "manifest.json"),
)


@configclass
class PUFOSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PUFO_TERRAIN_CFG,
        max_init_terrain_level=0,
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=mdp.rewards.UNDESIRED_CONTACT_HISTORY_WINDOW,
        track_air_time=False,
    )
    terrain_priv = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/waist_roll_link/waist_yaw_link",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(2.0, 1.2),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 2.0)),
        max_distance=3.0,
        update_period=0.02,
    )
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/waist_roll_link/waist_yaw_link",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=32,
            height=18,
        ),
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.09175, 0.011, 0.3982),
            rot=(0.0, 0.5, 0.0, 0.866),
            convention="world",
        ),
        data_histories={"distance_to_image_plane_noised": 8},
    )


@configclass
class PUFOCommandsCfg:
    motion = mdp.UFOMotionCommandCfg(asset_name="robot", manifest_path=MOTION_MANIFEST)


@configclass
class PUFOActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True
    )


@configclass
class PUFOObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
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
        depth_image = ObsTerm(func=mdp.pufo_depth_image)

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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    visual: VisualCfg = VisualCfg()
    heading: HeadingCfg = HeadingCfg()
    velocity_command: VelocityCommandCfg = VelocityCommandCfg()
    velocity_target_yaw: VelocityTargetCfg = VelocityTargetCfg()
    terrain_priv: TerrainPrivCfg = TerrainPrivCfg()


@configclass
class PUFOEventCfg:
    reset_root_state = EventTerm(
        func=mdp.events.reset_root_state_from_motion,
        mode="reset",
        params={"command_name": "motion"},
    )
    reset_joint_state = EventTerm(
        func=mdp.events.reset_joint_state_from_motion,
        mode="reset",
        params={"command_name": "motion"},
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.04, 0.04)},
        },
    )
    randomize_default_joint_pos = EventTerm(
        func=mdp.events.randomize_default_joint_pos,
        mode="startup",
        params={"offset_range": (-0.01, 0.01)},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.events.push_robot_by_setting_base_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


@configclass
class PUFORewardsCfg:
    zero = RewTerm(func=mdp.pufo_rewards.zero_reward, weight=0.0)


@configclass
class PUFOTerminationsCfg:
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)


@configclass
class PUFOPhysicsPresetCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            update_data_interval=1,
            njmax=4096,
            nconmax=512,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class PUFOEnvCfg(ManagerBasedRLEnvCfg):
    scene: PUFOSceneCfg = PUFOSceneCfg()
    commands: PUFOCommandsCfg = PUFOCommandsCfg()
    actions: PUFOActionsCfg = PUFOActionsCfg()
    observations: PUFOObservationsCfg = PUFOObservationsCfg()
    events: PUFOEventCfg = PUFOEventCfg()
    rewards: PUFORewardsCfg = PUFORewardsCfg()
    terminations: PUFOTerminationsCfg = PUFOTerminationsCfg()

    def __post_init__(self):
        self.scene.env_spacing = 5.0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PUFOPhysicsPresetCfg()
        self.sim.use_newton_actuators = False


@configclass
class RP1PUFOEnvCfg(PUFOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
