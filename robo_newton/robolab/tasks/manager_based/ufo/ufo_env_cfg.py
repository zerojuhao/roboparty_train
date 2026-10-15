"""Standalone UFO environment configuration (no inheritance from other tasks).

Observation groups (consumed by the FbCpr runner, which assembles the UFO
dict-obs `state / last_action / history_actor / privileged_state`):
  - policy:  base_ang_vel(3) + projected_gravity(3) + joint_pos(24) + joint_vel(24) + last_action(24) = 78
  - critic:  local_self_obs (373), computed with the vendored UFO observation math
"""

from __future__ import annotations

import os

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import SceneEntityCfg

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from robolab.assets import ISAAC_DATA_DIR
from robolab.assets.robots.roboparty import RP1_UFO_CFG

from . import mdp

# LAFAN motions shared with the RP1 WBC tracking task.
MOTION_MANIFEST = os.environ.get(
    "UFO_MOTION_MANIFEST",
    os.path.join(ISAAC_DATA_DIR, "motions", "rp1_lafan", "manifest.json"),
)


@configclass
class UFOSceneCfg(InteractiveSceneCfg):
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
    robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=mdp.rewards.UNDESIRED_CONTACT_HISTORY_WINDOW,
        track_air_time=False,
    )


@configclass
class UFOCommandsCfg:
    motion = mdp.UFOMotionCommandCfg(
        asset_name="robot",
        manifest_path=MOTION_MANIFEST,
    )


@configclass
class UFOActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True)


@configclass
class UFOObservationsCfg:
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
        local_self = ObsTerm(func=mdp.local_self_obs)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UFOEventCfg:
    resample_motion = EventTerm(
        func=mdp.events.resample_motion_for_reset,
        mode="reset",
        params={"command_name": "motion"},
    )
    reset_root_state = EventTerm(
        func=mdp.events.reset_root_state_from_motion,
        mode="reset",
        params={"command_name": "motion"},
    )
    reset_joint_state = EventTerm(
        func=mdp.events.reset_joint_state_from_motion,
        mode="reset",
        params={"command_name": "motion", "joint_pos_range": (0.0, 0.0)},
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
            "mass_distribution_params": (0.9, 1.1),
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
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class UFORewardsCfg:
    penalty_action_rate = RewTerm(
        func=mdp.rewards.penalty_action_rate,
        weight=-0.1,
    )
    limits_dof_pos = RewTerm(
        func=mdp.rewards.limits_dof_pos,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    penalty_slippage = RewTerm(
        func=mdp.rewards.penalty_slippage,
        weight=-2.0,
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
        func=mdp.rewards.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )

@configclass
class UFOTerminationsCfg:
    time_out = DoneTerm(func=mdp.terminations.time_out, time_out=True)


@configclass
class PhysicsPresetCfg(PresetCfg):
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
class UFOEnvCfg(ManagerBasedRLEnvCfg):
    scene: UFOSceneCfg = UFOSceneCfg()
    commands: UFOCommandsCfg = UFOCommandsCfg()
    actions: UFOActionsCfg = UFOActionsCfg()
    observations: UFOObservationsCfg = UFOObservationsCfg()
    events: UFOEventCfg = UFOEventCfg()
    rewards: UFORewardsCfg = UFORewardsCfg()
    terminations: UFOTerminationsCfg = UFOTerminationsCfg()

    def __post_init__(self):
        self.compute_final_obs = True
        self.scene.env_spacing = 5.0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsPresetCfg().default
        # The custom RoboParty actuator config is not Newton-native; use the
        # host-executed actuator path while keeping Newton as the physics solver.
        self.sim.use_newton_actuators = False
