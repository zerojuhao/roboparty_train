"""Newton PUFO environment with an isolated visual observation pipeline."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robolab.assets.robots.roboparty import RP1_UFO_CFG
from robolab.sensors import NoisyGroupedRayCasterCameraCfg
from robolab.tasks.manager_based.ufo.ufo_env_cfg import UFOEnvCfg, UFOSceneCfg

from . import mdp


@configclass
class PUFOSceneCfg(UFOSceneCfg):
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
        local_self = ObsTerm(func=mdp.local_self_obs)

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class VisualCfg(ObsGroup):
        depth_image = ObsTerm(func=mdp.depth_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HeadingCfg(ObsGroup):
        heading = ObsTerm(func=mdp.heading_context)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityCommandCfg(ObsGroup):
        velocity_command = ObsTerm(func=mdp.velocity_command)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VelocityTargetCfg(ObsGroup):
        velocity_target_yaw = ObsTerm(func=mdp.velocity_target_yaw)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    visual: VisualCfg = VisualCfg()
    heading: HeadingCfg = HeadingCfg()
    velocity_command: VelocityCommandCfg = VelocityCommandCfg()
    velocity_target_yaw: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class PUFOEnvCfg(UFOEnvCfg):
    scene: PUFOSceneCfg = PUFOSceneCfg()
    observations: PUFOObservationsCfg = PUFOObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 5.0


@configclass
class RP1PUFOEnvCfg(PUFOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RP1_UFO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
