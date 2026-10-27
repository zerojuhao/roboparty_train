import copy
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.utils.data_paths import motion_data_dir
from robolab.assets.robots.roboparty import PR1_LINKS, RP1_24DOF_CFG
from robolab.sensors import Grid3dPointsGeneratorCfg, get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import (
    ROUGH_TERRAINS_CFG,
    ParkourEnvCfg,
)

# The motion loader selects and reorders NPZ key-body tracks by these names.
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_link",
    "right_wrist_link",
]

RP1_24DOF_CFG.init_state.pos = (0.0, 0.0, 0.85)
AMP_NUM_STEPS = 3

# Measured ankle-roll origin to sole contact plane. The support tolerance covers
# the 3 cm Perlin amplitude while remaining below the 5 cm minimum stair.
RP1_SOLE_HEIGHT = 0.045

# Shared by feet_volume_points (penetration) and foot height-scan (critic / support).
# Tune the shoe box here; scan center and size follow x/y bounds.
FEET_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.09,
    x_max=0.14,
    x_num=24,
    y_min=-0.04,
    y_max=0.04,
    y_num=9,
    z_min=-0.05,
    z_max=-0.02,
    z_num=4,
)

FOOT_HEIGHT_SCAN_RESOLUTION = 0.01
FOOT_HEIGHT_SCAN_RAY_START_HEIGHT = 20.0
FOOT_HEIGHT_SCAN_CENTER = (
    0.5 * (FEET_VOLUME_POINTS_GRID.x_min + FEET_VOLUME_POINTS_GRID.x_max),
    0.5 * (FEET_VOLUME_POINTS_GRID.y_min + FEET_VOLUME_POINTS_GRID.y_max),
)
FOOT_HEIGHT_SCAN_SIZE = (
    FEET_VOLUME_POINTS_GRID.x_max - FEET_VOLUME_POINTS_GRID.x_min,
    0.04,
)

KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.02,
    x_max=0.09,
    x_num=12,
    y_min=-0.04,
    y_max=0.04,
    y_num=9,
    z_min=-0.3,
    z_max=0.0,
    z_num=31,
)

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class RP1ParkourEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.use_newton_actuators = False
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = RP1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/(?!.*left_ankle_roll_link[/]).*left_ankle_roll_link"
        self.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/(?!.*right_ankle_roll_link[/]).*right_ankle_roll_link"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
        self.scene.feet_volume_points.points_generator = FEET_VOLUME_POINTS_GRID
        self.scene.knee_volume_points.points_generator = KNEE_VOLUME_POINTS_GRID
        self.rewards.locomotion.feet_at_plane.params["height_offset"] = RP1_SOLE_HEIGHT
        for scanner_cfg in (
            self.scene.left_height_scanner,
            self.scene.right_height_scanner,
        ):
            scanner_cfg.offset.pos = (
                *FOOT_HEIGHT_SCAN_CENTER,
                FOOT_HEIGHT_SCAN_RAY_START_HEIGHT,
            )
            scanner_cfg.pattern_cfg.size = FOOT_HEIGHT_SCAN_SIZE
            scanner_cfg.pattern_cfg.resolution = FOOT_HEIGHT_SCAN_RESOLUTION
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link/waist_roll_link/waist_yaw_link"
        self.scene.camera.offset.pos = (0.09175, 0.011, 0.3982)
        self.scene.camera.offset.rot = (0.0, 0.5, 0.0, 0.866)
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(PR1_LINKS))
        self.motion_data.motion_dataset.motion_data_dir = motion_data_dir("rp1")
        self.motion_data.motion_dataset.key_body_names = KEY_BODY_NAMES
        self.motion_data.motion_dataset.motion_data_weights = {
            # CMU
            # "36_01": 1,
            # "36_11": 1,
            "114_09": 1,
            # ACCAD
            "A1-_Stand_stageii": 1,
            "B4_-_Stand_to_Walk_backwards_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            # GVHMR
            "move_back": 1,
            "move_l": 1,
            "move_r": 1,
            "turn_l": 1,
            "turn_r": 1,
            # SEED
            # "idle_loop_001__A046": 1,
            "stairs_climbing_down_loop_R_102__A301_M": 1,
            "stairs_climbing_down_stop_R_103__A301": 1,
            "stairs_climbing_up_start_R_001__A300": 1,
            "stairs_climbing_down_loop_R_001__A300": 1,
            "stairs_climbing_down_stop_R_001__A300": 1,
            "stairs_climbing_jog_down_loop_R_001__A300": 1,
            "stairs_climbing_jog_down_stop_R_001__A300": 1,
            "stairs_climbing_jog_up_start_R_001__A300": 1,
            # "walk_arc_cw_loop_001__A048_M": 1,
            # "walk_arc_cw_loop_001__A048": 1,
            # "walk_ff_loop_180_R_001__A048": 1,
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }

        self.rewards.locomotion.leg_yaw_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint"]
        )
        self.rewards.locomotion.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.rewards.locomotion.pelvis_ang_vel_xy_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["waist_yaw_link"])

        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base_link", "waist_yaw_link"])


@configclass
class RP1ParkourEnvCfg_PLAY(RP1ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.one_col_per_subterrain = False
        self.scene.terrain.debug_vis = True
        self.scene.feet_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
