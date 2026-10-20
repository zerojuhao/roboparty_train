"""Shared motion-data selection used by UFO and PUFO.

Filtering changes the training distribution without altering the source
datasets on disk:

* LAFAN subset  → ``data/motions/rp1_lafan/``
* RP1 locomotion → ``data/motions/rp1/``
* KEEP5 gestures → ``data/motions/rp1_action/``

``MOTION_DIR`` uses the three dataset directories under ``robolab/data/motions``.
"""

from pathlib import Path

MOTION_ROOT = Path(__file__).resolve().parents[3] / "data" / "motions"
MOTION_DIR = (
    str(MOTION_ROOT / "rp1_lafan"),
    str(MOTION_ROOT / "rp1"),
    str(MOTION_ROOT / "rp1_action"),
)

# 40-clip LAFAN subset (rp1_lafan).
LAFAN_MOTION_NAMES = (
    "dance1_subject1",
    "dance1_subject2",
    "dance1_subject3",
    "dance2_subject1",
    "dance2_subject2",
    "dance2_subject3",
    "dance2_subject4",
    "dance2_subject5",
    "fallAndGetUp1_subject1",
    "fallAndGetUp1_subject4",
    "fallAndGetUp1_subject5",
    "fallAndGetUp2_subject2",
    "fallAndGetUp2_subject3",
    "fallAndGetUp3_subject1",
    "fight1_subject2",
    "fight1_subject3",
    "fight1_subject5",
    "fightAndSports1_subject1",
    "fightAndSports1_subject4",
    "jumps1_subject1",
    "jumps1_subject2",
    "jumps1_subject5",
    "run1_subject2",
    "run1_subject5",
    "run2_subject1",
    "run2_subject4",
    "sprint1_subject2",
    "sprint1_subject4",
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
    "walk4_subject1",
)

# All active RP1 locomotion clips (rp1/).
LOCOMOTION_MOTION_NAMES = (
    "114_09",
    "36_01",
    "36_11",
    "A1-_Stand_stageii",
    "B10_-__Walk_turn_left_45_stageii",
    "B13_-__Walk_turn_right_90_stageii",
    "B14_-__Walk_turn_right_45_t2_stageii",
    "B15_-__Walk_turn_around_stageii",
    "B4_-_Stand_to_Walk_backwards_stageii",
    "B9_-__Walk_turn_left_90_stageii",
    "idle_loop_001__A046",
    "jog_arc_ccw_loop_003__A041",
    "jog_arc_cw_loop_004__A045",
    "jog_forward_A021",
    "jog_forward_A045",
    "move_back",
    "move_l",
    "move_r",
    "run_loop_180_R_003__A326",
    "run_start_180_R_001__A345",
    "run_stop_180_R_001__A345",
    "stairs_climbing_down_loop_R_001__A300",
    "stairs_climbing_down_loop_R_102__A301_M",
    "stairs_climbing_down_stop_R_001__A300",
    "stairs_climbing_down_stop_R_103__A301",
    "stairs_climbing_jog_down_loop_R_001__A300",
    "stairs_climbing_jog_down_stop_R_001__A300",
    "stairs_climbing_jog_up_start_R_001__A300",
    "stairs_climbing_up_start_R_001__A300",
    "stand_still",
    "turn_l",
    "turn_r",
    "walk_arc_ccw_loop_L_002__A041",
    "walk_arc_cw_loop_001__A048",
    "walk_ff_loop_180_R_001__A048",
)

# KEEP5 gestures (rp1_action), Desktop four-key NPZ stems.
ACTION_MOTION_NAMES = (
    "wave_R",
    "casual_greeting_R",
    "warm_welcome_R",
    "salute",
    "wave_two_hands_R",
)

# Combined selection for RP1 UFO / PUFO training (uniform clip sampling).
MOTION_NAMES = LAFAN_MOTION_NAMES + LOCOMOTION_MOTION_NAMES + ACTION_MOTION_NAMES
