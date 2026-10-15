"""Check the deferred evaluation/resume command contract without starting jobs."""

from pathlib import Path
import os

from finish_amp_eval_resume import build_commands, training_process_alive


def commands():
    run = Path("/tmp/source_run")
    reference = {
        "mujoco_xml": "/tmp/robot.xml", "ramp_seconds_override": 2.0,
        "deceleration_seconds": 2.0, "standing_seconds": 2.0, "duration_s": 11.5,
        "video_fps": 25, "resolution": [1920, 1080], "initial_root_height_m": 0.85,
    }
    return build_commands(run, run / "model_24499.pt", reference,
                          Path("/tmp/evaluation.mp4"), "next_run", 2500)


def test_evaluation_matches_saved_cruise_and_stop_conditions():
    evaluation, _ = commands()
    expected = {
        "--checkpoint": "/tmp/source_run/model_24499.pt", "--max-speed": "4",
        "--ramp-seconds": "2.0", "--deceleration-seconds": "2.0",
        "--standing-seconds": "2.0", "--duration": "11.5", "--fps": "25",
        "--width": "1920", "--height": "1080", "--initial-height": "0.85",
    }
    for key, value in expected.items():
        assert evaluation[evaluation.index(key) + 1] == value
    assert 11.5 - 0.5 - 2.0 - 2.0 - 2.0 == 5.0


def test_resume_changes_only_requested_weights_and_preserves_safety_terms():
    _, training = commands()
    assert training[training.index("--load_run") + 1] == "source_run"
    assert training[training.index("--checkpoint") + 1] == "model_24499.pt"
    assert training[training.index("--max_iterations") + 1] == "2500"
    assert training[training.index("--resume_learning_rate") + 1] == "5e-5"
    for override in (
        "agent.algorithm.schedule=fixed",
        "env.rewards.track_lin_vel_xy_exp.weight=1.2",
        "env.rewards.lateral_command_offset.weight=-0.2",
        "env.rewards.lateral_command_offset.params.std=0.5",
        "env.rewards.sound_suppression.weight=-5e-4",
        "env.rewards.feet_close_y_gauss.params.threshold=0.18",
        "env.rewards.knee_close_y_gauss.params.threshold=0.18",
    ):
        assert override in training


def test_process_check_does_not_match_an_unrelated_or_missing_pid():
    assert not training_process_alive(os.getpid(), "next_run")
    assert not training_process_alive(99999999, "next_run")
