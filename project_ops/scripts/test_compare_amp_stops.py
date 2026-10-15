"""Check stopping-window and failure accounting in the checkpoint comparison."""

import numpy as np
import pytest

from compare_amp_stops import (MJ_JOINT_NAMES, acceleration_metrics, braking_metrics,
                               elbow_metrics, step_command, stop_metrics, summarize, turn_metrics, yaw_metrics)


def telemetry():
    count = 875
    velocity = np.zeros((count, 3))
    velocity[:390, 0] = 4.0
    velocity[390:, 0] = 0.05
    positions = np.zeros((count, 3))
    positions[:, 0] = np.cumsum(velocity[:, 0]) / 50
    positions[:, 2] = 0.8
    return {
        "root_position_w": positions, "body_velocity": velocity,
        "joint_position": np.zeros((count, len(MJ_JOINT_NAMES))),
        "joint_velocity": np.zeros((count, len(MJ_JOINT_NAMES))),
        "joint_target": np.zeros((count, len(MJ_JOINT_NAMES))),
        "joint_torque_peak_nm": np.zeros((count, len(MJ_JOINT_NAMES))),
        "braking_power_peak_w": np.zeros(count), "braking_energy_j": np.zeros(count),
        "heading_rad": np.zeros(count), "foot_normal_force": np.full((count, 2), 10.0),
        "body_angular_velocity": np.zeros((count, 3)),
        "foot_contact_slip_speed": np.full((count, 2), 0.2),
        "foot_heading_relative_base_rad": np.zeros((count, 2)),
    }


def test_stopping_requires_half_second_below_threshold():
    data = telemetry()
    # One early sample below the threshold is not a stable stop.
    data["body_velocity"][376, 0] = 0.0
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    assert result["stopping"]["0.1"]["seconds"] == pytest.approx(0.3)
    assert result["final_pose"]["l1_rad"] == 0
    assert result["pose_at_five_seconds"]["l1_rad"] == 0
    assert result["final_pose"]["standing_gate_fraction"] == 1
    assert result["first_second_slip"]["mean_loaded_foot_speed_mps"] == pytest.approx(0.2)


def test_braking_metrics_include_first_step_and_exclude_outside_window():
    data = telemetry()
    data["joint_target"][375:, 0] = 0.5
    data["braking_power_peak_w"][375] = 200
    data["braking_power_peak_w"][525] = 1000
    data["braking_energy_j"][375:525] = 1
    data["joint_torque_peak_nm"][380, 0] = 20
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    assert result["braking"]["target_increment_max_rad"] == 0.5
    assert result["braking"]["mechanical_braking_power_peak_w"] == 200
    assert result["braking"]["full_stop_power_peak_w"] == 1000
    assert result["braking"]["mechanical_braking_energy_j"] == 150
    assert result["braking"]["joint_torque_peak_nm"] == 20
    assert summarize([result])["mechanical_braking_power_peak_w"]["mean"] == 200
    assert summarize([result])["final_standing_gate_fraction"]["mean"] == 1


def test_braking_metrics_exclude_post_fall_samples():
    data = telemetry()
    data["braking_power_peak_w"][386] = 1000
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, 7.7)
    assert result["braking"]["mechanical_braking_power_peak_w"] == 0
    assert braking_metrics(data, 375, np.zeros(875, dtype=bool)) is None


def test_torque_jump_metric_includes_only_valid_braking_physics_peaks():
    data = telemetry()
    data["joint_torque_delta_ratio_peak"] = np.zeros(875)
    data["joint_torque_delta_ratio_peak"][375] = 1.2
    data["joint_torque_delta_ratio_peak"][386] = 2.0
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, 7.7)
    assert result["braking"]["torque_delta_ratio_peak"] == 1.2
    assert summarize([result])["torque_delta_ratio_peak"]["mean"] == 1.2


def test_fall_is_not_reported_as_a_successful_stand_or_stop():
    result = stop_metrics(telemetry(), 375, np.zeros(len(MJ_JOINT_NAMES)), 100, 7.7)
    assert result["final_pose"] is None
    assert result["pose_at_five_seconds"] is None
    assert result["stopping"]["0.1"]["seconds"] is None
    summary = summarize([result])
    assert summary["falls"] == 1
    assert summary["standing_pose_l1_rad"] is None
    assert summary["failed_stops_below_0_1"] == 1


def test_summary_preserves_worst_case_not_only_average():
    result = stop_metrics(telemetry(), 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    other = {**result, "max_abs_world_y_m": 2.0}
    summary = summarize([result, other])
    assert summary["max_abs_world_y_m"] == {"mean": 1.0, "min": 0.0, "max": 2.0}


def test_segment_lateral_uses_start_pose_not_world_y():
    data = telemetry()
    data["root_position_w"][:, 1] = data["root_position_w"][:, 0] + 10
    data["root_position_w"][:, 0] = 7
    data["heading_rad"][:] = np.pi / 2
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    assert result["max_abs_world_y_m"] > 10
    assert result["cruise_segment_max_abs_lateral_m"] == pytest.approx(0)
    assert result["cruise_segment_end_heading_deg"] == pytest.approx(0)


def test_five_second_pose_excludes_later_samples():
    data = telemetry()
    data["joint_position"][625:] = 1
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    assert result["pose_at_five_seconds"]["l1_rad"] == 0
    assert result["final_pose"]["l1_rad"] == len(MJ_JOINT_NAMES)


def test_yaw_metrics_selects_hips_and_distinguishes_outward_from_inward():
    data = telemetry()
    data["joint_position"][:, MJ_JOINT_NAMES.index("left_hip_yaw_joint")] = 0.1
    data["joint_position"][:, MJ_JOINT_NAMES.index("right_hip_yaw_joint")] = -0.2
    data["foot_heading_relative_base_rad"][:] = [0.1, -0.2]
    result = yaw_metrics(data, slice(-50, None), np.zeros(len(MJ_JOINT_NAMES)))
    assert result["left_hip_yaw_abs_error_deg"] == pytest.approx(np.rad2deg(0.1))
    assert result["right_foot_outward_deg"] == pytest.approx(np.rad2deg(0.2))
    data["foot_heading_relative_base_rad"] *= -1
    result = yaw_metrics(data, slice(-50, None), np.zeros(len(MJ_JOINT_NAMES)))
    assert result["left_foot_outward_deg"] == 0
    assert result["right_foot_outward_deg"] == 0


def test_standing_yaw_excludes_frames_above_the_speed_gate():
    data = telemetry()
    data["body_velocity"][-1, 0] = 0.2
    data["foot_heading_relative_base_rad"][-1] = [1, -1]
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    assert result["final_pose"]["left_foot_outward_deg"] > 0
    assert result["final_pose"]["stopped_yaw"]["left_foot_outward_deg"] == 0


def test_cruise_elbows_use_actual_angles_and_only_the_cruise_window():
    data = telemetry()
    left = MJ_JOINT_NAMES.index("left_elbow_joint")
    right = MJ_JOINT_NAMES.index("right_elbow_joint")
    data["joint_position"][125:375, left] = np.deg2rad(-20)
    data["joint_position"][125:375, right] = np.deg2rad(25)
    data["joint_position"][200, left] = np.deg2rad(40)
    data["joint_position"][400:, left] = np.deg2rad(-69)
    data["joint_velocity"][125:375, left] = -3
    result = stop_metrics(data, 375, np.zeros(len(MJ_JOINT_NAMES)), 100, None)
    elbows = result["cruise_elbows"]
    assert elbows["left_elbow_min_deg"] == pytest.approx(-20)
    assert elbows["left_elbow_max_deg"] == pytest.approx(40)
    assert elbows["right_elbow_max_deg"] == pytest.approx(25)
    assert elbows["left_elbow_speed_p95_rps"] == pytest.approx(3)
    assert summarize([result])["cruise_left_elbow_min_deg"]["mean"] == pytest.approx(-20)


def test_empty_elbow_window_does_not_report_successful_motion():
    assert elbow_metrics(telemetry(), np.zeros(875, dtype=bool)) is None


@pytest.mark.parametrize("rate", [-1.0, 1.0])
def test_turn_metrics_unwraps_heading_and_tracks_both_directions(rate):
    data = telemetry()
    data["body_angular_velocity"][:, 2] = rate
    heading = np.arange(len(data["heading_rad"])) / 50 * rate
    data["heading_rad"] = np.arctan2(np.sin(heading), np.cos(heading))
    result = turn_metrics(data, rate, np.zeros(len(MJ_JOINT_NAMES)), None)
    assert result["mean_yaw_rate_rps"] == rate
    assert result["yaw_rate_mae_rps"] == 0
    assert result["turned_angle_deg"] == pytest.approx(np.rad2deg(heading[-1] - heading[24]))


@pytest.mark.parametrize("fallen_at", [0.0, 0.9])
def test_early_turn_failure_has_no_successful_tracking_measurement(fallen_at):
    result = turn_metrics(telemetry(), 1, np.zeros(len(MJ_JOINT_NAMES)), fallen_at)
    assert result["fallen_at_s"] == fallen_at
    assert result["mean_yaw_rate_rps"] is None
    assert result["hip_yaw_and_feet"] is None


def test_step_protocol_holds_four_for_exactly_five_seconds_without_ramps():
    commands = [step_command(i, 25, 275) for i in range(525)]
    assert commands[:25] == [0] * 25
    assert commands[25:275] == [4] * 250
    assert commands[275:] == [0] * 250


def test_step_cruise_metrics_use_custom_start_and_falls_are_counted():
    data = telemetry()
    data["root_position_w"][:25, 1] = 10
    data["root_position_w"][24:, 1] = 0
    data["root_position_w"][275:, 1] = 100
    result = stop_metrics(data, 275, np.zeros(len(MJ_JOINT_NAMES)), 100, 2.0, cruise_start_step=25)
    result["acceleration"] = acceleration_metrics(data, 25, 275, 2.0)
    assert result["cruise_segment_max_abs_lateral_m"] == 0
    assert result["acceleration"]["fallen_before_stop"] is True
    assert summarize([result])["falls_before_stop"] == 1
    assert result["stopping"]["0.1"]["seconds"] is None


def test_acceleration_requires_sustained_speed_and_excludes_post_fall_samples():
    data = telemetry()
    data["body_velocity"][:75, 0] = 0
    data["body_velocity"][26, 0] = 4
    result = acceleration_metrics(data, 25, 275, None)
    assert result["time_to_sustained_3_6_to_4_4_mps_s"] == 1.0
    assert acceleration_metrics(data, 25, 275, 1.5)["time_to_sustained_3_6_to_4_4_mps_s"] is None


def test_elbow_metrics_report_extent_and_fraction_of_range_violations():
    data = telemetry()
    data["joint_position"][:10, MJ_JOINT_NAMES.index("left_elbow_joint")] = np.deg2rad(-40)
    result = elbow_metrics(data, slice(0, 100))
    assert result["left_elbow_outside_fraction"] == pytest.approx(0.1)
    assert result["left_elbow_max_excess_deg"] == pytest.approx(10)
    assert result["right_elbow_max_excess_deg"] == 0
