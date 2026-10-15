"""Check the AMP elbow range penalty without physics or training."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.rewards import joint_position_range_l2
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpEnvCfg, RP1AmpPathEnvCfg


LOWER = (math.radians(-30.0), math.radians(-45.0))
UPPER = (math.radians(45.0), math.radians(30.0))


def make_env(angles, wrapped=False, commands=None):
    positions = torch.deg2rad(torch.tensor(angles, dtype=torch.float32))
    value = SimpleNamespace(torch=positions) if wrapped else positions
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=value))},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(commands, dtype=torch.float32)),
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_inside_and_boundary_angles_have_zero_penalty(wrapped):
    env = make_env([[-30, -45], [45, 30], [0, 0], [20, -20]], wrapped)
    torch.testing.assert_close(joint_position_range_l2(env, LOWER, UPPER), torch.zeros(4))


@pytest.mark.parametrize("wrapped", [False, True])
def test_penalty_is_squared_excess_for_each_joint_without_cancellation(wrapped):
    env = make_env([[-40, 40], [55, -55], [-50, 50]], wrapped)
    actual = joint_position_range_l2(env, LOWER, UPPER)
    unit = math.radians(10.0) ** 2
    torch.testing.assert_close(actual, torch.tensor([2 * unit, 2 * unit, 8 * unit]))


def test_selected_joint_order_controls_bounds_and_ignores_other_joints():
    env = make_env([[55, 1000, -40]])
    cfg = SceneEntityCfg("robot", joint_ids=[2, 0])
    actual = joint_position_range_l2(env, LOWER, UPPER, cfg)
    torch.testing.assert_close(actual, torch.tensor([math.radians(10) ** 2 + math.radians(25) ** 2]))


def test_penalty_is_invariant_to_left_right_mirroring():
    angles = torch.tensor([[-40, 20], [60, -65], [-69, 69]], dtype=torch.float32)
    mirrored = -angles.flip(dims=[1])
    actual = joint_position_range_l2(make_env(angles.tolist()), LOWER, UPPER)
    torch.testing.assert_close(
        actual, joint_position_range_l2(make_env(mirrored.tolist()), LOWER, UPPER),
    )


@pytest.mark.parametrize("lower,upper", [
    ((0,), (1, 1)), ((0, 0), (1,)), ((0, 0, 0), (1, 1, 1)),
    ((0, 0), (0, 1)), ((2, 0), (1, 1)),
    ((float("nan"), 0), (1, 1)), ((0, 0), (float("inf"), 1)),
])
def test_invalid_bounds_are_rejected(lower, upper):
    with pytest.raises(ValueError, match="bounds"):
        joint_position_range_l2(make_env([[0, 0]]), lower, upper)


@pytest.mark.parametrize("config_class", [RP1AmpEnvCfg, RP1AmpPathEnvCfg])
def test_rp1_enables_requested_absolute_elbow_bounds(config_class):
    term = config_class().rewards.elbow_position_range_l2
    assert term.func is joint_position_range_l2
    assert term.weight == -0.05
    assert term.params["angle_scale"] == math.radians(5.0)
    assert term.params["lower_bounds"] == LOWER
    assert term.params["upper_bounds"] == UPPER
    assert term.params["asset_cfg"].joint_names == ["left_elbow_joint", "right_elbow_joint"]
    assert term.params["asset_cfg"].preserve_order is True
    assert term.params["command_name"] == "single_axis_velocity"
    assert term.params["min_linear_command"] == 1.5


def test_absolute_bounds_penalize_the_existing_default_standing_elbows():
    actual = joint_position_range_l2(make_env([[-math.degrees(1.2), math.degrees(1.2)]]), LOWER, UPPER)
    torch.testing.assert_close(actual, torch.tensor([2 * (1.2 - math.radians(30)) ** 2]))


@pytest.mark.parametrize("wrapped", [False, True])
def test_speed_gate_uses_strict_commanded_planar_speed_not_yaw(wrapped):
    commands = [[0, 0, 0], [1.49, 0, 0], [1.5, 0, 0], [1.5001, 0, 0],
                [-2, 0, 0], [1, 1.2, 0], [0, 0, 2], [0, 1.5, 0]]
    env = make_env([[-40, 40]] * len(commands), wrapped, commands)
    actual = joint_position_range_l2(env, LOWER, UPPER, command_name="velocity", min_linear_command=1.5)
    unit = 2 * math.radians(10) ** 2
    torch.testing.assert_close(actual, torch.tensor([0, 0, 0, unit, unit, unit, 0, 0]))


@pytest.mark.parametrize("commanded_speed,measured_speed", [(0, 4), (4, 0)])
def test_configured_gate_ignores_actual_speed_and_disables_default_pose_penalty_at_stop(
    commanded_speed, measured_speed,
):
    term = RP1AmpPathEnvCfg().rewards.elbow_position_range_l2
    env = make_env([[-math.degrees(1.2), math.degrees(1.2)]], commands=[[commanded_speed, 0, 0]])
    env.scene["robot"].data.root_lin_vel_b = torch.tensor([[measured_speed, 0, 0]])
    params = dict(term.params)
    params["asset_cfg"] = SceneEntityCfg("robot", joint_ids=[0, 1])
    actual = term.func(env, **params)
    expected = 0 if commanded_speed <= 1.5 else 2 * ((1.2 - math.radians(30)) / math.radians(5)) ** 2
    torch.testing.assert_close(actual, torch.tensor([expected], dtype=torch.float32))


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_invalid_command_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="min_linear_command"):
        joint_position_range_l2(
            make_env([[0, 0]], commands=[[4, 0, 0]]), LOWER, UPPER,
            command_name="velocity", min_linear_command=threshold,
        )


def test_five_degree_scale_makes_ten_degree_excess_four_units_per_joint():
    penalty = joint_position_range_l2(make_env([[-40, 40]]), LOWER, UPPER, angle_scale=math.radians(5))
    torch.testing.assert_close(penalty, torch.tensor([8.0]))


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_angle_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="angle_scale"):
        joint_position_range_l2(make_env([[0, 0]]), LOWER, UPPER, angle_scale=scale)
