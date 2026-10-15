"""Check the AMP standing pose penalty without starting physics or training."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.rewards import joint_deviation_l1, stand_still_joint_deviation_l1
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg


def make_env(commands, linear_velocity=None, angular_velocity=None, wrapped=False):
    commands = torch.tensor(commands, dtype=torch.float32)
    count = len(commands)
    positions = torch.tensor([[0.4, -0.6, 10.0]]).repeat(count, 1)
    defaults = torch.tensor([[0.1, -0.1, 0.0]]).repeat(count, 1)

    def array(value):
        return SimpleNamespace(torch=value) if wrapped else value

    data = SimpleNamespace(
        joint_pos=array(positions), default_joint_pos=array(defaults),
        root_lin_vel_b=array(torch.zeros(count, 3) if linear_velocity is None
                             else torch.tensor(linear_velocity, dtype=torch.float32)),
        root_ang_vel_b=array(torch.zeros(count, 3) if angular_velocity is None
                             else torch.tensor(angular_velocity, dtype=torch.float32)),
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_standing_uses_default_pose_and_selected_joints(wrapped):
    env = make_env([[0, 0, 0]], wrapped=wrapped)
    cfg = SceneEntityCfg("robot", joint_ids=[0, 1])
    actual = stand_still_joint_deviation_l1(env, "velocity", asset_cfg=cfg)
    torch.testing.assert_close(actual, torch.tensor([0.8]))


def test_walking_lateral_and_turning_commands_disable_pose_penalty():
    env = make_env([[0, 0, 0], [0.1, 0, 0], [0, -0.1, 0],
                    [0, 0, 0.1], [0, 0, -0.2], [0.07, 0.08, 0]])
    actual = stand_still_joint_deviation_l1(
        env, "velocity", asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]),
    )
    torch.testing.assert_close(actual, torch.tensor([0.8, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize("wrapped", [False, True])
def test_measured_speed_must_be_strictly_below_threshold(wrapped):
    env = make_env(
        [[0, 0, 0]] * 6,
        linear_velocity=[[0, 0, 0], [0.05, 0, 0], [0, -0.1, 0],
                         [4, 0, 0], [0, 0, 0], [0.05, 0, 0]],
        angular_velocity=[[0, 0, 0], [0, 0, 0], [0, 0, 0],
                          [0, 0, 0], [0, 0, -0.1], [0, 0, 0.05]],
        wrapped=wrapped,
    )
    actual = stand_still_joint_deviation_l1(
        env, "velocity", asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]),
    )
    torch.testing.assert_close(actual, torch.tensor([0.8, 0.8, 0, 0, 0, 0.8]))


def test_full_body_penalty_includes_all_joints():
    actual = stand_still_joint_deviation_l1(make_env([[0, 0, 0]]), "velocity")
    torch.testing.assert_close(actual, torch.tensor([10.8]))


@pytest.mark.parametrize("name", ["command_threshold", "yaw_command_threshold",
                                  "velocity_threshold", "yaw_velocity_threshold"])
@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf")])
def test_invalid_gate_parameters_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        stand_still_joint_deviation_l1(make_env([[0, 0, 0]]), "velocity", **{name: value})


def test_rp1_enables_full_body_pose_reward_and_remaps_command():
    cfg = RP1AmpPathEnvCfg()
    term = cfg.rewards.stand_still_joint_deviation
    assert term.func is stand_still_joint_deviation_l1
    assert term.weight == -0.1
    assert term.params["command_name"] == "single_axis_velocity"
    assert term.params["asset_cfg"].joint_names == [".*"]
    for name in ("command_threshold", "yaw_command_threshold",
                 "velocity_threshold", "yaw_velocity_threshold"):
        assert term.params[name] == 0.1


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("command", [[0, 0, 0], [4, 0, 0], [0, 0, 1]])
def test_hip_yaw_deviation_is_active_for_standing_running_and_turning(wrapped, command):
    env = make_env([command], linear_velocity=[[4, 0, 0]], wrapped=wrapped)
    result = joint_deviation_l1(env, SceneEntityCfg("robot", joint_ids=[0, 1]))
    torch.testing.assert_close(result, torch.tensor([0.8]))


def test_rp1_hip_yaw_penalty_selects_only_both_hip_yaw_joints():
    term = RP1AmpPathEnvCfg().rewards.hip_yaw_deviation_l1
    assert term.func is joint_deviation_l1
    assert term.weight == -0.1
    assert term.params["asset_cfg"].joint_names == ["left_hip_yaw_joint", "right_hip_yaw_joint"]
    assert "command_name" not in term.params


def test_rp1_increases_only_the_lateral_penalty_scale():
    cfg = RP1AmpPathEnvCfg()
    assert cfg.rewards.lateral_command_offset.weight == -0.5
    assert cfg.rewards.lateral_command_offset.params["std"] == 0.5
    assert cfg.rewards.hip_yaw_deviation_l1.weight == -0.1
    assert cfg.rewards.stand_still_joint_deviation.weight == -0.1
    assert cfg.rewards.feet_slide.weight == -0.5
    assert cfg.commands.single_axis_velocity.stop_hold_time_range == (5.0, 8.0)
