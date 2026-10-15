"""Check the torque margin penalty's scale, selection and sign symmetry."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.rewards import joint_torque_saturation_l2
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg


def make_env(torques, limits, wrapped=False):
    proxy = lambda value: SimpleNamespace(torch=value) if wrapped else value
    robot = SimpleNamespace(data=SimpleNamespace(
        applied_torque=proxy(torch.tensor(torques, dtype=torch.float32)),
        joint_effort_limits=proxy(torch.tensor(limits, dtype=torch.float32)),
    ))
    return SimpleNamespace(scene={"robot": robot})


@pytest.mark.parametrize("wrapped", [False, True])
def test_threshold_half_margin_and_limit_have_expected_scale(wrapped):
    env = make_env([[90, -90], [95, -95], [97.5, -97.5], [100, -100]], [[100, 100]] * 4, wrapped)
    torch.testing.assert_close(joint_torque_saturation_l2(env), torch.tensor([0., 0., 0.25, 1.]),
                               atol=2e-6, rtol=1e-5)


def test_joint_selection_and_different_limits_preserve_relative_scale():
    env = make_env([[100, 1000, -20]], [[100, 100, 20]])
    actual = joint_torque_saturation_l2(env, asset_cfg=SceneEntityCfg("robot", joint_ids=[2, 0]))
    torch.testing.assert_close(actual, torch.ones(1), atol=2e-6, rtol=1e-5)


def test_one_saturated_joint_is_not_cancelled_by_opposite_torque_or_other_joints():
    env = make_env([[100, 0], [100, -100]], [[100, 100]] * 2)
    torch.testing.assert_close(joint_torque_saturation_l2(env), torch.tensor([0.5, 1.]), atol=2e-6, rtol=1e-5)


def test_over_limit_torque_continues_to_increase_penalty():
    env = make_env([[100], [105]], [[100], [100]])
    torch.testing.assert_close(joint_torque_saturation_l2(env), torch.tensor([1., 4.]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("threshold", [0, 1, -1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="limit_ratio"):
        joint_torque_saturation_l2(make_env([[0]], [[100]]), limit_ratio=threshold)


def test_config_keeps_threshold_and_weight_without_timed_speed_reference():
    cfg = RP1AmpPathEnvCfg()
    assert cfg.rewards.joint_torque_saturation_l2.weight == -0.1
    assert cfg.rewards.joint_torque_saturation_l2.params["limit_ratio"] == 0.95
    assert cfg.rewards.joint_torque_saturation_l2.params["use_substep_peak"] is True
    assert cfg.rewards.track_lin_vel_xy_exp.params["use_braking_reference"] is False
