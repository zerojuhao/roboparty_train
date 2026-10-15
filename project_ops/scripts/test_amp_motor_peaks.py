"""Motor peaks must retain short impulses and ignore discontinuities at reset."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.motor_load import MotorLoadTracker
from robolab.tasks.manager_based.amp.mdp.rewards import (
    braking_load_metrics, braking_power_peak_l2, braking_torque_step_l2, joint_torque_saturation_l2,
)
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg


def data(torques, velocities, wrapped=False):
    proxy = lambda x: SimpleNamespace(torch=x) if wrapped else x
    torque = torch.tensor(torques, dtype=torch.float32)
    return SimpleNamespace(applied_torque=proxy(torque),
                           joint_vel=proxy(torch.tensor(velocities, dtype=torch.float32)),
                           joint_effort_limits=proxy(torch.full_like(torque, 100)))


@pytest.mark.parametrize("wrapped", [False, True])
def test_tracks_within_step_peaks_without_cancelling_braking_against_motoring(wrapped):
    tracker = MotorLoadTracker(1, 2, "cpu")
    tracker.begin_step()
    tracker.sample(data([[-100, 100]], [[10, 20]], wrapped))
    tracker.sample(data([[-10, 0]], [[1, 0]], wrapped))
    torch.testing.assert_close(tracker.torque_ratio_peak, torch.ones(1, 2))
    torch.testing.assert_close(tracker.braking_power_peak, torch.tensor([1000.]))
    torch.testing.assert_close(tracker.torque_delta_ratio_peak, torch.tensor([1.]))


def test_control_step_boundary_retains_previous_torque_but_clears_peak():
    tracker = MotorLoadTracker(1, 1, "cpu")
    tracker.sample(data([[100]], [[0]]))
    tracker.begin_step()
    tracker.sample(data([[-100]], [[0]]))
    torch.testing.assert_close(tracker.torque_delta_ratio_peak, torch.tensor([2.]))
    tracker.begin_step()
    assert not tracker.torque_ratio_peak.any()
    assert not tracker.torque_delta_ratio_peak.any()
    assert not tracker.braking_power_peak.any()


def test_reset_discards_only_reset_environments_torque_history():
    tracker = MotorLoadTracker(2, 1, "cpu")
    tracker.sample(data([[100], [100]], [[0], [0]]))
    tracker.reset(torch.tensor([0]))
    tracker.begin_step()
    tracker.sample(data([[-100], [-100]], [[0], [0]]))
    torch.testing.assert_close(tracker.torque_delta_ratio_peak, torch.tensor([0., 2.]))


def test_peak_saturation_does_not_dilute_one_joint_or_miss_first_substep():
    tracker = MotorLoadTracker(1, 24, "cpu")
    torques = torch.zeros(1, 24)
    torques[0, 5] = 100
    tracker.sample(SimpleNamespace(applied_torque=torques, joint_vel=torch.zeros_like(torques),
                                 joint_effort_limits=torch.full_like(torques, 100)))
    tracker.sample(data([[0] * 24], [[0] * 24]))
    env = SimpleNamespace(motor_load=tracker)
    torch.testing.assert_close(joint_torque_saturation_l2(env, use_substep_peak=True), torch.ones(1),
                               atol=2e-6, rtol=1e-5)
    cfg = SceneEntityCfg("robot", joint_ids=[0, 1])
    assert joint_torque_saturation_l2(env, asset_cfg=cfg, use_substep_peak=True).item() == 0


def braking_env():
    # Raw stop, moving command, yaw-only command, and already stopped.
    commands = torch.tensor([[0., 0, 0], [4., 0, 0], [0., 0, 1], [0., 0, 0]])
    robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.tensor(
        [[3., 0, 0], [3., 0, 0], [0., 0, 0], [0.1, 0, 0]])))
    return SimpleNamespace(
        motor_load=SimpleNamespace(braking_power_peak=torch.tensor([5000.] * 4),
                                   torque_delta_ratio_peak=torch.tensor([2.] * 4)),
        command_manager=SimpleNamespace(get_command=lambda _: commands), scene={"robot": robot},
    )


def test_braking_rewards_only_apply_to_motion_under_low_linear_command():
    env = braking_env()
    torch.testing.assert_close(braking_power_peak_l2(env, "velocity"), torch.tensor([1., 0, 0, 0]))
    torch.testing.assert_close(braking_torque_step_l2(env, "velocity"), torch.tensor([4., 0, 0, 0]))
    env.motor_load.braking_power_peak[:] = 2500
    assert not braking_power_peak_l2(env, "velocity").any()


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_invalid_power_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="power_threshold"):
        braking_power_peak_l2(braking_env(), "velocity", power_threshold=threshold)


def test_braking_metrics_exclude_moving_commands_and_stopped_samples():
    env = braking_env()
    metrics = braking_load_metrics(env, "velocity", power_weight=-0.1)
    torch.testing.assert_close(metrics["Braking/active_fraction"], torch.tensor([1., 0, 0, 0]))
    torch.testing.assert_close(metrics["Braking/power_peak_w"], torch.tensor([5000.]))
    torch.testing.assert_close(metrics["Braking/power_excess_fraction"], torch.ones(1))
    torch.testing.assert_close(metrics["Braking/power_penalty"], torch.tensor([-0.1]))
    torch.testing.assert_close(metrics["Braking/torque_delta_ratio_peak"], torch.tensor([2.]))
    # Environment reset must not mutate samples already queued by the logger.
    env.motor_load.braking_power_peak.zero_()
    env.motor_load.torque_delta_ratio_peak.zero_()
    assert metrics["Braking/power_peak_w"].item() == 5000
    assert metrics["Braking/torque_delta_ratio_peak"].item() == 2


def test_braking_metrics_omit_empty_conditions_instead_of_zero_filling():
    env = braking_env()
    env.scene["robot"].data.root_lin_vel_b.zero_()
    metrics = braking_load_metrics(env, "velocity")
    assert list(metrics) == ["Braking/active_fraction"]
    assert not metrics["Braking/active_fraction"].any()


def test_conditional_samples_are_count_weighted_across_steps():
    env = braking_env()
    first = braking_load_metrics(env, "velocity")["Braking/power_peak_w"]
    env.command_manager.get_command("").zero_()
    env.scene["robot"].data.root_lin_vel_b[:] = torch.tensor([3., 0, 0])
    env.motor_load.braking_power_peak[:] = torch.tensor([1000., 2000., 3000., 4000.])
    second = braking_load_metrics(env, "velocity")["Braking/power_peak_w"]
    # The existing logger concatenates tensors before averaging.
    assert torch.cat([first, second]).mean().item() == 3000


def test_configuration_enables_impulse_penalties_without_command_smoothing():
    cfg = RP1AmpPathEnvCfg()
    for name in ("braking_power_peak_l2", "braking_torque_step_l2"):
        term = getattr(cfg.rewards, name)
        assert term.weight < 0
        assert term.params["command_name"] == "single_axis_velocity"
    assert cfg.rewards.joint_torque_saturation_l2.params["use_substep_peak"] is True
    assert cfg.commands.single_axis_velocity.braking_duration_range is None
    assert cfg.rewards.track_lin_vel_xy_exp.params["use_braking_reference"] is False
