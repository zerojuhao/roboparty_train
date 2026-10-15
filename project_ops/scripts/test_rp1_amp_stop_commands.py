"""Exercise AMP cruise-to-stop sampling without a simulator."""

from types import SimpleNamespace

import pytest
import torch

from robolab.tasks.manager_based.amp.mdp.commands import (
    SingleAxisVelocityCommand,
    SingleAxisVelocityCommandCfg,
    validate_braking_duration_range,
)
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1CommandsCfg


def command_fixture(probability=1.0, maximum_speed=4.0, count=64):
    term = SingleAxisVelocityCommand.__new__(SingleAxisVelocityCommand)
    term.cfg = SingleAxisVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        resampling_time_range=(7.5, 15.0),
        rel_standing_envs=0.0,
        stop_sequence_probability=probability,
        ranges=SingleAxisVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, maximum_speed), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0),
        ),
    )
    term._env = SimpleNamespace(num_envs=count, device="cpu")
    term.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(count, 3), heading_w=torch.zeros(count),
    ))
    term.vel_command_b = torch.zeros(count, 3)
    term.is_standing_env = torch.zeros(count, dtype=torch.bool)
    term.command_counter = torch.ones(count, dtype=torch.long)
    term.time_left = torch.full((count,), 10.0)
    term._stop_pending = torch.zeros(count, dtype=torch.bool)
    term._start_pending = torch.zeros(count, dtype=torch.bool)
    term._sequence_speed = torch.zeros(count)
    term._braking_start_velocity = torch.zeros(count, 2)
    term._braking_duration = torch.zeros(count)
    term.segment_start_pos_w = torch.zeros(count, 3)
    term.segment_start_heading_w = torch.zeros(count)
    term.segment_duration = torch.zeros(count)
    return term


def test_cruise_then_zero_command_with_hold_and_new_segment():
    term = command_fixture()
    ids = torch.arange(term.num_envs)
    term._resample_command(ids)
    assert torch.all((term.command[:, 0] >= 3.0) & (term.command[:, 0] <= 4.0))
    assert not term.command[:, 1:].any()
    assert term._stop_pending.all()
    assert not term.is_standing_env.any()
    assert torch.all((term.time_left >= 4.0) & (term.time_left <= 6.0))
    torch.testing.assert_close(term.segment_duration, term.time_left)

    term.robot.data.root_pos_w[:] = torch.tensor([12.0, 0.3, 0.8])
    term._resample_command(ids)
    term._update_command()
    assert not term.command.any()
    assert term.is_standing_env.all()
    assert not term._stop_pending.any()
    assert torch.all((term.time_left >= 2.0) & (term.time_left <= 3.0))
    torch.testing.assert_close(term.segment_duration, term.time_left)
    torch.testing.assert_close(term.segment_start_pos_w, term.robot.data.root_pos_w)


def test_episode_reset_discards_pending_stop_for_only_reset_envs():
    term = command_fixture(count=4)
    term._stop_pending[:] = True
    term.command_counter[:2] = 0
    term._resample_command(torch.arange(4))
    assert torch.all(term.command[:2, 0] >= 3.0)
    assert not term.command[2:].any()
    assert term._stop_pending.tolist() == [True, True, False, False]


def test_stop_sequences_respect_curriculum_and_disabled_sampling():
    for term in (command_fixture(maximum_speed=1.2), command_fixture(probability=0.0)):
        term._resample_command(torch.arange(term.num_envs))
        assert not term._stop_pending.any()
        assert torch.all(term.command[:, 0] <= term.cfg.ranges.lin_vel_x[1])
        torch.testing.assert_close(term.time_left, torch.full_like(term.time_left, 10.0))
        assert torch.all((term.command != 0).sum(dim=1) == 1)


def test_stop_sampling_fraction_and_subset_updates():
    torch.manual_seed(42)
    term = command_fixture(probability=0.2, count=10000)
    term._resample_command(torch.arange(5000))
    assert abs(term._stop_pending[:5000].float().mean().item() - 0.2) < 0.02
    assert not term.command[5000:].any()
    assert not term._stop_pending[5000:].any()


def test_pending_stop_survives_curriculum_speed_reduction():
    term = command_fixture()
    ids = torch.arange(term.num_envs)
    term._resample_command(ids)
    term.cfg.ranges.lin_vel_x = (-0.5, 2.9)
    term._resample_command(ids)
    assert not term.command.any()
    assert term.is_standing_env.all()


def test_standing_commands_are_zero_before_postprocessing():
    term = command_fixture(probability=0.0)
    term.cfg.rel_standing_envs = 1.0
    term._resample_command(torch.arange(term.num_envs))
    assert not term.command.any()


def test_rp1_stop_holds_zero_command_for_five_to_eight_seconds():
    cfg = RP1CommandsCfg().single_axis_velocity
    assert cfg.stop_hold_time_range == (5.0, 8.0)
    assert cfg.stop_sequence_probability == 0.2
    assert cfg.stop_cruise_time_range == (4.0, 6.0)
    term = command_fixture()
    term.cfg.stop_hold_time_range = cfg.stop_hold_time_range
    ids = torch.arange(term.num_envs)
    term._resample_command(ids)
    term._resample_command(ids)
    assert not term.command.any()
    assert term.is_standing_env.all()
    assert torch.all((term.time_left >= 5.0) & (term.time_left <= 8.0))
    torch.testing.assert_close(term.segment_duration, term.time_left)


def braking_fixture(start=4.0, count=4):
    term = command_fixture(count=count)
    term.cfg.braking_duration_range = (2.0, 2.0)
    term.vel_command_b[:, 0] = start
    term._stop_pending[:] = True
    term._resample(torch.arange(count))
    return term


def test_raw_stop_is_instant_but_tracking_reference_brakes_smoothly():
    term = braking_fixture()
    assert not term.command.any()
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), 4.0))
    assert not term.tracking_command[:, 1:].any()
    for elapsed, expected in [(0.5, 3.375), (1.0, 2.0), (1.5, 0.625), (2.0, 0.0), (3.0, 0.0)]:
        term.time_left[:] = term.segment_duration - elapsed
        torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), expected))
        assert not term.command.any()


def test_full_sequence_is_raw_zero_to_four_to_zero_with_reward_only_smoothing():
    term = command_fixture(count=4)
    term.cfg.stop_start_time_range = (0.5, 1.5)
    term.cfg.stop_peak_speed_probability = 1.0
    term.cfg.acceleration_duration_range = (2.0, 2.0)
    term.cfg.braking_duration_range = (3.0, 3.0)
    ids = torch.arange(4)
    term._resample(ids)
    assert not term.command.any()
    assert term._start_pending.all()
    assert not term._stop_pending.any()
    assert torch.all((term.time_left >= 0.5) & (term.time_left <= 1.5))
    term._resample(ids)
    torch.testing.assert_close(term.command[:, 0], torch.full((4,), 4.0))
    assert not term.tracking_command.any()
    assert term._stop_pending.all()
    assert not term._start_pending.any()
    term.time_left[:] = term.segment_duration - 1.0
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), 2.0))
    term.time_left[:] = term.segment_duration - 2.0
    torch.testing.assert_close(term.tracking_command, term.command)
    term._resample(ids)
    assert not term.command.any()
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), 4.0))
    term.time_left[:] = term.segment_duration - 3.0
    assert not term.tracking_command.any()


def test_reset_discards_pending_start_and_preserves_unaffected_environments():
    term = command_fixture(count=4)
    term.cfg.stop_start_time_range = (1.0, 1.0)
    term._resample(torch.arange(4))
    term.command_counter[:2] = 0
    term.cfg.stop_sequence_probability = 0.0
    term.cfg.rel_standing_envs = 1.0
    term._resample(torch.arange(2))
    assert not term.command[:2].any()
    assert term._start_pending.tolist() == [False, False, True, True]


def test_peak_sequence_sampling_matches_probability_and_survives_curriculum_change():
    torch.manual_seed(42)
    term = command_fixture(count=10000)
    term.cfg.stop_start_time_range = (1.0, 1.0)
    term.cfg.stop_peak_speed_probability = 0.5
    ids = torch.arange(term.num_envs)
    term._resample(ids)
    assert abs((term._sequence_speed == 4).float().mean().item() - 0.5) < 0.02
    expected = term._sequence_speed.clone()
    term.cfg.ranges.lin_vel_x = (-0.5, 2.9)
    term._resample(ids)
    torch.testing.assert_close(term.command[:, 0], expected)
    term._resample(ids)
    assert not term.command.any()


def test_optional_four_mps_stop_reference_finishes_in_two_to_three_seconds():
    term = command_fixture(count=32)
    term.cfg.braking_duration_range = (2.0, 3.0)
    term.vel_command_b[:, 0] = 4.0
    term._stop_pending[:] = True
    term._resample(torch.arange(term.num_envs))
    assert not term.command.any()
    assert torch.all((term._braking_duration >= 2.0) & (term._braking_duration <= 3.0))
    term.time_left[:] = term.segment_duration - term._braking_duration * 0.5
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((32,), 2.0))
    term.time_left[:] = term.segment_duration - term._braking_duration - 0.02
    torch.testing.assert_close(term.tracking_command, term.command)


def test_reference_reading_does_not_advance_time_or_mutate_raw_command():
    term = braking_fixture()
    term.time_left[:] = term.segment_duration - 0.5
    reference, timer = term.tracking_command.clone(), term.time_left.clone()
    for _ in range(3):
        torch.testing.assert_close(term.tracking_command, reference)
        torch.testing.assert_close(term.time_left, timer)
        assert not term.command.any()


def test_new_segment_starts_from_current_reference_before_timer_is_replaced():
    term = braking_fixture()
    term.time_left[:] = term.segment_duration - 1.0
    term._stop_pending[:] = True
    term._resample(torch.arange(term.num_envs))
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), 2.0))
    torch.testing.assert_close(term._braking_duration, torch.ones(4))


def test_episode_reset_discards_braking_reference_for_only_reset_subset():
    term = braking_fixture()
    term.time_left[:] = term.segment_duration - 0.5
    unaffected = term.tracking_command[2:].clone()
    term.command_counter[:2] = 0
    term.cfg.rel_standing_envs = 1.0
    term.cfg.stop_sequence_probability = 0.0
    term._resample(torch.arange(2))
    assert not term._braking_duration[:2].any()
    assert not term.tracking_command[:2].any()
    torch.testing.assert_close(term.tracking_command[2:], unaffected)


def test_disabled_reference_and_acceleration_keep_immediate_tracking():
    term = command_fixture()
    term.cfg.braking_duration_range = (2.0, 3.0)
    term._resample(torch.arange(term.num_envs))
    assert not term._braking_duration.any()
    torch.testing.assert_close(term.tracking_command, term.command)
    term.cfg.braking_duration_range = None
    term._resample(torch.arange(term.num_envs))
    assert not term.command.any()
    torch.testing.assert_close(term.tracking_command, term.command)


def test_backward_braking_also_preserves_raw_stop():
    term = braking_fixture(start=-4.0)
    term.time_left[:] = term.segment_duration - 1.0
    assert not term.command.any()
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), -2.0))


@pytest.mark.parametrize("target", [2.0, -4.0])
def test_partial_drop_and_reversal_use_raw_target_with_scaled_reference(target):
    term = command_fixture(count=4, probability=0.0)
    term.cfg.braking_duration_range = (2.0, 2.0)
    term.cfg.ranges.lin_vel_x = (target, target + 1.0e-5)
    term.cfg.ranges.ang_vel_z = (0.0, 0.0)
    term.vel_command_b[:, 0] = 4.0
    term._resample(torch.arange(4))
    torch.testing.assert_close(term.command[:, 0], torch.full((4,), target), atol=2.0e-5, rtol=0)
    torch.testing.assert_close(term.tracking_command[:, 0], torch.full((4,), 4.0))
    torch.testing.assert_close(term._braking_duration, (term.command[:, 0] - 4).abs() / 2)
    term.time_left[:] = term.segment_duration - term._braking_duration * 0.5
    torch.testing.assert_close(term.tracking_command[:, 0], (term.command[:, 0] + 4) * 0.5,
                               atol=2.0e-5, rtol=0)
    term.time_left[:] = term.segment_duration - term._braking_duration - 0.02
    torch.testing.assert_close(term.tracking_command, term.command)


@pytest.mark.parametrize("duration", [(0, 1), (3, 2), (-1, 2), (float("nan"), 3), (2, float("inf")), (2,)])
def test_invalid_braking_duration_range(duration):
    with pytest.raises(ValueError, match="braking_duration_range"):
        validate_braking_duration_range(duration)


def test_rp1_tracks_raw_commands_without_actor_observation_changes():
    from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg

    cfg = RP1AmpPathEnvCfg()
    assert cfg.commands.single_axis_velocity.braking_duration_range is None
    assert cfg.commands.single_axis_velocity.acceleration_duration_range is None
    assert cfg.rewards.track_lin_vel_xy_exp.params["use_braking_reference"] is False
    assert cfg.rewards.track_lin_vel_xy_exp.params["command_name"] == "single_axis_velocity"
    actor_terms = [k for k, v in vars(cfg.observations.policy).items() if hasattr(v, "func")]
    assert actor_terms == ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]


def test_rp1_start_and_stop_tracking_have_no_prescribed_transition_duration():
    cfg = RP1CommandsCfg().single_axis_velocity
    term = command_fixture(count=4)
    for name in ("braking_duration_range", "acceleration_duration_range", "stop_start_time_range"):
        setattr(term.cfg, name, getattr(cfg, name))
    term.cfg.stop_peak_speed_probability = 1.0
    ids = torch.arange(4)
    for speed in (0.0, 4.0, 0.0):
        term._resample(ids)
        torch.testing.assert_close(term.command[:, 0], torch.full((4,), speed))
        torch.testing.assert_close(term.tracking_command, term.command)
        assert not term._braking_duration.any()


@pytest.mark.parametrize("wrapped", [False, True])
def test_tracking_reward_uses_reference_and_returns_to_zero_speed_target(wrapped):
    from robolab.tasks.manager_based.amp.mdp.rewards import track_lin_vel_xy_exp

    term = braking_fixture(count=1)
    array = lambda x: SimpleNamespace(torch=x) if wrapped else x
    speed = torch.tensor([[4.0, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=array(speed), projected_gravity_b=array(torch.tensor([[0.0, 0.0, -1.0]])),
        ))},
        command_manager=SimpleNamespace(get_command=lambda name: term.command, get_term=lambda name: term),
    )
    raw = track_lin_vel_xy_exp(env, 0.5, "velocity")
    shaped = track_lin_vel_xy_exp(env, 0.5, "velocity", use_braking_reference=True)
    assert raw.item() < 1.0e-20
    assert shaped.item() == 1.0
    term.time_left[:] = term.segment_duration - 2.0
    torch.testing.assert_close(track_lin_vel_xy_exp(env, 0.5, "velocity", use_braking_reference=True), raw)
    speed[:] = 0
    assert track_lin_vel_xy_exp(env, 0.5, "velocity", use_braking_reference=True).item() == 1.0


@pytest.mark.parametrize("wrapped", [False, True])
def test_braking_tracking_scale_only_applies_during_raw_stop(wrapped):
    from robolab.tasks.manager_based.amp.mdp.rewards import track_lin_vel_xy_exp

    array = lambda x: SimpleNamespace(torch=x) if wrapped else x
    commands = torch.tensor([[0.0, 0, 0], [4.0, 0, 0], [0.0, 0, 0]], dtype=torch.float32)
    speed = torch.tensor([[4.0, 0, 0], [0.0, 0, 0], [0.1, 0, 0]], dtype=torch.float32)
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=array(speed),
            projected_gravity_b=array(torch.tensor([[0.0, 0.0, -1.0]]).expand(3, -1)),
        ))},
        command_manager=SimpleNamespace(get_command=lambda _: commands),
    )
    raw = track_lin_vel_xy_exp(env, 0.5, "velocity")
    scaled = track_lin_vel_xy_exp(env, 0.5, "velocity", braking_reward_scale=0.35)
    torch.testing.assert_close(scaled, torch.tensor([raw[0] * 0.35, raw[1], raw[2]]))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_braking_tracking_scale_is_rejected(scale):
    from robolab.tasks.manager_based.amp.mdp.rewards import track_lin_vel_xy_exp

    with pytest.raises(ValueError, match="braking_reward_scale"):
        track_lin_vel_xy_exp(SimpleNamespace(), 0.5, "velocity", braking_reward_scale=scale)
