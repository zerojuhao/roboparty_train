"""Check AMP link separation and contact vertical-motion rewards on CPU."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.rewards import link_close_y_gauss, sound_suppression
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg


def proxy(value, wrapped):
    return SimpleNamespace(torch=value) if wrapped else value


@pytest.mark.parametrize("wrapped", [False, True])
def test_link_separation_only_penalizes_below_threshold(wrapped):
    widths = torch.tensor([0.30, 0.18, 0.17, 0.10, 0.0])
    positions = torch.zeros(5, 2, 3)
    positions[:, 0, 1] = widths
    robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=proxy(positions, wrapped), heading_w=proxy(torch.zeros(5), wrapped),
    ))
    env = SimpleNamespace(scene={"robot": robot})
    cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    actual = link_close_y_gauss(env, 0.18, cfg, std=0.1)
    expected = 1 - torch.exp(-torch.clamp(0.18 - widths, min=0) / 0.01)
    torch.testing.assert_close(actual, expected)
    assert actual[0] == actual[1] == 0
    assert torch.all(actual[2:] > 0)


def test_link_separation_ignores_stride_and_is_yaw_translation_invariant():
    positions = torch.tensor([[[2.0, 0.07, 0.8], [-2.0, -0.07, 0.1]]])
    data = SimpleNamespace(body_pos_w=positions, heading_w=torch.zeros(1))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    before = link_close_y_gauss(env, 0.18, cfg)
    data.body_pos_w = positions.clone()
    data.body_pos_w[:, :, 0] = -positions[:, :, 1] + 100
    data.body_pos_w[:, :, 1] = positions[:, :, 0] + 200
    data.heading_w[:] = torch.pi / 2
    torch.testing.assert_close(link_close_y_gauss(env, 0.18, cfg), before, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("wrapped", [False, True])
def test_sound_suppression_masks_airborne_feet_and_uses_separate_indices(wrapped):
    velocity = torch.zeros(3, 4, 6)
    acceleration = torch.zeros_like(velocity)
    force = torch.zeros(3, 5, 3)
    velocity[:, 1, 2], velocity[:, 3, 2] = 2.0, -3.0
    acceleration[:, 1, 2], acceleration[:, 3, 2] = -4.0, 5.0
    # Horizontal motion and angular motion must not enter this term.
    velocity[:, :, 0], acceleration[:, :, 5] = 1000.0, 1000.0
    force[0, 4, 2], force[0, 0, 2] = 100.0, 200.0
    force[1, 4, 2], force[1, 0, 2] = 0.5, 100.0
    force[2, 4, 2], force[2, 0, 2] = 1.0, 0.0
    robot = SimpleNamespace(data=SimpleNamespace(
        body_vel_w=proxy(velocity, wrapped), body_acc_w=proxy(acceleration, wrapped),
    ))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=proxy(force, wrapped)))
    scene = {"robot": robot}
    class Scene(dict):
        sensors = {"contact_forces": sensor}
    env = SimpleNamespace(scene=Scene(scene))
    robot_cfg = SceneEntityCfg("robot", body_ids=[1, 3])
    sensor_cfg = SceneEntityCfg("contact_forces", body_ids=[4, 0])
    # No command manager is present: the term must also work at zero commands.
    actual = sound_suppression(env, sensor_cfg, robot_cfg)
    torch.testing.assert_close(actual, torch.tensor([54.0, 34.0, 0.0]))


def test_invalid_parameters_and_body_count_are_rejected():
    robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=torch.zeros(1, 3, 3), heading_w=torch.zeros(1),
    ))
    env = SimpleNamespace(scene={"robot": robot})
    with pytest.raises(ValueError, match="exactly two"):
        link_close_y_gauss(env, 0.18, SceneEntityCfg("robot", body_ids=[0, 1, 2]))
    for threshold, std in ((0.0, 0.1), (0.18, 0.0)):
        with pytest.raises(ValueError, match="positive"):
            link_close_y_gauss(env, threshold, std=std)


def test_rp1_enables_regularizers_without_changing_actor_observations():
    cfg = RP1AmpPathEnvCfg()
    for name, bodies in (
        ("feet_close_y_gauss", ["left_ankle_roll_link", "right_ankle_roll_link"]),
        ("knee_close_y_gauss", ["left_knee_link", "right_knee_link"]),
    ):
        term = getattr(cfg.rewards, name)
        assert term.func is link_close_y_gauss
        assert term.weight == -0.5
        assert term.params["threshold"] == 0.18
        assert term.params["asset_cfg"].body_names == bodies
        assert term.params["asset_cfg"].preserve_order
    sound = cfg.rewards.sound_suppression
    assert sound.func is sound_suppression
    assert sound.weight == -5e-4
    assert sound.params["asset_cfg"].body_names == sound.params["sensor_cfg"].body_names
    policy_terms = [name for name, term in vars(cfg.observations.policy).items()
                    if hasattr(term, "func")]
    assert policy_terms == ["base_ang_vel", "projected_gravity", "velocity_commands",
                            "joint_pos", "joint_vel", "actions"]
