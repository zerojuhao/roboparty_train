"""Check the selected RP1 AMP demonstrations without starting a simulator."""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import yaml

from isaaclab.managers import SceneEntityCfg
from robolab.tasks.manager_based.amp.mdp.observations import key_body_pos_b
from robolab.tasks.manager_based.amp.mdp.rewards import lateral_command_offset_l2, turn_command_offset_l2
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpEnvCfg


ROOT = Path(__file__).resolve().parents[1]
JOG_NAMES = ("jog_forward_A021", "jog_forward_A045")


@pytest.fixture(params=JOG_NAMES)
def jog_name(request):
    return request.param


@pytest.fixture
def jogging_motion(jog_name):
    return joblib.load(ROOT / "data/motions/rp1_24dof_lab" / f"{jog_name}.pkl")


def test_dataset_keeps_backward_and_excludes_arc_clips():
    weights = RP1AmpEnvCfg().motion_data.motion_dataset.motion_data_weights
    assert weights["move_back"] > 0
    assert all(weights[name] > 0 for name in JOG_NAMES)
    assert weights["walk_ff_loop_180_R_001__A048"] == 2.0
    assert weights["jog_forward_A021"] == weights["jog_forward_A045"] == 6.0
    assert not any("lafan" in name for name in weights)
    assert not any("arc" in name for name in weights)


def test_turn_sampling_matches_command_mix():
    cfg = RP1AmpEnvCfg()
    weights = cfg.motion_data.motion_dataset.motion_data_weights
    assert weights["turn_l"] == weights["turn_r"]
    turning_fraction = (weights["turn_l"] + weights["turn_r"]) / sum(weights.values())
    command_fraction = 0.5 * (1.0 - cfg.commands.single_axis_velocity.rel_standing_envs)
    assert abs(turning_fraction - command_fraction) < 0.02


def test_jogging_clip_preserves_source_motion(jog_name, jogging_motion):
    provenance = json.loads(
        (ROOT / "data/motions/rp1_24dof_lab" / f"{jog_name}.json").read_text()
    )
    mapping = yaml.safe_load((ROOT / "scripts/tools/retarget/config/rp1_24dof.yaml").read_text())
    frames = slice(provenance["start_frame"], provenance["end_frame_exclusive"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"))
    with np.load(ROOT / "data/motions" / provenance["source_npz"]) as source:
        state = source["joint_q"][frames].astype(np.float64)
        root = state[:, :3].copy()
        root[:, :2] -= root[0, :2].copy()
        np.testing.assert_array_equal(jogging_motion["root_pos"], root.astype(np.float32))
        np.testing.assert_array_equal(jogging_motion["root_rot"], state[:, 3:7].astype(np.float32))
        indices = [source["dof_names"].tolist().index(name) for name in mapping["lab_dof_names"]]
        source_joints = state[:, 7:][:, indices]
        limits = np.array([model.joint(name).range for name in mapping["lab_dof_names"]])
        expected_joints = np.clip(source_joints, limits[:, 0], limits[:, 1])
        np.testing.assert_array_equal(jogging_motion["dof_pos"], expected_joints.astype(np.float32))
        assert np.max(np.abs(jogging_motion["dof_pos"] - source_joints)) < 0.01
        assert jogging_motion["fps"] == source["fps"].item() == 50
    assert jogging_motion["loop_mode"] == 0
    assert len(jogging_motion["root_pos"]) == provenance["end_frame_exclusive"] - provenance["start_frame"]
    assert not provenance["ground_height_corrected"]


def test_jogging_clip_covers_continuous_mid_speed(jogging_motion):
    positions = jogging_motion["root_pos"]
    rotations = Rotation.from_quat(jogging_motion["root_rot"][:, [1, 2, 3, 0]])
    velocity = rotations[:-1].inv().apply(np.diff(positions, axis=0) * jogging_motion["fps"])
    assert 1.5 < np.median(velocity[:, 0]) < 2.2
    assert np.mean((velocity[:, 0] >= 1) & (velocity[:, 0] <= 2.5)) > 0.8
    assert np.mean((velocity[:, 0] >= 1) & (velocity[:, 0] <= 2)) > 0.25
    # Measure path straightness independently of cyclic torso yaw.
    centered_xy = positions[:, :2] - positions[:, :2].mean(axis=0)
    _, _, axes = np.linalg.svd(centered_xy, full_matrices=False)
    assert np.max(np.abs(centered_xy @ axes[1])) < 0.08


def test_jogging_clip_matches_robot_geometry_and_limits(jogging_motion):
    mapping = yaml.safe_load((ROOT / "scripts/tools/retarget/config/rp1_24dof.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"))
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(name).id for name in mapping["lab_dof_names"]])
    body_ids = [model.body(name).id for name in mapping["lab_key_body_names"]]
    joints = jogging_motion["dof_pos"]
    limited = model.jnt_limited[joint_ids].astype(bool)
    ranges = model.jnt_range[joint_ids][limited]
    assert np.all(joints[:, limited] >= ranges[:, 0] - 1e-6)
    assert np.all(joints[:, limited] <= ranges[:, 1] + 1e-6)
    for frame in range(len(joints)):
        data.qpos[:3] = jogging_motion["root_pos"][frame]
        data.qpos[3:7] = jogging_motion["root_rot"][frame]
        data.qpos[model.jnt_qposadr[joint_ids]] = joints[frame]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[body_ids], jogging_motion["key_body_pos"][frame], atol=1e-5)


def test_amp_key_positions_are_invariant_to_absolute_height(jogging_motion):
    data = SimpleNamespace(
        root_pos_w=torch.tensor(jogging_motion["root_pos"]),
        root_quat_w=torch.tensor(jogging_motion["root_rot"][:, [1, 2, 3, 0]]),
        body_pos_w=torch.tensor(jogging_motion["key_body_pos"]),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    asset = SceneEntityCfg("robot", body_ids=list(range(data.body_pos_w.shape[1])))
    before = key_body_pos_b(env, asset)
    data.root_pos_w[:, 2] += 0.1
    data.body_pos_w[:, :, 2] += 0.1
    torch.testing.assert_close(key_body_pos_b(env, asset), before, atol=1e-6, rtol=1e-6)


def test_lateral_penalties_separate_straight_motion_from_turning():
    command_term = SimpleNamespace(
        command=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        lateral_offset_error=torch.tensor([0.2, 0.2, 0.2]),
        planar_offset_error=torch.tensor([[0.2, 0.1], [0.2, -0.1], [0.2, 0.2]]),
    )
    robot = SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=torch.tensor([[0.3, 0.2, 0.0], [0.0, 0.2, 0.0]])))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
        scene={"robot": robot},
    )

    torch.testing.assert_close(
        lateral_command_offset_l2(env, "single_axis_velocity"),
        torch.tensor([0.08, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        turn_command_offset_l2(env, "single_axis_velocity"),
        torch.tensor([0.0, 0.05, 0.0]),
    )

    large_offset_term = SimpleNamespace(
        command=torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        lateral_offset_error=torch.tensor([4.0, 0.0]),
        planar_offset_error=torch.tensor([[4.0, 0.0], [3.0, 4.0]]),
    )
    large_offset_env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: large_offset_term),
    )
    torch.testing.assert_close(
        lateral_command_offset_l2(large_offset_env, "single_axis_velocity"),
        torch.tensor([7.5, 0.0]),
    )
    torch.testing.assert_close(
        turn_command_offset_l2(large_offset_env, "single_axis_velocity"),
        torch.tensor([0.0, 25.0]),
    )
