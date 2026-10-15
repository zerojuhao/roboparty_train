"""Check conditional upper AMP frames and the bounded derived running clip."""

from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import pytest
import torch
import yaml

from prepare_amp_upper_run_reference import ROOT, bounded_run_reference
from robolab.tasks.manager_based.amp.mdp import observations
from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpPathEnvCfg


@pytest.mark.parametrize("name", [
    "ref_root_local_rot_tan_norm", "ref_root_lin_vel_b", "ref_root_ang_vel_b",
    "ref_joint_pos", "ref_joint_vel", "ref_key_body_pos_b",
])
@pytest.mark.parametrize("flatten", [False, True])
def test_all_upper_channels_switch_coherently_only_for_forward_running(monkeypatch, name, flatten):
    shape = (6, 12) if flatten else (6, 3, 4)
    calls = []

    def reference(env, animation, **params):
        calls.append((animation, params))
        return torch.full(shape, 2.0 if animation == "fast" else 1.0)

    monkeypatch.setattr(observations, name, reference)
    commands = torch.tensor([[4, 0, 0], [1.5, 0, 0], [0, 0, 1], [-4, 0, 0],
                             [4, 0, 0.1], [2, 0, -0.09]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: commands))
    result = observations.high_speed_upper_reference(
        env, name, "ordinary", "fast", "velocity", flatten_steps_dim=flatten,
        joint_indices=[1, 2], body_indices=[4, 5],
    )
    torch.testing.assert_close(result[:, 0] if flatten else result[:, 0, 0],
                               torch.tensor([2., 1., 1., 1., 1., 2.]))
    assert [c[0] for c in calls] == ["ordinary", "fast"]
    assert calls[0][1] == calls[1][1]
    if name in ("ref_joint_pos", "ref_joint_vel"):
        assert calls[0][1]["joint_indices"] == [1, 2]
    elif name == "ref_key_body_pos_b":
        assert calls[0][1]["body_indices"] == [4, 5]


def test_config_uses_separate_upper_animation_and_preserves_general_demo():
    cfg = RP1AmpPathEnvCfg()
    assert cfg.animation.animation_upper.motion_data_term == "motion_dataset_upper"
    assert cfg.animation.animation_upper.num_steps_to_use == cfg.animation.animation.num_steps_to_use
    assert cfg.observations.disc_demo.ref_joint_pos.func is observations.ref_joint_pos
    for name, term in vars(cfg.observations.upper_disc_demo).items():
        if hasattr(term, "func"):
            assert term.func is observations.high_speed_upper_reference
            assert term.params["reference_function"] == name
            assert term.params["animation"] == "animation"
            assert term.params["fast_animation"] == "animation_upper"


def test_derived_clip_only_changes_elbows_and_recomputes_consistent_fk():
    data_root = ROOT / "robo_newton/data"
    source_path = data_root / "motions/v0/run_loop_180_R_003__A326.pkl"
    if not source_path.exists():
        pytest.skip("Motion data is not installed")
    source = joblib.load(source_path)
    original = source["dof_pos"].copy()
    mapping = yaml.safe_load((ROOT / "robo_newton/scripts/tools/retarget/config/rp1_24dof.yaml").read_text())
    joints, bodies = mapping["lab_dof_names"], mapping["lab_key_body_names"]
    model = mujoco.MjModel.from_xml_path(str(data_root / "robots/roboparty/rp1.3/mjcf/rp1_flat.xml"))
    derived = bounded_run_reference(source, model, joints, bodies)
    np.testing.assert_array_equal(source["dof_pos"], original)
    elbow_ids = [joints.index(f"{side}_elbow_joint") for side in ("left", "right")]
    other = [i for i in range(len(joints)) if i not in elbow_ids]
    np.testing.assert_array_equal(derived["dof_pos"][:, other], original[:, other])
    for index, low, high in zip(elbow_ids, [-30, -45], [45, 30]):
        assert np.rad2deg(derived["dof_pos"][:, index]).min() >= low - 1e-4
        assert np.rad2deg(derived["dof_pos"][:, index]).max() <= high + 1e-4
    np.testing.assert_array_equal(derived["root_pos"], source["root_pos"])
    np.testing.assert_array_equal(derived["root_rot"], source["root_rot"])
    assert np.isfinite(derived["key_body_pos"]).all()
    assert np.abs(derived["key_body_pos"][:, 4:6] - source["key_body_pos"][:, 4:6]).max() > 0
    data = mujoco.MjData(model)
    for frame in (0, len(original) // 2, len(original) - 1):
        data.qpos[:3] = derived["root_pos"][frame]
        data.qpos[3:7] = derived["root_rot"][frame]
        for index, joint in enumerate(joints):
            data.qpos[model.jnt_qposadr[model.joint(joint).id]] = derived["dof_pos"][frame, index]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(derived["key_body_pos"][frame],
                                   data.xpos[[model.body(b).id for b in bodies]], atol=1e-6)
