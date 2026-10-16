"""Exercise task loaders on CPU without constructing a physics scene."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_motion_npz import amp_clip, full_clip

pytest.importorskip("isaaclab")
from robolab.tasks.manager_based.amp.managers.motion_data_manager import MotionDataTerm
from robolab.tasks.manager_based.amp.managers.motion_data_term_cfg import MotionDataTermCfg
from robolab.tasks.manager_based.beyondmimic.mdp.commands import MotionLoader
from robolab.utils.math import vel_forward_diff, ang_vel_from_quat_diff


def test_amp_named_columns_and_velocity_parity(tmp_path):
    source = amp_clip()
    source["root_pos_w"] = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    np.savez(tmp_path / "test.npz", **source)
    term = MotionDataTerm.__new__(MotionDataTerm)
    term.cfg = MotionDataTermCfg(motion_data_dir=str(tmp_path), motion_data_weights={"test": 1},
                                 key_body_names=["right_foot", "left_foot"])
    term._env = SimpleNamespace(device="cpu", scene={"robot": SimpleNamespace(joint_names=["right", "left"])})
    term._load_motion_data()
    torch.testing.assert_close(term.dof_pos, torch.from_numpy(source["joint_pos"][:, [1, 0]]), rtol=0, atol=0)
    torch.testing.assert_close(term.root_quat, torch.from_numpy(source["root_quat_w"]), rtol=0, atol=0)
    torch.testing.assert_close(term.root_vel_w, vel_forward_diff(torch.from_numpy(source["root_pos_w"]), .02), rtol=0, atol=0)
    torch.testing.assert_close(term.root_ang_vel_w, ang_vel_from_quat_diff(torch.from_numpy(source["root_quat_w"]), .02, in_frame="world"), rtol=0, atol=0)
    torch.testing.assert_close(term.dof_vel, vel_forward_diff(term.dof_pos, .02), rtol=0, atol=0)
    assert term.motion_loop_modes.tolist() == [1]
    state = term.get_motion_state(torch.tensor([0]), torch.tensor([.01]))
    torch.testing.assert_close(state["dof_pos"][0], (term.dof_pos[0] + term.dof_pos[1]) / 2)


def test_beyondmimic_reorders_bodies_and_keeps_xyzw(tmp_path):
    source = full_clip()
    path = tmp_path / "test.npz"
    np.savez(path, **source)
    loaded = MotionLoader(str(path), [2, 0], joint_names=["right", "left"],
                          body_names=["base_link", "right_foot", "left_foot"])
    np.testing.assert_array_equal(loaded.joint_pos.numpy(), source["joint_pos"][:, [1, 0]])
    np.testing.assert_array_equal(loaded.body_pos_w.numpy(), source["body_pos_w"][:, [1, 0]])
    np.testing.assert_array_equal(loaded.body_quat_w.numpy(), source["body_quat_w"][:, [1, 0]])


def test_rp1_amp_observations_use_urdf_order():
    from robolab.assets.robots.roboparty import RP1_24DOF_CFG, RP1_JOINT_NAMES, RPO_CFG, RPO_JOINT_NAMES
    from robolab.tasks.manager_based.amp.rp1_amp_env_cfg import RP1AmpEnvCfg
    cfg = RP1AmpEnvCfg()
    assert list(RP1_24DOF_CFG.joint_ordering) == RP1_JOINT_NAMES
    assert list(RPO_CFG.joint_ordering) == RPO_JOINT_NAMES
    assert cfg.observations.disc.joint_pos.params["asset_cfg"].joint_names == RP1_JOINT_NAMES
    upper = cfg.observations.upper_disc.joint_pos.params["asset_cfg"].joint_names
    assert [RP1_JOINT_NAMES[i] for i in cfg.observations.upper_disc_demo.ref_joint_pos.params["joint_indices"]] == upper
