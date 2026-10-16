"""CPU-only format and named-axis regressions."""

import numpy as np
import pytest

from robolab.utils.motion_npz import FORMAT, FULL_BODY_FIELDS, load_motion, name_indices, urdf_names, validate_motion
import robolab.utils.motion_npz as motion_npz


def amp_clip():
    return dict(format=np.asarray(FORMAT), quaternion_order=np.asarray("xyzw"), joint_order=np.asarray("urdf"),
                fps=np.asarray(50.0), loop_mode=np.asarray(1), joint_names=np.asarray(["left", "right"]),
                joint_pos=np.arange(8, dtype=np.float32).reshape(4, 2),
                root_pos_w=np.zeros((4, 3), dtype=np.float32),
                root_quat_w=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (4, 1)),
                key_body_names=np.asarray(["left_foot", "right_foot"]),
                key_body_pos_w=np.arange(24, dtype=np.float32).reshape(4, 2, 3))


def full_clip():
    data = amp_clip()
    data["body_names"] = np.asarray(["base_link", "left_foot", "right_foot"])
    data["joint_vel"] = data["joint_pos"] * 2
    for field in FULL_BODY_FIELDS:
        data[field] = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
    data["body_quat_w"] = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (4, 3, 1))
    return data


@pytest.mark.parametrize("factory", [amp_clip, full_clip])
def test_roundtrip_and_named_mapping(tmp_path, factory):
    data = factory()
    path = tmp_path / "clip.npz"
    np.savez_compressed(path, **data)
    loaded = load_motion(path, joint_names=["right", "left"], key_body_names=["right_foot", "left_foot"])
    np.testing.assert_array_equal(loaded["joint_pos"], data["joint_pos"][:, [1, 0]])
    np.testing.assert_array_equal(loaded["key_body_pos_w"], data["key_body_pos_w"][:, [1, 0]])
    np.testing.assert_array_equal(loaded["root_quat_w"], data["root_quat_w"])
    if "body_names" in data:
        loaded = load_motion(path, body_names=["base_link", "right_foot", "left_foot"])
        for field in FULL_BODY_FIELDS:
            np.testing.assert_array_equal(loaded[field], data[field][:, [0, 2, 1]])


@pytest.mark.parametrize("key,value", [
    ("quaternion_order", np.asarray("wxyz")), ("joint_order", np.asarray("physx")),
    ("format", np.asarray("legacy")), ("fps", np.asarray(0)), ("fps", np.asarray(np.nan)),
    ("joint_names", np.asarray(["left", "left"])), ("joint_pos", np.zeros((1, 2), dtype=np.float32)),
    ("root_pos_w", np.full((4, 3), np.nan)), ("root_quat_w", np.zeros((4, 4), dtype=np.float32)),
    ("key_body_pos_w", np.zeros((3, 2, 3), dtype=np.float32)), ("loop_mode", np.asarray(2)),
])
def test_bad_metadata_or_shapes_rejected(key, value):
    data = amp_clip()
    data[key] = value
    with pytest.raises(ValueError):
        validate_motion(data)


def test_missing_names_rejected():
    with pytest.raises(ValueError):
        name_indices(["a", "b"], ["a", "c"])


def test_urdf_declaration_order_skips_fixed(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text('<robot><link name="base_link"/><link name="child"/>'
                    '<joint name="right" type="revolute"/><joint name="fixed" type="fixed"/>'
                    '<joint name="left" type="continuous"/></robot>')
    assert urdf_names(path) == (["right", "left"], ["base_link", "child"])


def test_full_body_requires_all_fields():
    data = full_clip()
    del data["body_lin_vel_w"]
    with pytest.raises(ValueError):
        validate_motion(data)


def test_retired_manifest_alias_without_symlinks(tmp_path, monkeypatch):
    package = tmp_path / "robo_newton"
    monkeypatch.setattr(motion_npz, "ROBO_NEWTON_ROOT", package)
    canonical = package / "data/motions/newton/rp1_lafan/manifest.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{}")
    for base in (package / "data", tmp_path / "data", tmp_path / "ufo_data"):
        old = base / "motions/rp1_lafan/manifest.json"
        with pytest.warns(UserWarning, match="Retired LAFAN"):
            assert motion_npz.resolve_motion_manifest(old) == canonical
    custom = tmp_path / "custom/motions/rp1_lafan/manifest.json"
    assert motion_npz.resolve_motion_manifest(custom) == custom
    assert motion_npz.resolve_motion_manifest(canonical) == canonical
