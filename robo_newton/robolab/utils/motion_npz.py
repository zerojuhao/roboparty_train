"""Named, pickle-free motion interchange for Newton training.

Joint columns follow non-fixed URDF declaration order; rotations are XYZW.
AMP clips contain root/key-body tracks, while tracking clips contain full-body
tracks. Missing full-body measurements are never fabricated from AMP clips.
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET
import warnings

import numpy as np

from .data_paths import motion_file

FORMAT = "roboparty_motion_v1"
FULL_BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
# ``robolab`` is the package root; its data directory is a sibling of assets.
ROBO_NEWTON_ROOT = Path(__file__).resolve().parents[1]


def resolve_motion_manifest(path):
    """Resolve known retired LAFAN entrances without redirecting custom datasets."""
    manifest = Path(str(path)).expanduser()
    if manifest.is_dir() or manifest.name == "rp1_lafan":
        manifest = manifest / "manifest.json"
    canonical = ROBO_NEWTON_ROOT / "data/motions/rp1_lafan/manifest.json"
    if not canonical.is_file():
        canonical = Path(motion_file("rp1_lafan", "manifest.json"))
    old = [
        ROBO_NEWTON_ROOT / "data/motions/rp1_lafan/manifest.json",
        ROBO_NEWTON_ROOT.parent / "data/motions/rp1_lafan/manifest.json",
        ROBO_NEWTON_ROOT.parent / "ufo_data/motions/rp1_lafan/manifest.json",
    ]
    if manifest.resolve() in {entry.resolve() for entry in old} and canonical.is_file():
        warnings.warn(f"Retired LAFAN path {manifest}; using {canonical}", stacklevel=2)
        return canonical
    return manifest


def urdf_names(path):
    """Read the explicit on-disk order, not a backend-dependent tree traversal."""
    root = ET.parse(path).getroot()
    joints = [j.attrib["name"] for j in root.findall("joint") if j.attrib["type"] != "fixed"]
    bodies = [b.attrib["name"] for b in root.findall("link")]
    name_indices(joints, joints)
    name_indices(bodies, bodies)
    return joints, bodies


def name_indices(source, target):
    source, target = list(source), list(target)
    if not source or len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("Motion names must be nonempty and unique.")
    lookup = {name: i for i, name in enumerate(source)}
    missing = set(target) - lookup.keys()
    if missing:
        raise ValueError(f"Motion is missing named entries: {sorted(missing)}")
    return [lookup[name] for name in target]


def resolve_name_patterns(source, patterns):
    """Expand ordered regular-expression name patterns against source names."""
    source = list(source)
    patterns = list(patterns)
    resolved = []
    for pattern in patterns:
        matches = [name for name in source if re.fullmatch(pattern, name)]
        if not matches:
            raise ValueError(f"Name pattern {pattern!r} did not match any source name.")
        resolved.extend(matches)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Name patterns resolve to duplicate names.")
    return resolved


def validate_motion(data):
    for key, expected in (("format", FORMAT), ("quaternion_order", "xyzw"), ("joint_order", "urdf")):
        if key not in data or np.asarray(data[key]).shape != () or str(data[key].item()) != expected:
            raise ValueError(f"Motion metadata {key!r} must be {expected!r}.")
    fps = np.asarray(data["fps"]).reshape(-1)
    if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
        raise ValueError("Motion fps must be finite and positive.")
    names = np.asarray(data["joint_names"])
    if names.ndim != 1 or names.dtype.kind != "U":
        raise ValueError("joint_names must be a Unicode string array.")
    name_indices(names.tolist(), names.tolist())
    joints = np.asarray(data["joint_pos"])
    if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] != len(names):
        raise ValueError("joint_pos must have shape (frames >= 2, len(joint_names)).")
    frames = len(joints)
    shapes = {"joint_pos": joints.shape, "joint_vel": joints.shape,
              "root_pos_w": (frames, 3), "root_quat_w": (frames, 4)}
    full = any(key in data for key in FULL_BODY_FIELDS)
    if full:
        for key in (*FULL_BODY_FIELDS, "body_names", "joint_vel"):
            if key not in data:
                raise ValueError(f"Full-body motion is missing {key}.")
        bodies = np.asarray(data["body_names"])
        if bodies.ndim != 1 or bodies.dtype.kind != "U":
            raise ValueError("body_names must be a Unicode string array.")
        name_indices(bodies.tolist(), bodies.tolist())
        if bodies[0] != "base_link":
            raise ValueError("Full-body motion must put base_link at index zero.")
        shapes.update({key: (frames, len(bodies), 4 if key == "body_quat_w" else 3)
                       for key in FULL_BODY_FIELDS})
    else:
        for key in ("root_pos_w", "root_quat_w", "key_body_pos_w", "key_body_names", "loop_mode"):
            if key not in data:
                raise ValueError(f"AMP motion is missing {key}.")
    if "key_body_pos_w" in data:
        keys = np.asarray(data["key_body_names"])
        if keys.ndim != 1 or keys.dtype.kind != "U":
            raise ValueError("key_body_names must be a Unicode string array.")
        name_indices(keys.tolist(), keys.tolist())
        shapes["key_body_pos_w"] = (frames, len(keys), 3)
    if "loop_mode" in data:
        loop = np.asarray(data["loop_mode"])
        if loop.shape != () or loop.item() not in (0, 1):
            raise ValueError("loop_mode must be scalar 0 (clamp) or 1 (wrap).")
    for key, shape in shapes.items():
        if key not in data:
            continue
        value = np.asarray(data[key])
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}: expected finite floating-point array {shape}.")
        if "quat" in key and np.max(np.abs(np.linalg.norm(value, axis=-1) - 1.0)) > 1e-2:
            raise ValueError(f"{key} contains non-unit quaternions.")
    return data


def load_motion(path, *, joint_names=None, body_names=None, key_body_names=None):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_motion(data)
    for requested, names_key, fields in (
        (joint_names, "joint_names", ("joint_pos", "joint_vel")),
        (body_names, "body_names", FULL_BODY_FIELDS),
        (key_body_names, "key_body_names", ("key_body_pos_w",)),
    ):
        if requested is None:
            continue
        indices = name_indices(data[names_key].tolist(), requested)
        for field in fields:
            if field in data and indices != list(range(data[field].shape[1])):
                data[field] = data[field][:, indices]
        data[names_key] = np.asarray(requested, dtype=str)
    return data
