"""Named, pickle-free motion interchange for Newton training.

Canonical **on-disk** format is Desktop ``joint_q`` (four keys only):
``joint_q``, ``sample_rate``, ``dof_names``, ``root_quat_format='xyzw'``.
Training derives everything else in memory:

* ``load_motion`` / ``adapt_desktop_joint_q`` → root + joints (+ fps/loop)
* AMP → ``key_body_pos_w`` via Newton FK when missing (MotionDataTerm)
* UFO / BeyondMimic → ``body_*`` + ``joint_vel`` via Newton FK when the
  training scene exists (``newton_motion_fk``); URDF ``ensure_full_body_tracks``
  remains an offline / pytest fallback only

Legacy ``roboparty_motion_v1`` named arrays remain readable for migration.
On-disk training data should be Desktop ``joint_q`` only.
Missing body tracks are never fabricated without FK.
"""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET
import warnings

import numpy as np

from .data_paths import motion_data_dir

FORMAT = "roboparty_motion_v1"
FORMAT_DESKTOP_JOINT_Q = "desktop_joint_q"
FULL_BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
# ``robolab`` is the package root; its data directory is a sibling of assets.
ROBO_NEWTON_ROOT = Path(__file__).resolve().parents[1]


def is_desktop_joint_q(data):
    """True when the NPZ archive uses the Desktop retarget ``joint_q`` layout."""
    return "joint_q" in data


def adapt_desktop_joint_q(data, path=None):
    """Unpack Desktop ``joint_q`` into the internal AMP field layout.

    Expected keys: ``joint_q`` (N, 7+J), ``sample_rate`` (or ``fps``),
    ``dof_names`` (J,), optional ``root_quat_format`` (must be ``xyzw``).

    Returns a dict with ``format=roboparty_motion_v1`` metadata plus
    ``root_pos_w``, ``root_quat_w``, ``joint_pos``, ``fps``, ``joint_names``,
    ``loop_mode``. Does not fabricate ``key_body_*`` or full-body tracks —
    call Newton FK (``newton_motion_fk`` / AMP MotionDataTerm) or URDF
    ``ensure_full_body_tracks`` offline for those.
    """
    joint_q = np.asarray(data["joint_q"], dtype=np.float32)
    if joint_q.ndim != 2 or joint_q.shape[0] < 2 or joint_q.shape[1] < 8:
        raise ValueError(
            "Desktop joint_q must have shape (frames >= 2, 7+J) with J >= 1; "
            f"got {joint_q.shape}."
        )
    if "dof_names" not in data:
        raise ValueError("Desktop joint_q archive is missing dof_names.")
    joint_names = np.asarray(data["dof_names"])
    if joint_names.ndim != 1 or joint_names.dtype.kind not in ("U", "S", "O"):
        raise ValueError("dof_names must be a 1-D string array.")
    joint_names = np.asarray([str(n) for n in joint_names.tolist()], dtype=str)
    n_joints = int(joint_names.shape[0])
    if joint_q.shape[1] != 7 + n_joints:
        raise ValueError(
            f"joint_q width {joint_q.shape[1]} != 7 + len(dof_names)={7 + n_joints}."
        )
    name_indices(joint_names.tolist(), joint_names.tolist())

    quat_fmt = "xyzw"
    if "root_quat_format" in data:
        quat_fmt = str(np.asarray(data["root_quat_format"]).reshape(-1)[0]).lower()
    if quat_fmt != "xyzw":
        raise ValueError(f"Desktop joint_q requires root_quat_format='xyzw', got {quat_fmt!r}.")

    if "sample_rate" in data:
        fps = float(np.asarray(data["sample_rate"]).reshape(-1)[0])
    elif "fps" in data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    else:
        raise ValueError("Desktop joint_q archive needs sample_rate or fps.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid sample_rate/fps: {fps}")

    root_pos_w = np.ascontiguousarray(joint_q[:, :3], dtype=np.float32)
    root_quat_w = np.ascontiguousarray(joint_q[:, 3:7], dtype=np.float32)
    norms = np.linalg.norm(root_quat_w, axis=-1, keepdims=True)
    root_quat_w = root_quat_w / np.clip(norms, 1e-8, None)
    joint_pos = np.ascontiguousarray(joint_q[:, 7:], dtype=np.float32)

    stem = Path(path).stem.lower() if path is not None else ""
    # Loop clips (name contains "loop") wrap; otherwise clamp — matches Desktop intent.
    loop_mode = 1 if "loop" in stem else 0
    if "loop_mode" in data:
        loop_mode = int(np.asarray(data["loop_mode"]).reshape(-1)[0])

    adapted = {
        "format": np.asarray(FORMAT),
        "quaternion_order": np.asarray("xyzw"),
        "joint_order": np.asarray("urdf"),
        "fps": np.asarray(float(fps)),
        "loop_mode": np.asarray(int(loop_mode)),
        "joint_names": joint_names,
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "joint_pos": joint_pos,
        "source_format": np.asarray(FORMAT_DESKTOP_JOINT_Q),
    }
    # Pass through optional full-body / key-body tracks if a hybrid file has them.
    for key in (*FULL_BODY_FIELDS, "body_names", "joint_vel", "key_body_pos_w", "key_body_names"):
        if key in data and key not in adapted:
            adapted[key] = np.asarray(data[key])
    return adapted


def resolve_motion_directory(path):
    """Resolve a clip directory, including retired paths saved in run configs."""
    directory = Path(str(path)).expanduser()
    if directory.name == "manifest.json":
        warnings.warn("Motion manifests are retired; loading NPZ files from the directory.", stacklevel=2)
        directory = directory.parent
    canonical = Path(motion_data_dir("rp1_lafan"))
    old = {
        base / "motions" / prefix / "rp1_lafan"
        for base in (ROBO_NEWTON_ROOT / "data", ROBO_NEWTON_ROOT.parent / "data",
                     ROBO_NEWTON_ROOT.parent / "ufo_data",
                     ROBO_NEWTON_ROOT.parent.parent / "data", ROBO_NEWTON_ROOT.parent.parent / "ufo_data")
        for prefix in ("", "newton")
    }
    if directory.name == "npz" and directory.parent.resolve() in {entry.resolve() for entry in old}:
        directory = directory.parent
    if directory.resolve() in {entry.resolve() for entry in old} and canonical.is_dir():
        return canonical
    return directory


def motion_clip_paths(path, motion_names=None):
    """Discover flat NPZ clips in stable filename order; metadata lives in each clip.

    ``path`` may be one directory or a sequence of directories (e.g. LAFAN +
    KEEP5 action).  Stems must be unique across directories when combining.
    """
    if isinstance(path, (str, Path)):
        directories = [path]
    else:
        directories = list(path)
    if not directories:
        raise ValueError("motion_clip_paths requires at least one motion directory.")

    paths = []
    seen_stems = {}
    for entry in directories:
        directory = resolve_motion_directory(entry)
        if not directory.is_dir():
            raise FileNotFoundError(f"Motion directory does not exist: {directory}")
        found = sorted(p for p in directory.glob("*.npz") if p.is_file())
        if not found:
            raise ValueError(f"Motion directory contains no NPZ clips: {directory}")
        for p in found:
            if p.stem in seen_stems:
                raise ValueError(
                    f"Duplicate motion stem {p.stem!r} in {seen_stems[p.stem]} and {p}"
                )
            seen_stems[p.stem] = p
            paths.append(p)
    paths = sorted(paths, key=lambda p: p.stem)
    if motion_names is not None:
        selected = set(motion_names)
        missing = selected - {p.stem for p in paths}
        if not selected or missing:
            raise ValueError(f"Invalid motion selection; missing names: {sorted(missing)}")
        paths = [p for p in paths if p.stem in selected]
    return paths


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
        # AMP minimum: root + joints + loop. key_body_* is optional — MotionDataTerm
        # computes missing links with Newton FK (needed for Desktop joint_q clips).
        for key in ("root_pos_w", "root_quat_w", "loop_mode"):
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



def default_robot_urdf(robot: str = "rp1") -> Path:
    """Package-local URDF used for offline full-body FK (24-DoF RP1 by default)."""
    name = {"rp1": "rp1/urdf/rp1_24dof.urdf", "rpo": "rpo/urdf/rpo.urdf"}.get(robot, f"{robot}/urdf/{robot}.urdf")
    return ROBO_NEWTON_ROOT / "data" / "robots" / name


def ensure_full_body_tracks(data, *, urdf_path=None, robot: str = "rp1"):
    """Fill ``body_*`` / ``joint_vel`` from URDF FK (offline / pytest fallback).

    Training (BeyondMimic / UFO) prefers Newton FK via ``newton_motion_fk`` when
    a live articulation exists. This helper remains for CPU tests without GPU/Newton.

    No-op when full-body fields are already present. Requires ``root_pos_w``,
    ``root_quat_w``, ``joint_pos``, ``fps``, and ``joint_names`` (as produced by
    ``load_motion`` for Desktop or v1 AMP clips). Does not invent tracks without FK.
    """
    if all(key in data for key in (*FULL_BODY_FIELDS, "body_names", "joint_vel")):
        return data
    missing_src = [k for k in ("root_pos_w", "root_quat_w", "joint_pos", "fps", "joint_names") if k not in data]
    if missing_src:
        raise ValueError(
            "Cannot derive full-body tracks; motion is missing "
            f"{missing_src}. Load Desktop joint_q or v1 root/joint clips first."
        )
    from .motion_fk import compute_full_body_tracks

    path = Path(urdf_path) if urdf_path is not None else default_robot_urdf(robot)
    if not path.is_file():
        raise FileNotFoundError(f"URDF for full-body FK not found: {path}")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    tracks = compute_full_body_tracks(
        data["root_pos_w"],
        data["root_quat_w"],
        data["joint_pos"],
        urdf_path=path,
        fps=fps,
        joint_names=np.asarray(data["joint_names"]).tolist(),
    )
    # Only fill absent fields; never overwrite measured body tracks.
    for key, value in tracks.items():
        if key not in data:
            data[key] = value
    data["body_tracks_source"] = np.asarray("urdf_fk")
    return data



def load_motion(path, *, joint_names=None, body_names=None, key_body_names=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if is_desktop_joint_q(data):
        # Prefer Desktop joint_q as the on-disk format; adapt to internal fields.
        data = adapt_desktop_joint_q(data, path=path)
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
