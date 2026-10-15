"""Prepare the reviewed forward-jog windows for RP1 AMP; preserve source NPZs."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "data/motions/jog_rp1_24dof_50fps_npz"
# End frames are exclusive. Keep complete continuous windows, not isolated speed-matched frames.
JOG_WINDOWS = {
    "jog_forward_A021": ("220713__jog_forward_loop_001__A021_rp1_24dof", 163, 380),
    "jog_forward_A045": ("221010__jog_ff_loop_180_R_002__A045_rp1_24dof", 72, 293),
}


def prepare_clip(source: Path, start: int, end: int, model, mapping: dict) -> tuple[dict, dict]:
    with np.load(source, allow_pickle=False) as raw:
        if str(raw["root_quat_format"].item()) != "wxyz":
            raise ValueError(f"Expected wxyz root quaternions: {source}")
        names = raw["dof_names"].tolist()
        state = raw["joint_q"][start:end].astype(np.float64)
        fps = float(raw["sample_rate"].item())

    joint_ids = np.array([model.joint(name).id for name in names])
    joint_positions = state[:, 7:]
    original_joints = joint_positions.copy()
    limited = model.jnt_limited[joint_ids].astype(bool)
    limits = model.jnt_range[joint_ids][limited]
    joint_positions[:, limited] = np.clip(joint_positions[:, limited], limits[:, 0], limits[:, 1])
    correction = float(np.max(np.abs(joint_positions - original_joints)))
    if correction > 0.01:
        raise ValueError(f"Joint correction exceeds the reviewed 0.01 rad bound: {source}")

    state[:, :2] -= state[0, :2].copy()
    body_ids = [model.body(name).id for name in mapping["lab_key_body_names"]]
    data = mujoco.MjData(model)
    key_positions = []
    ground_penetration = []
    for frame in state:
        data.qpos[:7] = frame[:7]
        data.qpos[model.jnt_qposadr[joint_ids]] = frame[7:]
        mujoco.mj_forward(model, data)
        key_positions.append(data.xpos[body_ids].copy())
        ground_penetration.append(max(
            [0.0] + [-float(contact.dist) for contact in data.contact
                     if 0 in model.geom_bodyid[[contact.geom1, contact.geom2]]]
        ))

    lab_indices = [names.index(name) for name in mapping["lab_dof_names"]]
    motion = {
        "fps": fps,
        "root_pos": state[:, :3].astype(np.float32),
        "root_rot": state[:, 3:7].astype(np.float32),
        "dof_pos": joint_positions[:, lab_indices].astype(np.float32),
        "key_body_pos": np.asarray(key_positions, dtype=np.float32),
        "loop_mode": 0,
    }
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "source_npz": str(source.relative_to(ROOT / "data/motions")),
        "source_sha256": source_hash,
        "start_frame": start,
        "end_frame_exclusive": end,
        "fps": fps,
        "max_joint_correction_rad": correction,
        "ground_penetration_p50_p95_max_m": np.quantile(ground_penetration, [.5, .95, 1]).tolist(),
        "root_quaternion_order": "wxyz",
        "ground_height_corrected": False,
        "transforms": ["center root XY", "clamp small joint-limit overshoot", "named joint mapping", "MuJoCo FK"],
        "note": "Absolute height is preserved. AMP uses body-relative key positions, not root height.",
    }
    return motion, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/motions/rp1_24dof_lab")
    args = parser.parse_args()
    mapping = yaml.safe_load((ROOT / "scripts/tools/retarget/config/rp1_24dof.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in JOG_WINDOWS:
        for suffix in (".pkl", ".json"):
            target = args.output_dir / f"{name}{suffix}"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
    for name, (source_name, start, end) in JOG_WINDOWS.items():
        motion, provenance = prepare_clip(SOURCE_DIR / f"{source_name}.npz", start, end, model, mapping)
        with (args.output_dir / f"{name}.pkl").open("xb") as stream:
            pickle.dump(motion, stream)
        with (args.output_dir / f"{name}.json").open("x") as stream:
            json.dump(provenance, stream, indent=2)
        print(name, json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
