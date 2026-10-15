"""Create a derived upper AMP run clip; keep original motion files unchanged."""

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def bounded_run_reference(source, model, joint_names, body_names):
    result = dict(source)
    positions = np.asarray(source["dof_pos"]).copy()
    for name, lower, upper in (
        ("left_elbow_joint", -30.0, 45.0), ("right_elbow_joint", -45.0, 30.0),
    ):
        index = joint_names.index(name)
        positions[:, index] = np.clip(positions[:, index], np.deg2rad(lower), np.deg2rad(upper))
    result["dof_pos"] = positions
    # Recompute wrist positions after changing elbows; velocities are derived by the loader.
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in joint_names]
    body_ids = [model.body(name).id for name in body_names]
    data = mujoco.MjData(model)
    key_positions = np.empty_like(source["key_body_pos"])
    for i in range(len(positions)):
        data.qpos[:3] = source["root_pos"][i]
        data.qpos[3:7] = source["root_rot"][i]
        data.qpos[addresses] = positions[i]
        mujoco.mj_forward(model, data)
        key_positions[i] = data.xpos[body_ids]
    result["key_body_pos"] = key_positions
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "robo_newton/data")
    args = parser.parse_args()
    source_path = args.data_root / "motions/v0/run_loop_180_R_003__A326.pkl"
    output = args.data_root / "motions/rp1_upper_run_bounded/run_loop_180_R_003__A326.pkl"
    mapping = yaml.safe_load((ROOT / "robo_newton/scripts/tools/retarget/config/rp1_24dof.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(args.data_root / "robots/roboparty/rp1.3/mjcf/rp1_flat.xml"))
    source = joblib.load(source_path)
    derived = bounded_run_reference(source, model, mapping["lab_dof_names"], mapping["lab_key_body_names"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = joblib.load(output)
        if not all(np.array_equal(previous[k], v) for k, v in derived.items()):
            raise FileExistsError(f"Derived reference differs: {output}")
    else:
        joblib.dump(derived, output)
    print(json.dumps({"source": str(source_path), "output": str(output),
                      "frames": len(derived["dof_pos"]),
                      "largest_angle_adjustment_deg": float(np.rad2deg(
                          np.abs(derived["dof_pos"] - source["dof_pos"]).max()))}))


if __name__ == "__main__":
    main()
