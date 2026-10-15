"""Audit RP1 AMP demonstration kinematics and velocity coverage with MuJoCo."""

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2] / "robo_newton"
DEFAULT_MOTIONS = (
    "stand_still", "walk_arc_cw_loop_001__A048", "walk_arc_cw_loop_001__A048_M",
    "walk_ff_loop_180_R_001__A048", "move_back", "run_start_180_R_001__A345",
    "run_stop_180_R_001__A345", "turn_l", "turn_r",
)


def audit(motion_names):
    mapping_path = ROOT / "scripts/tools/retarget/config/rp1_24dof.yaml"
    model_path = ROOT / "data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"
    mapping = yaml.safe_load(mapping_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_position_addresses = [
        model.jnt_qposadr[model.joint(name).id] for name in mapping["lab_dof_names"]
    ]
    key_body_ids = [model.body(name).id for name in mapping["lab_key_body_names"]]
    results = {}
    for name in motion_names:
        motion = joblib.load(ROOT / "data/motions/rp1_24dof_lab" / f"{name}.pkl")
        fps = float(motion["fps"])
        position = np.asarray(motion["root_pos"])
        quat = np.asarray(motion["root_rot"])
        # Dataset rotations are wxyz; both velocities below are in the body frame.
        rotations = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        body_velocity = rotations[:-1].inv().apply(np.diff(position, axis=0) * fps)
        body_angular_velocity = (rotations[:-1].inv() * rotations[1:]).as_rotvec() * fps
        errors = []
        sampled_frames = np.linspace(0, len(position) - 1, min(100, len(position)), dtype=int)
        for frame in sampled_frames:
            data.qpos[:3] = position[frame]
            data.qpos[3:7] = quat[frame]
            data.qpos[joint_position_addresses] = motion["dof_pos"][frame]
            mujoco.mj_forward(model, data)
            position_error = data.xpos[key_body_ids] - motion["key_body_pos"][frame]
            errors.append(np.linalg.norm(position_error, axis=-1))
        mid_speed_frames = (body_velocity[:, 0] >= 1) & (body_velocity[:, 0] <= 2)
        mean_error_by_body = dict(zip(mapping["lab_key_body_names"], np.mean(errors, axis=0).tolist()))
        results[name] = {
            "duration_s": (len(position) - 1) / fps,
            "fps": fps,
            "body_vx_p05_p50_p95": np.quantile(body_velocity[:, 0], [.05, .5, .95]).tolist(),
            "body_vy_p05_p50_p95": np.quantile(body_velocity[:, 1], [.05, .5, .95]).tolist(),
            "body_yaw_rate_p05_p50_p95": np.quantile(body_angular_velocity[:, 2], [.05, .5, .95]).tolist(),
            "fraction_vx_1_to_2": float(np.mean(mid_speed_frames)),
            "key_body_fk_mean_error_m": float(np.mean(errors)),
            "key_body_fk_max_error_m": float(np.max(errors)),
            "key_body_fk_error_by_name_m": mean_error_by_body,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", nargs="+", default=DEFAULT_MOTIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.motions), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)
