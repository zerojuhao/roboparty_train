"""Export an RP1 AMP actor and its deployment contract without starting Isaac Lab."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))

from rsl_rl.models.amp_model import AMPModel, resolve_velocity_target_indices
from sim2sim_rp1_amp import (
    MJ_JOINT_NAMES, POLICY_TERM_WIDTHS, policy_histories_from_config,
    read_config, values_by_regex,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing export: {output}")
    agent = read_config(checkpoint.parent / "params/agent.yaml")
    env = read_config(checkpoint.parent / "params/env.yaml")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    histories = policy_histories_from_config(env)
    if "command_path_state" in histories or agent["obs_groups"]["actor"] != ["policy"]:
        raise ValueError("This export requires a proprioceptive RP1 actor using only policy observations.")
    input_size = sum(h.length * h.width for h in histories.values())
    if state["actor_state_dict"]["obs_normalizer._mean"].shape != (1, input_size):
        raise ValueError("Checkpoint input dimension does not match its saved observation configuration.")

    actor_cfg = dict(agent["actor"])
    if actor_cfg.pop("class_name") != "rsl_rl.models.amp_model:AMPModel":
        raise ValueError("Expected an AMPModel checkpoint.")
    # Targets are only used during training, but reconstruct their real layout
    # rather than supplying dummy indices to the production model constructor.
    critic_cfg = env["observations"]["critic"]
    critic_names, critic_shapes = [], []
    for name, term in critic_cfg.items():
        if not isinstance(term, dict) or "func" not in term:
            continue
        length = critic_cfg.get("history_length")
        if length is None:
            length = term.get("history_length", 0)
        width = 3 if name == "base_lin_vel" else POLICY_TERM_WIDTHS[name]
        critic_names.append(name)
        critic_shapes.append((width * max(1, length),))
    actor_cfg["velocity_target_indices"] = resolve_velocity_target_indices(critic_names, critic_shapes)
    policy = AMPModel(
        TensorDict({"policy": torch.zeros(1, input_size)}, batch_size=[1]),
        agent["obs_groups"], "actor", len(MJ_JOINT_NAMES), **actor_cfg,
    )
    policy.load_state_dict(state["actor_state_dict"], strict=True)
    policy.eval()
    exported = policy.as_onnx().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exported, exported.get_dummy_inputs(), str(output), opset_version=18,
        input_names=exported.input_names, output_names=exported.output_names,
        export_params=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)), full_check=True)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(42)
    samples = (
        policy.obs_normalizer._mean
        + torch.randn(128, input_size, generator=generator) * (policy.obs_normalizer._std + 0.01)
    ).float()
    max_error = 0.0
    with torch.inference_mode():
        for sample in samples.split(1):
            reference = policy(TensorDict({"policy": sample}, batch_size=[1])).numpy()
            actual = session.run(["actions"], {"obs": sample.numpy()})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - reference))))

    robot = env["scene"]["robot"]
    actuators = robot["actuators"]["all_joints"]
    offset = 0
    layout = []
    for name, history in histories.items():
        end = offset + history.width * history.length
        layout.append({"name": name, "start": offset, "end_exclusive": end,
                       "frame_width": history.width, "history_length": history.length,
                       "scale": env["observations"]["policy"][name].get("scale")})
        offset = end
    manifest = {
        "checkpoint": str(checkpoint), "iteration": int(state["iter"]),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "input": {"name": "obs", "shape": [1, input_size], "dtype": "float32"},
        "output": {"name": "actions", "shape": [1, len(MJ_JOINT_NAMES)], "dtype": "float32"},
        "observation_layout": layout,
        "history_order": "Per term, oldest to newest; concatenate terms in observation_layout order.",
        "history_initialization": "Repeat the first frame three times; previous action initially zero.",
        "coordinate_frame": "base_link body frame for angular velocity and projected gravity",
        "gravity_convention": "World unit gravity [0, 0, -1] rotated into the base_link frame.",
        "command_order": ["vx_mps", "vy_mps", "yaw_rate_radps"],
        "normalization_and_velocity_estimator_included": True,
        "control_period_s": env["sim"]["dt"] * env["decimation"],
        "joint_order": list(MJ_JOINT_NAMES),
        "joint_position_observation": "Measured position minus default_joint_positions_rad.",
        "previous_action_observation": "Previous raw network action, before scale and default-position offset.",
        "action_scale": env["actions"]["joint_pos"]["scale"],
        "action_clip": agent.get("clip_actions"),
        "target_position": "default_joint_positions_rad + action_scale * actions",
        "default_joint_positions_rad": values_by_regex(robot["init_state"]["joint_pos"], MJ_JOINT_NAMES, 0.0).tolist(),
        "stiffness": values_by_regex(actuators["stiffness"], MJ_JOINT_NAMES).tolist(),
        "damping": values_by_regex(actuators["damping"], MJ_JOINT_NAMES).tolist(),
        "effort_limits_nm": values_by_regex(actuators["effort_limit_sim"], MJ_JOINT_NAMES).tolist(),
        "validation": {"samples": len(samples), "max_abs_action_error": max_error},
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"onnx": str(output), "contract": str(output.with_suffix(".json")),
                      "input": manifest["input"], "output": manifest["output"],
                      "max_abs_action_error": max_error}, indent=2))


if __name__ == "__main__":
    main()
