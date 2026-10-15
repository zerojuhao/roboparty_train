"""CPU-only matched rollouts and depth interventions for a saved PUFO checkpoint."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robo_newton/scripts/mujoco"))

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from sim2sim_rp1_pufo import Policy, Simulation, make_model, place_on_stairs, read_config


def normalize(policy, observations):
    result = policy.normalizer.normalize_dict(observations)
    result["depth_image"] = observations["depth_image"]
    return result


def metrics(positions, targets, joints, reference, quaternions, heading):
    displacement = positions[:, :2] - positions[0, :2]
    forward = displacement @ heading
    late = slice(len(targets) // 2, None)
    increments = np.diff(targets, axis=0)
    tilt = np.arccos(np.clip(Rotation.from_quat(quaternions).as_matrix()[:, 2, 2], -1, 1))
    low = np.flatnonzero(positions[:, 2] < .35)
    return dict(
        forward_displacement_m=float(forward[-1]),
        max_backward_excursion_m=float(max(0., -forward.min())),
        displacement_xy_m=displacement[-1].tolist(),
        min_root_height_m=float(positions[:, 2].min()),
        first_root_below_035m_s=float(low[0] / 50) if len(low) else None,
        max_tilt_degrees=float(np.rad2deg(tilt).max()),
        late_tilt_rms_degrees=float(np.rad2deg(np.sqrt(np.mean(tilt[late] ** 2)))),
        late_target_increment_rms_rad=float(np.sqrt(np.mean(increments[late] ** 2))),
        late_joint_error_rms_rad=float(np.sqrt(np.mean((joints[late] - reference) ** 2))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--follow", action="store_true", help="Advance the reference instead of holding a frame")
    parser.add_argument("--start-default", action="store_true", help="Start upright with default joints and zero velocities")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest_path = ROOT / "robo_newton/data/motions/rp1_lafan/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    clip = next(item for item in manifest["clips"] if item["name"] == "dance1_subject2")
    with np.load(manifest_path.parent / clip["npz"]) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    motion["body_pos_w"][..., :2] -= motion["body_pos_w"][args.frame, 0, :2].copy()
    cfg = read_config(args.checkpoint.parent / "params/env.yaml")
    policy = Policy(args.checkpoint)
    names = manifest["npz"]["joint_order"]
    modes = ("baseline", "delay0", "frozen_initial", "actor_stair", "estimator_stair",
             "both_stair", "zero_depth", "reverse_history", "true_velocity", "zero_command",
             "prefilled_history", "zero_estimate", "frozen_estimate")
    sims = [Simulation(make_model(False), names, cfg) for _ in modes]
    donor = Simulation(make_model(True), names, cfg)
    for sim in (*sims, donor):
        sim.set_reference(motion, args.frame)
        if args.start_default:
            yaw = Rotation.from_quat(motion["body_quat_w"][args.frame, 0]).as_euler("xyz")[2]
            sim.data.qpos[3:7] = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
            sim.data.qpos[sim.qids] = sim.default
            sim.data.qvel[:] = 0
            mujoco.mj_forward(sim.model, sim.data)
    place_on_stairs(donor)
    sims[modes.index("delay0")].depth_delay = 0
    contexts, headings, velocities = policy.contexts(motion, sims[0].default)
    context = contexts[args.frame:args.frame + 1]
    heading, velocity = headings[args.frame], velocities[args.frame]
    prefilled = sims[modes.index("prefilled_history")]
    _, initial_values = prefilled.observe(heading, velocity)
    for _ in range(4):
        prefilled.history.add(initial_values)
    prefilled.depth_history.clear()
    steps, count = round(args.duration * 50), len(modes)
    if args.follow and args.frame + steps >= len(contexts):
        raise ValueError("Requested follow interval exceeds the motion")
    positions = np.empty((steps + 1, count, 3))
    joints = np.empty((steps + 1, count, 24))
    quats = np.empty((steps + 1, count, 4))
    targets = np.empty((steps, count, 24))
    estimates = np.empty((steps, count, 3))
    actual_velocities = np.empty_like(estimates)
    images = []
    corpus = {}
    corpus_velocities = []
    fixed_depth = None
    fixed_estimate = None
    actor = policy.actor
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(steps + 1):
                for index, sim in enumerate(sims):
                    positions[step, index] = sim.data.qpos[:3]
                    joints[step, index] = sim.data.qpos[sim.qids]
                    quats[step, index] = sim.data.qpos[[4, 5, 6, 3]]
                if step == steps:
                    break
                frame = args.frame + step if args.follow else args.frame
                heading, velocity = headings[frame], velocities[frame]
                context = contexts[frame + int(args.follow):frame + int(args.follow) + 1]
                pairs = [sim.observe(heading, velocity) for sim in sims]
                observations = {key: torch.cat([item[0][key] for item in pairs])
                                for key in pairs[0][0]}
                donor_obs, donor_values = donor.observe(heading, velocity)
                if fixed_depth is None:
                    fixed_depth = observations["depth_image"][0].clone()
                if step % 25 == 0:
                    images.append(observations["depth_image"][0].numpy().copy())
                observations["depth_image"][modes.index("frozen_initial")] = fixed_depth
                observations["depth_image"][modes.index("zero_depth")] = 0
                reverse = modes.index("reverse_history")
                observations["depth_image"][reverse] = observations["depth_image"][reverse].flip(0)
                observations["velocity_command"][modes.index("zero_command")] = 0
                norm = normalize(policy, observations)
                native_latent = actor.depth_encoder(norm["depth_image"])
                donor_latent = actor.depth_encoder(donor_obs["depth_image"])
                direct, estimator = native_latent.clone(), native_latent.clone()
                for mode in ("actor_stair", "both_stair"):
                    direct[modes.index(mode)] = donor_latent[0]
                for mode in ("estimator_stair", "both_stair"):
                    estimator[modes.index(mode)] = donor_latent[0]
                for index, sim in enumerate(sims):
                    rot = sim.data.xmat[sim.base].reshape(3, 3)
                    yaw = np.arctan2(rot[1, 0], rot[0, 0])
                    c, s = np.cos(yaw), np.sin(yaw)
                    x, y, z = sim.data.qvel[:3]
                    actual_velocities[step, index] = (c*x + s*y, -s*x + c*y, z)
                if step % 10 == 0:
                    for key, value in pairs[0][0].items():
                        corpus.setdefault(key, []).append(value[0].numpy().copy())
                    corpus_velocities.append(actual_velocities[step, 0].copy())
                # Intervene on selected estimator outputs without changing model weights.
                def replace_true_velocity(module, inputs, output):
                    nonlocal fixed_estimate
                    result = output.clone()
                    row = modes.index("true_velocity")
                    result[row] = torch.from_numpy(actual_velocities[step, row]).to(output)
                    result[modes.index("zero_estimate")] = 0
                    frozen = modes.index("frozen_estimate")
                    if fixed_estimate is None:
                        fixed_estimate = output[frozen].clone()
                    result[frozen] = fixed_estimate
                    return result
                hook = actor.velocity_estimator.register_forward_hook(replace_true_velocity)
                try:
                    action, estimate = actor._mean_from_depth_latent(
                        norm, context.expand(count, -1), direct, estimator)
                finally:
                    hook.remove()
                donor_action = policy.act(donor_obs, context).numpy()[0]
                estimates[step] = estimate.numpy()
                for index, (sim, (_, values)) in enumerate(zip(sims, pairs)):
                    target = sim.default + sim.action_scale * np.clip(
                        action[index].numpy() * policy.action_rescale,
                        -policy.action_rescale, policy.action_rescale)
                    targets[step, index] = target
                    sim.advance(action[index].numpy(), policy.action_rescale)
                    sim.history.add(values)
                donor.advance(donor_action, policy.action_rescale)
                donor.history.add(donor_values)
                if step % 100 == 0:
                    print(f"t={step/50:.1f}s elapsed={time.monotonic()-started:.1f}s", flush=True)
    finally:
        for sim in (*sims, donor):
            sim.close()
    initial_heading = Rotation.from_quat(motion["body_quat_w"][args.frame, 0]).as_matrix()[:2, 0]
    results = {}
    for values in (positions, targets, joints, quats, estimates, actual_velocities):
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite diagnostic trace")
    reference = (motion["joint_pos"][args.frame:args.frame + steps + 1]
                 if args.follow else motion["joint_pos"][args.frame])
    for index, mode in enumerate(modes):
        results[mode] = metrics(positions[:, index], targets[:, index], joints[:, index],
                                reference[len(reference)//2:] if args.follow else reference,
                                quats[:, index], initial_heading)
        results[mode]["late_velocity_estimate_mae_m_s"] = np.abs(
            estimates[steps//2:, index] - actual_velocities[steps//2:, index]).mean(axis=0).tolist()
    np.savez_compressed(args.output / "rollouts.npz", modes=modes, root_pos=positions,
                        joint_pos=joints, root_quat=quats, targets=targets,
                        velocity_estimate=estimates, actual_velocity=actual_velocities,
                        baseline_depth_history=np.stack(images), fps=50)
    np.savez_compressed(args.output / "estimator_corpus.npz",
                        **{f"obs__{key}": np.stack(value) for key, value in corpus.items()},
                        true_velocity=np.stack(corpus_velocities), sample_period_s=.2,
                        source_checkpoint=str(args.checkpoint.resolve()))
    report = dict(checkpoint=str(args.checkpoint.resolve()), frame=args.frame,
                  duration_s=args.duration, inference_device="cpu", results=results,
                  follow=args.follow, start_default=args.start_default,
                  limitations="Diagnostic counterfactuals, not deployment fixes. Donor follows its own stair trajectory.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
