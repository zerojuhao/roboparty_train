"""Compare training-camera and sim2sim-camera inputs at identical robot poses."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "robo_newton/scripts"), str(ROOT / "robo_newton")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--robot-usd", type=Path, default=Path(
        "/data/zym/roboparty_train/tmp/IsaacLab/usd_20261010_113634_6693/rp1_24dof/rp1_24dof.usda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import train_newton  # noqa: F401
    import gymnasium as gym
    import mujoco
    import numpy as np
    import torch
    import warp as wp
    from isaaclab.app.sim_launcher import launch_simulation
    import isaaclab.sim as sim_utils
    from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEvalEnvCfg, PUFOPhysicsPresetCfg
    from robolab.utils.tensors import as_torch_tensor
    from mujoco.sim2sim_rp1_pufo import Policy, Simulation, latest_checkpoint, make_model, read_config

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else latest_checkpoint()
    saved = read_config(checkpoint.parent / "params/env.yaml")
    manifest_path = ROOT / "robo_newton/data/motions/rp1_lafan/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(c for c in manifest["clips"] if c["name"] == "dance1_subject2")
    source = manifest_path.parent / record["npz"]
    with np.load(source) as data:
        motion = {k: data[k] for k in data.files}
    names = manifest["npz"]["joint_order"]
    motion["body_pos_w"][..., :2] -= motion["body_pos_w"][50, 0, :2].copy()
    sim = Simulation(make_model(False), names, saved)
    policy = Policy(checkpoint)
    contexts, headings, velocities = policy.contexts(motion, sim.default)
    cfg = RP1PUFOEvalEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.sim.physics = PUFOPhysicsPresetCfg().newton_mjwarp
    cfg.seed = 42
    cached_usd = args.robot_usd.resolve()
    if not cached_usd.is_file():
        raise FileNotFoundError(cached_usd)
    spawn = cfg.scene.robot.spawn
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(cached_usd), activate_contact_sensors=spawn.activate_contact_sensors,
        articulation_props=spawn.articulation_props, rigid_props=spawn.rigid_props)
    for name, value in vars(cfg.events).items():
        if hasattr(value, "mode"):
            setattr(cfg.events, name, None)
    cfg.observations.policy.enable_corruption = False
    cfg.commands.motion.motion_names = ["dance1_subject2"]
    print("Creating one nominal, nonrandomized training-camera environment", flush=True)
    with launch_simulation(cfg, {"device": args.device, "require_kit": False,
                                "visualizer": None, "visualizer_explicit": True}):
        wrapped = gym.make("RP1-PUFO-EVAL", cfg=cfg)
        env = wrapped.unwrapped
        try:
            wrapped.reset()
            robot = env.scene["robot"]
            camera = env.scene["camera"]
            order = [names.index(name) for name in robot.joint_names]
            ids = torch.tensor([0], device=args.device)
            reports = []
            for frame in (45, 50, 100, 200):
                sim.set_reference(motion, frame)
                pose = np.r_[sim.data.qpos[:3], sim.data.qpos[[4, 5, 6, 3]]]
                robot.write_joint_position_to_sim_index(
                    position=torch.tensor(motion["joint_pos"][frame, order], device=args.device)[None], env_ids=ids)
                robot.write_joint_velocity_to_sim_index(
                    velocity=torch.zeros(1, 24, device=args.device), env_ids=ids)
                robot.write_root_pose_to_sim_index(
                    root_pose=torch.tensor(pose, dtype=torch.float32, device=args.device)[None], env_ids=ids)
                robot.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(1, 6, device=args.device), env_ids=ids)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(dt=0.0)
                camera.reset(ids)
                camera.update(dt=0.0, force_recompute=True)
                train_raw = as_torch_tensor(camera.data.output["distance_to_image_plane"])[0, :, :, 0].cpu().numpy()
                train_depth = camera.data.output["distance_to_image_plane_noised"][0, :, :, 0].mul(255).round().byte().cpu().numpy()
                sim.depth_history.clear()
                observations, _ = sim.observe(headings[frame], velocities[frame])
                sim_depth = sim.last_depth.copy()
                train_obs = dict(observations)
                train_obs["depth_image"] = torch.from_numpy(train_depth)[None, None].expand(1, 8, -1, -1)
                action_sim = policy.act(observations, contexts[frame:frame+1]).numpy()[0]
                action_train = policy.act(train_obs, contexts[frame:frame+1]).numpy()[0]
                rot = sim.data.xmat[sim.waist].reshape(3, 3)
                eye = sim.data.xpos[sim.waist] + rot @ sim.camera_pos
                dirs = np.ascontiguousarray(sim.rays @ (rot @ sim.camera_rot).T)
                distance = np.empty(len(dirs))
                geomids = np.empty(len(dirs), dtype=np.int32)
                mujoco.mj_multiRay(sim.model, sim.data, eye, dirs.ravel(), np.array([1,0,1,0,0,0], dtype=np.uint8),
                                  True, -1, geomids, distance, None, len(dirs), 10.)
                sim_raw = np.where(distance >= 0, distance*sim.rays[:,0], np.inf).reshape(36,64)
                difference = abs(train_depth.astype(float) - sim_depth.astype(float))
                body_positions = as_torch_tensor(robot.data.body_pos_w)[0].cpu().numpy()
                body_error = max(np.linalg.norm(body_positions[i] - sim.data.xpos[sim.model.body(name).id])
                                 for i, name in enumerate(robot.body_names))
                report = dict(frame=frame, body_position_max_error_m=float(body_error),
                              image_mae_uint8=float(difference.mean()), image_max_error_uint8=float(difference.max()),
                              fraction_pixels_error_over_2=float((difference > 2).mean()),
                              actor_target_delta_rms_rad=float(np.sqrt(np.mean(((action_sim-action_train)*3.2)**2))),
                              train_depth_mean_uint8=float(train_depth.mean()), sim_depth_mean_uint8=float(sim_depth.mean()),
                              train_eye=as_torch_tensor(camera.data.pos_w)[0].cpu().tolist(), sim_eye=eye.tolist())
                reports.append(report)
                print(json.dumps(report), flush=True)
                np.savez_compressed(args.output / f"frame{frame}.npz", train_raw=train_raw, sim_raw=sim_raw,
                                    train_depth=train_depth, sim_depth=sim_depth, geomids=geomids.reshape(36,64),
                                    train_ray_starts=wp.to_torch(camera._ray_starts_w).cpu().numpy(),
                                    train_ray_directions=wp.to_torch(camera._ray_directions_w).cpu().numpy(),
                                    sim_ray_directions=dirs, action_train=action_train, action_sim=action_sim)
            (args.output / "report.json").write_text(json.dumps(dict(
                checkpoint=str(checkpoint), robot_usd=str(cached_usd), frames=reports,
                scope="Identical-pose camera parity; not a training-physics closed-loop rollout",
                camera_installation_randomization=False), indent=2))
        finally:
            wrapped.close()
            sim.close()


if __name__ == "__main__":
    main()
