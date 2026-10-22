"""Create, reset, and step a RoboLab task without an RL runner."""

import argparse
import importlib
import sys
from pathlib import Path


def observation_tensors(observations, prefix=""):
    for name, value in observations.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "items"):
            yield from observation_tensors(value, path)
        else:
            yield path, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--reward_terms", action="store_true")
    parser.add_argument("--random_actions", action="store_true")
    parser.add_argument("--check_pufo_replay", action="store_true")
    parser.add_argument("--check_pufo_contacts", action="store_true")
    parser.add_argument("--check_parkour", action="store_true",
                        help="Validate terrain colliders and reject immediate mass termination.")
    parser.add_argument("--episode_length_s", type=float)
    parser.add_argument("--terrain_difficulty", type=float,
                        help="Fix generated terrain difficulty for a bounded stress test (0 to 1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_kit", action="store_true", help="Initialize Kit for URDF asset conversion.")
    args = parser.parse_args()
    run_smoke(args)


def run_smoke(args):
    import gymnasium as gym
    import torch
    from robolab.utils.kitless_runtime import configure_kitless_runtime

    configure_kitless_runtime()
    import robolab.tasks  # noqa: F401
    from isaaclab.app.sim_launcher import launch_simulation

    module_name, class_name = gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    cfg.sim.device = args.device
    if hasattr(cfg.sim.physics, "default"):
        cfg.sim.physics = cfg.sim.physics.default
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.terrain_difficulty is not None:
        if not 0. <= args.terrain_difficulty <= 1.:
            raise ValueError("terrain_difficulty must be between 0 and 1")
        cfg.scene.terrain.terrain_generator.difficulty_range = (args.terrain_difficulty, args.terrain_difficulty)
        cfg.scene.terrain.terrain_generator.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s

    with launch_simulation(cfg, {"device": args.device, "require_kit": args.require_kit,
                                "visualizer": None, "visualizer_explicit": True}):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, _ = env.reset()
            check_parkour = getattr(args, "check_parkour", False)
            if check_parkour:
                from isaaclab_newton.physics.newton_manager import NewtonManager
                import mujoco
                import numpy as np
                model = NewtonManager._solver.mj_model
                static = model.geom_bodyid == 0
                actual = {
                    "convex": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH))),
                    "heightfield": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD))),
                    "box": int(np.count_nonzero(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_BOX))),
                }
                expected = env.unwrapped.scene.terrain.collision_counts
                assert actual == expected, f"Terrain collider mismatch: {actual}, expected {expected}"
                from robolab.terrains.native_importer import validate_terrain_colliders
                validate_terrain_colliders(env.unwrapped, None)
                terrain = env.unwrapped.scene.terrain
                assert terrain.stair_edge_validation["tiles"] == len(terrain.terrain_generator.stair_collision_tiles)
                print(f"PARKOUR solver colliders verified: {actual}", flush=True)
            print(f"RESET {args.task}: {list(obs)}", flush=True)
            for group, values in observation_tensors(obs):
                if not torch.isfinite(values).all():
                    raise RuntimeError(f"non-finite {group} observations on reset")
            actions = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=args.device)
            replay = None
            if args.check_pufo_replay:
                from rsl_rl.storage.replay_buffer import ReplayBuffer
                params = cfg.observations.visual.depth_image.params
                offsets = tuple(i * params["history_skip_frames"]
                                for i in range(params["num_output_frames"] - 1, -1, -1))
                replay = ReplayBuffer(
                    args.num_envs * 64, args.num_envs, compact_depth_history=True,
                    depth_history_offsets=offsets, output_key_t=("observation",),
                    output_key_tp1=("observation",),
                )
                reference_depth = []
                previous_done = torch.ones(args.num_envs, 1, dtype=torch.bool)
            checked_contacts = False
            for step in range(args.steps):
                if replay is not None:
                    from robolab.utils.tensors import as_torch_tensor
                    root = as_torch_tensor(env.unwrapped.scene["robot"].data.body_pos_w)[:, 0]
                    hits = as_torch_tensor(env.unwrapped.scene["terrain_priv"].data.ray_hits_w)
                    ground = hits[:, 6 * 21 + 4]
                    torch.testing.assert_close(ground[:, :2], root[:, :2], atol=1e-4, rtol=0)
                    torch.testing.assert_close(obs["critic"][:, 0], root[:, 2] - ground[:, 2])
                    depth = obs["visual"].mul(255.).round().to(device="cpu", dtype=torch.uint8)
                    reference_depth.append(depth)
                    replay.extend({
                        "observation": {
                            "depth_image": depth.unsqueeze(0),
                            "step": torch.full((1, args.num_envs, 1), step, dtype=torch.long),
                            "env_id": torch.arange(args.num_envs).view(1, -1, 1),
                        },
                        "truncated": previous_done.unsqueeze(0),
                        "depth_episode_start": previous_done.unsqueeze(0),
                    })
                    if len(replay) >= 4:
                        sampled = replay.sample(128)
                        for sample in (sampled["observation"], sampled["next"]["observation"]):
                            expected = torch.stack([
                                reference_depth[t][e]
                                for t, e in zip(sample["step"].flatten().tolist(), sample["env_id"].flatten().tolist())
                            ])
                            torch.testing.assert_close(sample["depth_image"], expected)
                if args.random_actions:
                    actions.normal_()
                boundary_probe = args.check_pufo_contacts and step == 7
                if boundary_probe:
                    from robolab.utils.tensors import as_torch_tensor
                    robot = env.unwrapped.scene["robot"]
                    pose = as_torch_tensor(robot.data.root_state_w)[:1, :7].clone()
                    terrain_cfg = cfg.scene.terrain.terrain_generator
                    pose[0, 0] = terrain_cfg.num_rows * terrain_cfg.size[0] / 2 + terrain_cfg.border_width - 1.
                    pose[0, 1:3] = pose.new_tensor([0., 1.])
                    ids = torch.tensor([0], device=args.device)
                    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=args.device), env_ids=ids)
                    env.unwrapped.scene.write_data_to_sim()
                    env.unwrapped.sim.forward()
                obs, reward, terminated, truncated, _ = env.step(actions)
                if check_parkour:
                    terms = env.unwrapped.termination_manager
                    counts = {name: int(terms.get_term(name).sum()) for name in terms.active_terms}
                    print(f"PARKOUR termination counts: {counts}", flush=True)
                    if step == 0:
                        assert not terminated.any(), f"Unexpected first-step termination: {counts}"
                if args.check_pufo_contacts or check_parkour:
                    sensor = env.unwrapped.scene["contact_forces"]
                    history = sensor.data.contact_excess_history
                    assert history.shape[1] == cfg.decimation and history.shape[2] > 0
                    assert torch.isfinite(history).all()
                    if not checked_contacts and not (terminated | truncated).any():
                        from isaaclab_newton.physics import NewtonManager
                        import warp as wp
                        contacts = NewtonManager._contacts
                        count = int(wp.to_torch(contacts.rigid_contact_count)[0])
                        shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:count].long()
                        shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:count].long()
                        forces = wp.to_torch(contacts.force)[:count, :3]
                        rows = wp.to_torch(sensor._contact_shape_rows)
                        expected = torch.zeros(history.shape[0] * history.shape[2], device=history.device)
                        scores = (forces[:, :2].norm(dim=-1) - sensor.stumble_horizontal_ratio * forces[:, 2].abs()).clamp(min=0)
                        matched = 0
                        for shapes in (shape0, shape1):
                            valid_shapes = shapes >= 0
                            contact_rows = rows[shapes[valid_shapes]].long()
                            valid_rows = contact_rows >= 0
                            expected.scatter_reduce_(0, contact_rows[valid_rows], scores[valid_shapes][valid_rows], reduce="amax")
                            matched += int(valid_rows.sum())
                        torch.testing.assert_close(history[:, 0], expected.reshape(history.shape[0], history.shape[2]))
                        if matched:
                            checked_contacts = True
                            print(f"All-contact stumble: {matched} body contacts match raw Newton forces", flush=True)
                    if boundary_probe:
                        assert truncated[0] and not terminated[0], "Boundary must truncate, not terminate"
                        assert env.unwrapped.termination_manager.get_term("terrain_boundary")[0]
                        print("PUFO global boundary: forced edge crossing truncated and reset", flush=True)
                if replay is not None:
                    previous_done = (terminated | truncated).cpu().unsqueeze(-1)
                if not torch.isfinite(reward).all():
                    raise RuntimeError(f"non-finite rewards on step {step}")
                for group, values in observation_tensors(obs):
                    if not torch.isfinite(values).all():
                        bad_ids = (~torch.isfinite(values).reshape(args.num_envs, -1).all(dim=1)).nonzero(as_tuple=True)[0]
                        raise RuntimeError(f"non-finite {group} observations in envs {bad_ids[:12].tolist()} on step {step}")
                print(
                    f"STEP {step}: reward={reward.mean().item():.4f}"
                    f" done={int((terminated | truncated).sum().item())}",
                    flush=True,
                )
                if args.reward_terms:
                    reward_manager = env.unwrapped.reward_manager
                    values = [
                        (name, value.abs().max().item(), value.mean().item())
                        for name, value in zip(reward_manager._term_names, reward_manager._step_reward.T)
                    ]
                    for name, _, mean in sorted(values, key=lambda item: item[1], reverse=True)[:8]:
                        print(f"  {name}={mean:.6g}", flush=True)
            if replay is not None:
                print("PUFO compact replay: real camera current/next histories match exactly", flush=True)
            if args.check_pufo_contacts or check_parkour:
                assert checked_contacts, "Contact smoke must compare actual contacts with the shared sensor"
        finally:
            env.close()


if __name__ == "__main__":
    main()
