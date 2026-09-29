"""Create, reset, and step a RoboLab task without an RL runner."""

import argparse
import importlib

import gymnasium as gym
import torch

import train_newton  # noqa: F401
from isaaclab.app.sim_launcher import launch_simulation


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
    args = parser.parse_args()

    module_name, class_name = gym.spec(args.task).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module_name), class_name)()
    cfg.sim.device = args.device
    cfg.sim.physics = cfg.sim.physics.default
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42

    with launch_simulation(cfg, {"device": args.device}):
        env = gym.make(args.task, cfg=cfg)
        try:
            obs, _ = env.reset()
            print(f"RESET {args.task}: {list(obs)}", flush=True)
            for group, values in observation_tensors(obs):
                if not torch.isfinite(values).all():
                    raise RuntimeError(f"non-finite {group} observations on reset")
            actions = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=args.device)
            for step in range(args.steps):
                if args.random_actions:
                    actions.normal_()
                obs, reward, terminated, truncated, _ = env.step(actions)
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
        finally:
            env.close()


if __name__ == "__main__":
    main()
