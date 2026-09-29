"""Check registered RoboLab task configs and their default physics backend."""

import importlib
import traceback

import gymnasium as gym

import robolab.tasks  # noqa: F401
from isaaclab_newton.physics import NewtonCfg


def main() -> int:
    failed = 0
    for task_id, spec in sorted(gym.registry.items()):
        if not task_id.startswith(("RPO-", "RP1-")):
            continue
        try:
            module_name, class_name = spec.kwargs["env_cfg_entry_point"].split(":")
            cfg = getattr(importlib.import_module(module_name), class_name)()
            backend = cfg.sim.physics.default
            if not isinstance(backend, NewtonCfg):
                raise TypeError(f"default physics is {type(backend).__name__}")
            print(f"OK {task_id}: {type(backend).__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {task_id}: {traceback.format_exc().splitlines()[-1]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
