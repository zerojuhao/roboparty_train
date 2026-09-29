"""Train RoboLab tasks with Isaac Lab 3.0 and their configured physics backend."""

from pathlib import Path
import sys

import warp as wp
from pxr import Plug

import ovphysx

wp.config.enable_backward = False

# The Kitless URDF importer applies PhysX USD schemas while preparing assets.
_physx_schema = Path(ovphysx.__file__).parent / "plugins" / "usd" / "PhysxSchema" / "resources"
Plug.Registry().RegisterPlugins(str(_physx_schema / "plugInfo.json"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))

import robolab.tasks  # noqa: F401, E402
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_train_cli())
