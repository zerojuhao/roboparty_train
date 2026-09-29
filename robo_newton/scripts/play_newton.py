"""Play RoboLab tasks with Isaac Lab 3.0 and their configured physics backend."""

from pathlib import Path
import sys

import warp as wp
from pxr import Plug

import ovphysx

wp.config.enable_backward = False

# Kitless URDF conversion still needs the PhysX USD schema plugin.
_physx_schema = Path(ovphysx.__file__).parent / "plugins" / "usd" / "PhysxSchema" / "resources"
Plug.Registry().RegisterPlugins(str(_physx_schema / "plugInfo.json"))

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "rsl_rl"))

import robolab.tasks  # noqa: F401, E402
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_play_cli())
