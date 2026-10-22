"""Kitless Isaac Lab 3.0 process setup shared by train, play, and smoke scripts."""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
from collections.abc import Callable
from typing import Any


def configure_kitless_runtime() -> Path:
    """Register kitless USD plugins and default data/log paths.

    Kitless URDF import still applies PhysX USD schemas. Call this before
    constructing environments. Does not start Kit. Packages come from the uv env.
    """
    package_dir = Path(__file__).resolve().parents[1]
    root = package_dir.parents[1]
    local_data = package_dir / "data"
    if local_data.is_dir():
        os.environ.setdefault("ROBO_TRAIN_DATA_ROOT", str(local_data))
    os.environ.setdefault("ROBO_TRAIN_LOG_ROOT", str(root / "logs" / "rsl_rl"))

    import ovphysx
    import warp as wp
    from pxr import Plug

    wp.config.enable_backward = False
    registry = Plug.Registry()
    # Newton + MuJoCo schemas must load before USD builds its process-wide registry.
    # The kitless URDF importer applies MjcJointAPI during conversion.
    try:
        import newton_usd_schemas  # noqa: F401
    except ImportError:
        pass
    try:
        import mujoco_usd_converter
    except ImportError:
        pass
    else:
        mjc_plugins = Path(mujoco_usd_converter.__file__).resolve().parent / "plugins"
        if mjc_plugins.is_dir() and not registry.GetPluginWithName("mjcPhysics"):
            registry.RegisterPlugins(str(mjc_plugins))
    # Official ovphysx 0.6 layout: schemas/physx/<Schema>/resources.
    registered = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    schema_paths = [
        str(path)
        for path in ovphysx.codeless_schema_paths()
        if path.parent.name.casefold() not in registered
    ]
    if schema_paths:
        registry.RegisterPlugins(schema_paths)
    try:
        import ovstage
    except ImportError:
        pass
    else:
        register_ovstage_schemas = getattr(getattr(ovstage, "population", None), "register_usd_schemas", None)
        schema_root = getattr(ovphysx, "codeless_schema_root", None)
        if callable(register_ovstage_schemas) and callable(schema_root):
            register_ovstage_schemas(str(schema_root()))
    return root


def parse_isaaclab_cli(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse script and Isaac Lab launcher args, leaving Hydra tokens on argv."""
    from isaaclab.app import add_launcher_args
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    add_launcher_args(parser)
    args_cli, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    if getattr(args_cli, "video", False):
        args_cli.enable_cameras = True
    return args_cli


def run_resolved_task(
    args_cli: argparse.Namespace,
    main_fn: Callable[..., Any],
    *,
    play_mode: bool = False,
) -> Any:
    """Resolve Hydra/preset configs, launch the matching runtime, then run *main_fn*."""
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils.hydra import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=play_mode)
    with launch_simulation(env_cfg, args_cli):
        return main_fn(env_cfg, agent_cfg)


def sim_is_running(env) -> bool:
    """Return whether the launched simulation should keep stepping."""
    sim = getattr(getattr(env, "unwrapped", env), "sim", None)
    if sim is None:
        return True
    running = getattr(sim, "is_running", None)
    if callable(running):
        return bool(running())
    app = getattr(sim, "app", None)
    if app is not None and callable(getattr(app, "is_running", None)):
        return bool(app.is_running())
    return True
