"""Environment definitions for reproducible nominal and robustness tracking."""

from __future__ import annotations

import copy


EVALUATION_MODES = ("nominal", "robustness")


def configured_tracking_modes(cfg):
    modes = getattr(cfg, "tracking_eval_modes", None)
    if modes is None:
        return (None,)
    modes = (modes,) if isinstance(modes, str) else tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in EVALUATION_MODES for mode in modes):
        raise ValueError("tracking_eval_modes must contain unique nominal/robustness modes")
    if getattr(cfg, "tracking_eval_priority_mode", "nominal") != "nominal":
        raise ValueError("Only nominal tracking evaluation can update training priorities")
    return modes


def tracking_environment_config(training_cfg, mode, seed):
    """Copy the environment; nominal uses fixed physical parameters and no noise."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"Unknown tracking evaluation mode: {mode}")
    cfg = copy.deepcopy(training_cfg)
    cfg.seed = int(seed)
    cfg.episode_length_s = 100000.0
    cfg._tracking_eval_mode = mode
    if mode == "robustness":
        return cfg

    fixed_parameters = {
        "randomize_joint_calibration": {"offset_range": (0.0, 0.0)},
        "scale_link_mass": {"mass_distribution_params": (1.0, 1.0)},
        "scale_actuator_gains": {"stiffness_distribution_params": (1.0, 1.0),
                                 "damping_distribution_params": (1.0, 1.0)},
        "randomize_rigid_body_material": {"static_friction_range": (1.0, 1.0),
                                          "dynamic_friction_range": (1.0, 1.0),
                                          "restitution_range": (0.0, 0.0)},
        "randomize_rigid_body_com": {"com_range": {axis: (0.0, 0.0) for axis in "xyz"}},
        "randomize_camera_installation": {"position_error": 0.0, "angle_error_rpy": (0.0, 0.0, 0.0)},
    }
    for name, value in vars(cfg.events).items():
        if value is None:
            continue
        if name == "push_robot" or getattr(value, "mode", None) == "interval":
            setattr(cfg.events, name, None)
        elif name in fixed_parameters:
            value.params.update(fixed_parameters[name])
        elif name.startswith(("randomize_", "scale_")):
            raise ValueError(f"Nominal evaluation needs an explicit fixed setting for event {name!r}")
    for group in vars(cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "rfi_lim"):
            actuator.rfi_lim = 0.0
    material = getattr(cfg.scene.terrain, "physics_material", None)
    if material is not None:
        material.static_friction = material.dynamic_friction = 1.0
        material.restitution = 0.0
    return cfg


def tracking_chunk_seed(seed, first_motion_id):
    """Keep a chunk's random conditions independent of prior rollout durations."""
    return (int(seed) + 1_000_003 * int(first_motion_id)) % (2**31)
