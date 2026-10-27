"""Pre-reset physical snapshots for the optional flat-UFO world model."""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from robolab.utils.tensors import as_torch_tensor


def world_model_state(env):
    robot = env.scene["robot"]
    data = robot.data
    pos = as_torch_tensor(data.body_pos_w)[:, 0]
    return torch.cat((
        pos[:, 2:3] - env.scene.env_origins[:, 2:3],
        as_torch_tensor(data.body_quat_w)[:, 0],
        as_torch_tensor(data.body_link_lin_vel_w)[:, 0],
        as_torch_tensor(data.body_link_ang_vel_w)[:, 0],
        as_torch_tensor(data.joint_pos),
        as_torch_tensor(data.joint_vel),
    ), dim=-1)


def enable_world_model_observations(env_cfg):
    group = ObservationGroupCfg()
    group.state = ObservationTermCfg(func=world_model_state)
    group.enable_corruption = False
    group.concatenate_terms = True
    env_cfg.observations.world_model = group
    env_cfg.compute_final_obs = True


class WorldModelAdapter:
    """Read physical state and fixed simulator context ``xi`` for learned dynamics."""

    def __init__(self, env, action_rescale, clip_actions):
        self.env = env
        self.robot = env.scene["robot"]
        if env.scene.terrain.cfg.terrain_type != "plane":
            raise ValueError("World model currently supports flat RP1-UFO only")
        if set(self.robot.actuators) != {"all_joints"} or self.robot.num_joints != 24:
            raise ValueError("World model requires the RP1 24-DoF all_joints actuator")
        actuator = self.robot.actuators["all_joints"]
        if actuator.cfg.min_delay != 0 or actuator.cfg.max_delay != 0:
            raise ValueError("UFO world model requires zero actuator delay")
        if set(env.termination_manager.active_terms) != {"time_out"}:
            raise ValueError("World model expects timeout-only UFO termination")
        action = env.action_manager.get_term("joint_pos")
        if action.cfg.clip is not None:
            raise ValueError("World model does not support per-joint action clipping")
        import warp as wp

        model = env.sim.physics_manager.get_model()
        self.friction = wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        material_cfg = env.cfg.events.randomize_rigid_body_material
        self.friction_range = tuple(float(value) for value in material_cfg.params["static_friction_range"])
        if self.friction_range[1] <= self.friction_range[0]:
            raise ValueError("World model requires an increasing friction normalization range")
        torso_ids, torso_names = self.robot.find_bodies("waist_yaw_link", preserve_order=True)
        if len(torso_ids) != 1:
            raise ValueError(f"World model requires one waist_yaw_link body, got {torso_names}")
        self.torso_body_ids = torch.as_tensor(torso_ids, dtype=torch.long, device=env.device)
        self.action_rescale = float(action_rescale)
        self.clip_actions = float(clip_actions) if clip_actions is not None else None
        self._mass_term = self._find_event_term("scale_link_mass")
        self._gain_term = self._find_event_term("scale_actuator_gains")
        data = self.robot.data
        current_mass = as_torch_tensor(data.body_mass).detach().clone()
        default_mass = getattr(self._mass_term, "default_mass", None)
        self._nominal_mass = (as_torch_tensor(default_mass).detach().clone()
                              if default_mass is not None else current_mass)
        self._gain_name = "all_joints"
        self._nominal_stiffness, self._nominal_damping = self._gain_snapshot()
        self._root_com_pos = as_torch_tensor(data.body_com_pos_b)[0, 0].detach().clone()
        self._validate_root_com()
        env._ufo_wm_push_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ufo_wm_pre_push_state = torch.zeros_like(world_model_state(env))
        env._ufo_wm_push_target = torch.zeros(env.num_envs, 6, device=env.device)

    def begin_step(self):
        """Start event collection before the simulator integrates a policy action."""
        self.env._ufo_wm_push_mask.zero_()
        self.env._ufo_wm_push_target.zero_()

    def transition_outcome(self, next_state, dones):
        """Separate post-integration pushes from the learned physical transition.

        ``next_state`` must already contain pre-reset final observations. Interval
        events execute after reset, so pushes on terminal rows belong to the new
        episode and must not alter the outgoing transition.
        """
        active = self.env._ufo_wm_push_mask & ~dones.reshape(-1).bool()
        clean_next_state = torch.where(active[:, None], self.env._ufo_wm_pre_push_state, next_state)
        disturbance = torch.where(active[:, None], self.env._ufo_wm_push_target, 0.0)
        return clean_next_state, disturbance, active[:, None]

    def _validate_root_com(self):
        root_com = as_torch_tensor(self.robot.data.body_com_pos_b)[:, 0]
        if not torch.allclose(root_com, self._root_com_pos.expand_as(root_com), rtol=0.0, atol=1e-7):
            raise ValueError("World model push events require a fixed root COM shared by all environments")

    def _find_event_term(self, name):
        manager = getattr(self.env, "event_manager", None)
        if manager is None:
            return None
        for mode, names in manager._mode_term_names.items():
            for index, term_name in enumerate(names):
                if term_name == name:
                    return manager._mode_term_cfgs[mode][index].func
        return None

    def _gain_snapshot(self):
        actuator = self.robot.actuators[self._gain_name]
        try:
            from isaaclab.actuators.newton import read_group_parameter
            stiffness = read_group_parameter(self.robot.actuators, self._gain_name, "controller", "kp")
            damping = read_group_parameter(self.robot.actuators, self._gain_name, "controller", "kd")
        except (ImportError, AttributeError, KeyError, ValueError):
            stiffness = actuator.stiffness
            damping = actuator.damping
        return stiffness.detach().clone(), damping.detach().clone()

    @staticmethod
    def _event_default(term, name, fallback):
        values = getattr(term, name, None)
        if isinstance(values, dict):
            values = values.get("all_joints")
        return fallback if values is None else values.detach().clone()

    def xi(self):
        data = self.robot.data
        self._validate_root_com()
        joint_offset = getattr(self.env, "_ufo_joint_calibration_offset", None)
        if joint_offset is None:
            joint_offset = torch.zeros(
                self.env.num_envs, self.robot.num_joints, device=self.env.device, dtype=data.joint_pos.dtype
            )
        nominal_com = getattr(self.env, "_ufo_nominal_body_com_pos", None)
        if nominal_com is None:
            nominal_com = as_torch_tensor(data.body_com_pos_b).detach().clone()
            self.env._ufo_nominal_body_com_pos = nominal_com
        torso_com = as_torch_tensor(data.body_com_pos_b)[:, self.torso_body_ids, :].squeeze(1)
        nominal_torso_com = nominal_com[:, self.torso_body_ids, :].squeeze(1)
        # Match SuperMimic's simulator-context normalization. The nominal
        # values are zero, and each scale is tied to the corresponding
        # randomization range used by the UFO environment.
        lower, upper = self.friction_range
        friction = (self.friction.reshape(self.env.num_envs, -1) - (lower + upper) / 2.0) / ((upper - lower) / 2.0)
        mass = as_torch_tensor(data.body_mass)
        mass_term = self._find_event_term("scale_link_mass") or self._mass_term
        nominal_mass = self._event_default(mass_term, "default_mass", self._nominal_mass)
        mass_scale = (mass / nominal_mass.clamp_min(1e-6) - 1.0) / 0.2
        stiffness, damping = self._gain_snapshot()
        gain_term = self._find_event_term("scale_actuator_gains") or self._gain_term
        nominal_stiffness = self._event_default(
            gain_term, "default_actuator_stiffness", self._nominal_stiffness
        )
        nominal_damping = self._event_default(
            gain_term, "default_actuator_damping", self._nominal_damping
        )
        stiffness_scale = (stiffness / nominal_stiffness.clamp_min(1e-6) - 1.0) / 0.3
        damping_scale = (damping / nominal_damping.clamp_min(1e-6) - 1.0) / 0.3
        parts = [
            friction,
            joint_offset / 0.01,
            (torso_com - nominal_torso_com) / 0.05,
            mass_scale,
            stiffness_scale,
            damping_scale,
        ]
        xi = torch.cat(parts, dim=-1).detach().clone()
        if not torch.isfinite(xi).all():
            raise FloatingPointError("UFO world-model xi contains non-finite randomized parameters")
        return xi

    def checkpoint_contract(self):
        """Describe fixed simulator semantics that must agree when resuming a model."""
        action = self.env.action_manager.get_term("joint_pos")
        actuator = self.robot.actuators[self._gain_name]
        data = self.robot.data

        def values(value):
            if torch.is_tensor(value) or hasattr(value, "torch"):
                value = as_torch_tensor(value).detach().cpu()
                if value.ndim > 1:
                    value = value[0]
                return value.tolist()
            return value

        mass_term = self._find_event_term("scale_link_mass") or self._mass_term
        gain_term = self._find_event_term("scale_actuator_gains") or self._gain_term
        nominal_com = getattr(self.env, "_ufo_nominal_body_com_pos", data.body_com_pos_b)
        nominal_offset = getattr(action, "_ufo_nominal_offset", action._offset)
        # Newton exposes shape indices in its view order, not grouped by body.
        # Recording each body's indices pins the semantics of the friction axis.
        body_shapes = self.robot._root_view.body_shapes
        solver = self.env.cfg.sim.physics.solver_cfg
        solver_settings = {name: getattr(solver, name, None) for name in (
            "solver_type", "solver", "integrator", "cone", "impratio", "iterations", "ls_iterations",
            "tolerance", "update_data_interval", "njmax", "nconmax", "disable_contacts",
            "use_mujoco_contacts", "use_mujoco_cpu", "ccd_iterations", "enable_multiccd",
            "default_actuator_gear", "actuator_gears",
        )}
        return {
            "version": 2,
            "physics_dt": float(self.env.cfg.sim.dt),
            "decimation": int(self.env.cfg.decimation),
            "physics_type": f"{type(self.env.cfg.sim.physics).__module__}.{type(self.env.cfg.sim.physics).__qualname__}",
            "physics_num_substeps": int(self.env.cfg.sim.physics.num_substeps),
            "gravity": list(self.env.cfg.sim.gravity),
            "solver_type": f"{type(solver).__module__}.{type(solver).__qualname__}",
            "solver_settings": solver_settings,
            "joint_names": list(self.robot.joint_names),
            "body_names": list(self.robot.body_names),
            "action_joint_names": list(action._joint_names),
            "action_scale": values(action._scale),
            "action_offset": values(nominal_offset),
            "action_clip": action.cfg.clip,
            "action_rescale": self.action_rescale,
            "clip_actions": self.clip_actions,
            "actuator_type": f"{type(actuator).__module__}.{type(actuator).__qualname__}",
            "actuator_delay": [int(actuator.cfg.min_delay), int(actuator.cfg.max_delay)],
            "actuator_rfi_lim": float(actuator.cfg.rfi_lim),
            "effort_limit": values(actuator.effort_limit),
            "velocity_limit": values(actuator.velocity_limit),
            "default_joint_pos": values(data.default_joint_pos),
            "default_joint_vel": values(data.default_joint_vel),
            "default_mass": values(self._event_default(mass_term, "default_mass", self._nominal_mass)),
            "default_com": values(nominal_com),
            "default_stiffness": values(self._event_default(
                gain_term, "default_actuator_stiffness", self._nominal_stiffness
            )),
            "default_damping": values(self._event_default(
                gain_term, "default_actuator_damping", self._nominal_damping
            )),
            "friction_range": list(self.friction_range),
            "friction_shape_count": self.friction.shape[-1],
            "friction_shape_names": list(self.robot._root_view.shape_names),
            "friction_body_shapes_backend": [[int(shape) for shape in shapes] for shapes in body_shapes],
            "disturbance": "post_integration_world_com_velocity_overwrite",
        }

    def codec_kwargs(self):
        data = self.robot.data
        self._validate_root_com()
        policy = self.env.cfg.observations.policy
        terms = (policy.joint_pos, policy.joint_vel, policy.projected_gravity, policy.base_ang_vel)
        widths = (24, 24, 3, 3)
        scales, lower, upper = [], [], []
        for term, width in zip(terms, widths):
            if term.clip is not None or (term.scale is not None and not isinstance(term.scale, (float, int))):
                raise ValueError("World model requires scalar, unclipped policy observation scales")
            scale = 1.0 if term.scale is None else float(term.scale)
            scales.extend([scale] * width)
            noise = term.noise if policy.enable_corruption else None
            lower.extend([float(noise.n_min) if noise is not None else 0.0] * width)
            upper.extend([float(noise.n_max) if noise is not None else 0.0] * width)
        return dict(
            root_com_pos=self._root_com_pos.clone(),
            default_joint_pos=as_torch_tensor(data.default_joint_pos)[0].clone(),
            default_joint_vel=as_torch_tensor(data.default_joint_vel)[0].clone(),
            obs_scales=scales, noise_lower=lower, noise_upper=upper,
            action_rescale=self.action_rescale, clip_actions=self.clip_actions,
        )
