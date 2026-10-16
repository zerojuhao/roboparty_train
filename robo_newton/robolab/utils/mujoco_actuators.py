"""CPU MuJoCo adapter for the same RoboParty actuator used during training."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from isaaclab.actuators._compat import _resolve_limit_aliases
from isaaclab.actuators.actuator_base import resolve_joint_parameter
from isaaclab.utils.types import ArticulationActions
from robolab.assets.robots.roboparty_actuators import RoboPartyActuator, RoboPartyActuatorCfg


class RoboPartyMujocoActuator:
    """One explicitly ordered actuator group, evaluated on CPU without a simulator.

    Config values remain unchanged. Initialization follows ActuatorCollection:
    normalize old aliases first, then pass authored effort and resolved joint
    velocity defaults to RoboPartyActuator. Nominal evaluation disables delay/RFI.
    """

    def __init__(self, urdf, names, config, *, nominal=True):
        self.names = list(names)
        if not self.names or len(self.names) != len(set(self.names)):
            raise ValueError("Motor joint names must be nonempty and unique")
        values = deepcopy(config if isinstance(config, dict) else config.to_dict())
        if "RoboPartyActuator" not in str(values.pop("class_type", "")):
            raise ValueError("RP1 sim2sim requires a RoboPartyActuator configuration")
        cfg = RoboPartyActuatorCfg(**values)
        _resolve_limit_aliases("sim2sim", cfg, self.names, warn_deprecated=False)
        if nominal:
            cfg.min_delay = cfg.max_delay = 0
            cfg.rfi_lim = 0.0
        joints = {j.get("name"): j for j in ET.parse(Path(urdf)).getroot().findall("joint")}
        defaults = {}
        for field in ("effort", "velocity"):
            defaults[field] = torch.tensor([[float(joints[n].find("limit").get(field)) for n in self.names]])
        speed = resolve_joint_parameter(cfg.joint_velocity_limit, defaults["velocity"], self.names, 1, "cpu")
        self.actuator = RoboPartyActuator(
            cfg, joint_names=self.names, joint_ids=slice(None), num_envs=1, device="cpu",
            actuator_effort_limit=defaults["effort"], actuator_velocity_limit=speed,
            stiffness=0.0, damping=0.0,
        )
        self.actuator.reset(None)
        if not all(torch.isfinite(value).all() for value in (
            self.actuator.actuator_effort_limit, self.actuator.actuator_velocity_limit,
            self.actuator.stiffness, self.actuator.damping,
        )):
            raise ValueError("Motor parameters must be finite")

    @property
    def effort_limit(self):
        return self.actuator.actuator_effort_limit[0].numpy().astype(float)

    @property
    def velocity_limit(self):
        return self.actuator.actuator_velocity_limit[0].numpy().astype(float)

    @property
    def keep_velocity(self):
        return self.actuator._keep_velocity[0].numpy().astype(float)

    @torch.inference_mode()
    def reset(self):
        self.actuator.reset(None)

    def bind(self, model):
        """Set passive joint properties and return name-mapped torque actuator IDs."""
        joints = np.asarray([model.joint(name).id for name in self.names])
        dofs = model.jnt_dofadr[joints]
        ids = []
        for joint in joints:
            matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
            if len(matches) != 1:
                raise ValueError(f"Expected one torque actuator for {model.joint(joint).name}")
            ids.append(matches[0])
        ids = np.asarray(ids)
        if not np.allclose(model.actuator_gear[ids, 0], 1):
            raise ValueError("Expected unit torque actuator gears")
        for config_name, model_name in (("armature", "dof_armature"),
                                         ("viscous_friction", "dof_damping"),
                                         ("friction", "dof_frictionloss")):
            target = getattr(model, model_name)
            values = resolve_joint_parameter(getattr(self.actuator.cfg, config_name),
                                             torch.tensor(target[dofs][None]), self.names, 1, "cpu")
            target[dofs] = values[0].numpy()
        model.actuator_ctrllimited[ids] = 0
        model.actuator_forcelimited[ids] = 0
        return ids

    @torch.inference_mode()
    def compute(self, target_pos, joint_pos, joint_vel):
        def tensor(value):
            array = np.asarray(value)
            if array.shape != (len(self.names),) or not np.isfinite(array).all():
                raise ValueError(f"Expected finite motor input with shape {(len(self.names),)}")
            return torch.as_tensor(array, dtype=torch.float32).reshape(1, -1)

        target, pos, vel = map(tensor, (target_pos, joint_pos, joint_vel))
        action = ArticulationActions(joint_positions=target, joint_velocities=torch.zeros_like(target),
                                     joint_efforts=torch.zeros_like(target))
        result = self.actuator.compute(action, pos, vel).joint_efforts[0].numpy().astype(float)
        if not np.isfinite(result).all():
            raise FloatingPointError("RoboPartyActuator produced non-finite torque")
        return result
