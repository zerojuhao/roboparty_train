"""AMP command generators."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


class SingleAxisVelocityCommand(UniformVelocityCommand):
    """Sample exactly one of forward, lateral, or yaw velocity per environment."""

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids] = 0.0
        axis = torch.randint(0, 3, (len(env_ids),), device=self.device)
        values = (
            r.uniform_(*self.cfg.ranges.lin_vel_x),
            r.uniform_(*self.cfg.ranges.lin_vel_y),
            r.uniform_(*self.cfg.ranges.ang_vel_z),
        )
        for index, value in enumerate(values):
            selected = env_ids[axis == index]
            if len(selected):
                self.vel_command_b[selected, index] = value[axis == index]
        self.is_standing_env[env_ids] = False


@configclass
class SingleAxisVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type[SingleAxisVelocityCommand] = SingleAxisVelocityCommand
