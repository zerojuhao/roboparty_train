"""RP1 UFO task using the shared RoboParty robot configuration."""

from isaaclab.utils.configclass import configclass

from .ufo_env_cfg import UFOEnvCfg


@configclass
class RP1UFOEnvCfg(UFOEnvCfg):
    pass
