"""PUFO visual FB-CPR-Aux task, isolated from terrain task."""

import gymnasium as gym

from . import agents

gym.register(
    id="RP1-PUFO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pufo_env_cfg:RP1PUFOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rp1_pufo_agent_cfg:RP1PUFORunnerCfg",
    },
)
