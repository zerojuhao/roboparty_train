# UFO (FB-CPR-Aux) tracking task, ported from the humanoidverse/UFO training stack.
# Motion handling is vendored under `ufo.motion` (original UFO logic).

import gymnasium as gym

from . import agents

gym.register(
    id="RP1-UFO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp1_ufo_env_cfg:RP1UFOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rp1_ufo_agent_cfg:RP1UFORunnerCfg",
    },
)
