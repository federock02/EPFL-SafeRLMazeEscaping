"""
JetBot RL environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-JetBot-RL-Direct-v0",
    # entry point must be the class name of the environment, in the _env.py file in the same folder
    entry_point=f"{__name__}.jetbot_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Maze-RL-Direct-v0",
    entry_point=f"{__name__}.jetbot_maze_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_maze_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Maze-RL-Direct-v1",
    entry_point=f"{__name__}.jetbot_maze_no_obs_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_maze_no_obs_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)