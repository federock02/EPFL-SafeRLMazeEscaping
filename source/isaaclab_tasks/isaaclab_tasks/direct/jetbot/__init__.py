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
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-RL-Direct-v1",
    entry_point=f"{__name__}.jetbot_lagrangian_test_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_lagrangian_test_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Prob-Dual-RL-Direct-v0",
    entry_point=f"{__name__}.jetbot_lagrangian_prob_test_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_lagrangian_prob_test_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_lagrangian_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Dual-RL-Direct-v0",
    entry_point=f"{__name__}.jetbot_lagrangian_test_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_lagrangian_test_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_lagrangian_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Moving-Dual-RL-Direct-v0",
    entry_point=f"{__name__}.jetbot_moving_obs_test_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_moving_obs_test_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_lagrangian_cfg.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Maze-RL-Direct-v0",
    entry_point=f"{__name__}.jetbot_maze_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_maze_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_maze.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_maze.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Maze-RL-Direct-v1",
    entry_point=f"{__name__}.jetbot_maze_no_obs_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_maze_no_obs_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_maze.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_maze.yaml",
    },
)

gym.register(
    id="Isaac-JetBot-Maze-RL-Direct-v2",
    entry_point=f"{__name__}.jetbot_maze_lagrangian_env:JetBotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_maze_lagrangian_env:JetBotEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_maze.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_maze.yaml",
    },
)