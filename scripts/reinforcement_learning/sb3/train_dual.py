# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--load", type=str, default=None, help="Load a model from a file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from typing import Callable

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import csv


class CustomCSVLogger:
    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"reward_log_{timestamp}.csv")

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "dual", "mean_reward", "mean_cost", "total_reward"])
    
    def record(self, timestep: int, dual: float, mean_reward: float, mean_cost: float, total_reward: float) -> None:
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestep, dual, mean_reward, mean_cost, total_reward])




# PPO wrapper for Lagrangian dual updates
# This wrapper is used to train a PPO agent with a constraint cost in the environment
# The constraint cost is used to update a Lagrange multiplier (dual variable) that adjusts the reward function
class PPO_LagrangianWrapper:
    def __init__(self, ppo_agent: PPO, env, constraint_limit: float, alpha_lambda: float, initial_lambda: float = 1.0):
        """
        ppo_agent: an instance of Stable Baselines3 PPO.
        env: your environment instance (which should support dual updates).
        constraint_limit: the acceptable threshold for the constraint cost.
        alpha_lambda: learning rate for the dual update.
        initial_lambda: starting value for the Lagrange multiplier.
        """
        self.ppo = ppo_agent
        self.env = get_unwrapped_env(env)
        self.constraint_limit = constraint_limit
        self.alpha_lambda = alpha_lambda
        self.lambda_val = initial_lambda
        self.logger = CustomCSVLogger()
        self.p2dratio = 1

        self.env._set_dual_multiplier(self.lambda_val)
        self.logger.record(0, self.lambda_val, 0.0, 0.0, 0.0)

    def update_dual(self):
        """
        Get the current constraint cost from the environment and update lambda.
        """
        # Ensure the environment provides a method to return the constraint cost
        constraint_cost = self.compute_total_constraint_cost()  # expected to be a scalar (float)
        # Dual update: increase lambda if constraint is violated, decrease otherwise.
        self.lambda_val = max(0.0, self.lambda_val + self.alpha_lambda * (constraint_cost - self.constraint_limit))
        # Pass the updated dual multiplier to the environment (so it adjusts the reward function)
        self.env._set_dual_multiplier(self.lambda_val)
        print(f"Dual variable updated to {self.lambda_val:.4f}, constraint cost: {constraint_cost:.4f} with constraint limit: {self.constraint_limit:.4f}")

    def learn(self, callback, total_timesteps: int, base: int = 1310720, p2dratio: int=1, **kwargs):
        """
        Run training in chunks, updating the dual variable after each chunk.
        """
        self.p2dratio = p2dratio
        chunk_timesteps = base*p2dratio
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:
            # Train PPO for a chunk of timesteps
            self.env._reset_reward_and_cost()
            self.ppo.learn(chunk_timesteps, reset_num_timesteps=False, callback=callback, **kwargs)
            timesteps_so_far += chunk_timesteps
            r, c  = self.env._get_reward_and_cost()
            r = r.mean().item()
            c = c.mean().item()
            # After each chunk, update the dual variable
            self.update_dual()
            self.logger.record(timesteps_so_far, self.lambda_val, r, c, r - self.lambda_val * c)
            print(f"Training progress: {timesteps_so_far/total_timesteps*100:.2f}%")

        return self.ppo

    def set_logger(self, logger):
        self.ppo.set_logger(logger)
    
    def save(self, path):
        self.ppo.save(path)

    def compute_total_constraint_cost(self):
        """
        Get the average constraint cost across all parallel environments.
        """
        constraint_costs = self.env._get_constraint_cost()  # Shape: (num_envs,)
        constraint_costs = constraint_costs / self.p2dratio
        return constraint_costs.mean().item()  # Average over environments



@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    learning_rate = agent_cfg["learning_rate"]

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    decaying_learning_rate = agent_cfg.pop("decaying_learning_rate", False)
    if decaying_learning_rate:
        callable_lr = linear_schedule(learning_rate)
        agent_cfg["learning_rate"] = callable_lr
    else:
        agent_cfg["learning_rate"] = learning_rate

    num_envs = env_cfg.scene.num_envs
    num_steps = agent_cfg["n_steps"]
    base = num_envs * num_steps

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )



    # create agent from stable baselines
    if args_cli.load is None:
        agent = PPO_LagrangianWrapper(PPO(policy_arch, env, verbose=1, **agent_cfg), env, constraint_limit=1.0, alpha_lambda=1e-4)
    else:
        model = PPO.load(args_cli.load, env=env, verbose=1, **agent_cfg)
        model.learning_rate = 1e-4
        print("Learning rate: ", model.learning_rate)
        agent = PPO_LagrangianWrapper(model, env, constraint_limit=0.0, alpha_lambda=1e-4)
    # agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    #new_logger = configure(log_dir, ["tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    agent.learn(total_timesteps=n_timesteps, base=base, p2dratio=5, callback=checkpoint_callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()

def get_unwrapped_env(env):
    while hasattr(env, "env") or hasattr(env, "venv"):
        env = getattr(env, "env", getattr(env, "venv", env))
    return env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return schedule


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
