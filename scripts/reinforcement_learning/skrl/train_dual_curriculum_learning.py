# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

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
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

import csv


class CustomCSVLogger:
    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"reward_log_{timestamp}.csv")

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "dual", "mean_reward", "mean_cost", "total_reward", "safety_prob"])
    
    def record(self, timestep: int, dual: float, mean_reward: float, mean_cost: float, total_reward: float, safety_prob: float) -> None:
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestep, dual, mean_reward, mean_cost, total_reward, safety_prob])

def get_unwrapped_env(env):
    """
    Iteratively unwrap the environment until you reach the base environment.
    """
    while hasattr(env, "env") or hasattr(env, "venv"):
        env = getattr(env, "env", getattr(env, "venv", env))
    return env

# PPO wrapper for Lagrangian dual updates
# This wrapper is used to train a PPO agent with a constraint cost in the environment
# The constraint cost is used to update a Lagrange multiplier (dual variable) that adjusts the reward function
class PPO_LagrangianWrapper:
    def __init__(self, runner, constraint_limit: float = 0.0, alpha_lambda: float = 1.0e-4, initial_lambda: float = 1.0):
        """
        runner: an instance of skrl Runner, which encapsulates the agent and environment.
        constraint_limit: the acceptable threshold for the constraint cost.
        alpha_lambda: learning rate for the dual update.
        initial_lambda: starting value for the Lagrange multiplier.
        """
        self.runner = runner
        self.env = get_unwrapped_env(runner._env)
        self.constraint_limit = constraint_limit
        self.alpha_lambda = alpha_lambda
        self.lambda_val = initial_lambda  # dual variable
        self.logger = CustomCSVLogger()  # Initialize a custom CSV logger
        self.total_timesteps_run = 0
        self.chunk = None

        # Set the dual multiplier in the environment so that the reward function uses it
        self.env._set_dual_multiplier(self.lambda_val)
        self.env._set_obstacle_size(0.05)
        self.logger.record(0, self.lambda_val, 0.0, 0.0, 0.0, 0.0)

    def compute_total_constraint_cost(self) -> float:
        """
        Compute the average constraint cost over all environments.
        Assumes that _get_constraint_cost() returns a tensor of shape [num_envs, 1].
        """
        constraint_costs = self.env._get_constraint_cost()  # shape: (num_envs, 1)
        constraint_costs = constraint_costs / self.chunk
        return constraint_costs.mean().item()

    def update_dual(self):
        """
        Update the dual variable (Lagrange multiplier) based on the current constraint cost.
        """
        # constraint_cost = self.env._get_last_episode_costs().mean().item()
        constraint_cost = self.compute_total_constraint_cost()
        # Dual update (gradient ascent) with projection onto non-negative values:
        self.lambda_val = max(0.0, self.lambda_val + self.alpha_lambda * (constraint_cost - self.constraint_limit))
        # Update the environment with the new dual value:
        self.env._set_dual_multiplier(self.lambda_val)
        print(f"Dual updated: {self.lambda_val:.4f} (constraint cost: {constraint_cost:.4f}, limit: {self.constraint_limit:.4f})")

    def run(self, total_timesteps: int, chunk_timesteps: int):
        """
        Run training in chunks, updating the dual variable after each chunk.
        """
        self.chunk = chunk_timesteps
        print(f"Chunk timesteps: {self.chunk}")
        self.start_time = datetime.now()
        while self.total_timesteps_run < total_timesteps:

            # reset reward and cost accumulators
            self.env._reset_reward_and_cost()

            # Run training for a chunk
            self.runner.run()
            self.total_timesteps_run += chunk_timesteps

            r, c = self.env._get_reward_and_cost()
            mean_reward = r.mean().item()
            mean_cost = c.mean().item()
            safety_prob = self.env._get_safety_probability().item()
            total_reward = mean_reward - self.lambda_val * mean_cost

            # Update the dual variable
            self.update_dual()
            #self.env._domain_randomize()

            # Log current timestep and statistics
            self.logger.record(self.total_timesteps_run, self.lambda_val, mean_reward, mean_cost, total_reward, safety_prob)
            progress = self.total_timesteps_run / total_timesteps * 100
            print(f"Progress: {progress:.2f}%")
            print("Expected time remaining: ", (datetime.now() - self.start_time) * (total_timesteps - self.total_timesteps_run) / self.total_timesteps_run)
            if int(progress) % 10 == 0 and int(progress) > 0:
                print("Increasing obstacle size")
                self.env._change_obstacle_size(int(progress)/100)

        return self.runner



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    print(f"[INFO] Starting creation of the environment")
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"[INFO] Environment created")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    print("[INFO] Runner created.")

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    c_limit = agent_cfg["lagrangian"]["cost_limit"]
    print(f"[INFO] Constraint limit: {c_limit}")
    lr_lambda = agent_cfg["lagrangian"]["learning_rate_lambda"]
    print(f"[INFO] Learning rate for dual update: {lr_lambda}")
    init_l = agent_cfg["lagrangian"]["initial_lambda"]
    print(f"[INFO] Initial value for dual variable: {init_l}")
    # wrap the Runner with PPO-Lagrangian wrapper
    lagrangian_agent = PPO_LagrangianWrapper(runner, constraint_limit=c_limit, alpha_lambda=lr_lambda, initial_lambda=init_l)
    print("[INFO] PPO-Lagrangian wrapper created.")

    chunk_timesteps = agent_cfg["trainer"]["timesteps"]
    total_timesteps = agent_cfg["trainer"]["total_timesteps"]
    print(f"[INFO] Running training with {total_timesteps} timesteps in chunks of {chunk_timesteps} timesteps")
    init_time = datetime.now()
    # Run training with the wrapper for a total number of timesteps
    runner_trained = lagrangian_agent.run(total_timesteps=total_timesteps, chunk_timesteps=chunk_timesteps)

    # Optionally, save the final model
    save_path = os.path.join(log_dir, "final_model.zip")
    print(f"[INFO] Saving final model to: {save_path}")
    #runner.agent.save(os.path.join(agent_cfg["agent"]["experiment"]["experiment_name"], "final_model.zip"))
    runner_trained.agent.save(save_path)
    print(f"[INFO] Training completed in {datetime.now() - init_time}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
