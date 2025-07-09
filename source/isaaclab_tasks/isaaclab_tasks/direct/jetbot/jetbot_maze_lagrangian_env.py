from __future__ import annotations

import math
import torch
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Sequence

from isaaclab_assets.robots.jetbot import JETBOT_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

import yaml
import os
import csv
from datetime import datetime
import random
import numpy as np
from typing import Tuple

import csv

class CustomCSVLogger:
    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"iter_reward_log_{timestamp}.csv")
        self.header = ["dual", "dist_reward", "dr_std", "goal_reward", "gr_std", "term_reward", "tr_std", "expl_reward", "expl_std",
                       "mean_reward", "std_reward", "mean_cost", "std_cost", "total_reward", "safety_prob"]
        self._write_header()

    def _write_header(self):
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def record(self, dual: float, dist_reward: float, dist_std: float, goal_reward: float, goal_std: float,
               term_reward: float, term_std: float, expl_reward: float, expl_std: float, mean_reward: float,
               std_reward: float, mean_cost: float, std_cost: float, total_reward: float, safety_prob: float) -> None:
        data = [dual, dist_reward, dist_std, goal_reward, goal_std, term_reward, term_std,
                expl_reward, expl_std, mean_reward, std_reward, mean_cost, std_cost, total_reward, safety_prob]
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)

def get_unwrapped_env(env):
    """
    Recursively unwraps the environment until the base environment is reached.
    Handles 'env' or 'venv' attributes.
    """
    current_env = env
    while hasattr(current_env, "env") or hasattr(current_env, "venv"):
        current_env = getattr(current_env, "env", getattr(current_env, "venv", current_env))
    return current_env

@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation: int = 4
    episode_length_s: float = 120.0
    action_scale: float = 1.0
    action_space: int = 2
    observation_space: int = 12  # x, y, distance_to_goal, yaw_diff, base_lin_vel, base_ang_vel, 3*distance_to_obstacle, 3*angle_diff
    state_space: int = 0


    # simulation
    # keep small dt for smooth simulation, increase decimation, step is done every dt * decimation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation) # original dt = 1 / 120

    # robot
    # the path represent the position of the robot prim in the hierarchical representation of the scene
    # using regex to match the robot prim in all environments for multi-environment training
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/JetBot")
    left_wheel_joint_name = "left_wheel_joint"
    right_wheel_joint_name = "right_wheel_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)

    # Reset
    max_distance_from_goal: float = 3.5
    spawn_height: float = 0.0437

    # Discretization
    grid_boundaries = (-1.5, 1.5, -3.0, 3.0)  # x_min, x_max, y_min, y_max
    grid_cell_size: float = 0.5

    # World
    wall_width: float = 0.05
    goal_pos = [[-2.0,1.0], [-2.0,0.0], [2.0,0.0], [0.5,3.5], [-0.5,-3.5]]

    # Robot Geometry
    wheelbase: float = 0.12
    wheelradius: float = 0.032

class JetBotEnv(DirectRLEnv):
    cfg: JetBotEnvCfg

    def __init__(self, cfg: JetBotEnvCfg, render_mode: str | None = None, **kwargs):
        self.maze = Maze()
        super().__init__(cfg, render_mode, **kwargs)

        # use the joint names from the configuration to find indices in the articulation
        self._left_wheel_idx, _ = self.jetbot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_wheel_idx, _ = self.jetbot.find_joints(self.cfg.right_wheel_joint_name)
        self.action_scale = self.cfg.action_scale

        # cache joint positions and velocities data for efficiency
        self.joint_pos = self.jetbot.data.joint_pos
        self.joint_vel = self.jetbot.data.joint_vel

        # previous distance to goal, used in reward computation
        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        # gamma for reward computation
        config_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_cfg_maze.yaml")
        print("Config path: ", config_path)
        try:
            with open(config_path) as stream:
                data = yaml.safe_load(stream)
                self.gamma = data["agent"]["discount_factor"]
                self.reward_scale_goal = data["rewards"]["goal"]
                self.reward_scale_distance = data["rewards"]["distance"]
                self.reward_scale_terminated = data["rewards"]["termination"]
                self.cost_obstacle = data["costs"]["obstacle"]
                self.min_distance_to_obstacle = data["min_dist"]
                self.reward_scale_exploration = data["rewards"].get("exploration_reward")
                self.penalty_scale_exploration = data["rewards"].get("exploration_penalty")
        except yaml.YAMLError as exc:
            print(f"Error loading YAML config: {exc}. Using default reward/cost parameters.")
            self.gamma = 0.99
            self.reward_scale_goal = 60.0
            self.reward_scale_distance = 10.0
            self.reward_scale_terminated = -50.0
            self.cost_obstacle = 6.0
            self.min_distance_to_obstacle = 0.15
            self.reward_scale_exploration = None
            self.penalty_scale_exploration = 0.0

        print("Gamma: ", self.gamma)
        print("Reward scale goal: ", self.reward_scale_goal)
        print("Reward scale distance: ", self.reward_scale_distance)
        print("Reward scale terminated: ", self.reward_scale_terminated)
        print("Cost obstacle: ", self.cost_obstacle)
        print("Min dist to obstacle: ", self.min_distance_to_obstacle)
        print("Exploration reward: ", self.reward_scale_exploration)
        print("Exploration penalty: ", self.penalty_scale_exploration)

        self.dual_multiplier = 0.0
        self.n_envs = self.cfg.scene.num_envs

        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.episode_costs = torch.zeros(self.num_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.num_envs, device=self.device)
        self.safety_indicator = torch.ones(self.num_envs, device=self.device)

        if self.reward_scale_exploration is not None:
            #self.exploration_tracker = ExplorationTracker(self.cfg.scene.num_envs, self.device, self.cfg.grid_boundaries,
                                                      #self.exploration_reward, self.exploration_penalty, cell_size=0.3)
            
            num_cells = int((self.cfg.grid_boundaries[1] - self.cfg.grid_boundaries[0]) / self.cfg.grid_cell_size) * int((self.cfg.grid_boundaries[3] - self.cfg.grid_boundaries[2]) / self.cfg.grid_cell_size)
            print("Number of cells: ", num_cells)
            self.grids = torch.zeros(self.cfg.scene.num_envs, num_cells, device=self.device)

        self.logger = CustomCSVLogger()
        self.i = 0

        self.spawned_robots = 0
        self.failed_robots = 0

        self.action_accumulation = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.action_count = 0

    def _setup_scene(self):
        self.jetbot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add a cube as a goal
        goal_size = (0.5, 0.5, 0.25)
        for i, g in enumerate(self.cfg.goal_pos):
            goal_pos = list(g) + [goal_size[2] / 2.0]
            cfg_goal_cube = sim_utils.CuboidCfg(
                size=goal_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.2, 0.92)),
            )
            cfg_goal_cube.func(f"/World/Objects/goal_{i}", cfg_goal_cube, translation=tuple(goal_pos))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["jetbot"] = self.jetbot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.maze.spawn_maze(width=self.cfg.wall_width, height=0.5)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print("Raw actions: ", actions)
        self.actions = actions.clone()
        # store actions over multiple steps
        self.action_accumulation += self.actions
        self.action_count += 1
        # print statistics of accumulated actions
        # mean
        mean_actions = self.action_accumulation.mean(dim=0) / self.action_count
        # std
        std_actions = self.action_accumulation.std(dim=0) / self.action_count
        print("Mean actions: ", mean_actions)
        print("Std actions: ", std_actions)
        # self.actions = torch.tanh(self.actions) * 6.0  # Ensure actions are in [-6, 6] range

    def _apply_action(self) -> None:
        left_action = self.actions[:, 0].unsqueeze(-1)
        right_action = self.actions[:, 1].unsqueeze(-1)
        #print("Left action: ", left_action[0,0].item())
        #print("Right action: ", right_action[0,1].item())
        print("Left action: ", left_action, "\nRight action: ", right_action)
        self.jetbot.set_joint_velocity_target(left_action, joint_ids=self._left_wheel_idx)
        self.jetbot.set_joint_velocity_target(right_action, joint_ids=self._right_wheel_idx)

    def _get_observations(self) -> dict:
        # Get robot position (x, y)
        position = self.jetbot.data.root_pos_w[:, :2]

        # Compute robot yaw
        orientation = self.jetbot.data.root_quat_w
        yaw = self._compute_yaw(orientation).squeeze(-1)

        # Check collisions and get distances and angles to obstacles
        distance_to_obstacles, obstacle_angle_difference = self.maze.check_collision_ang(position, yaw.clone())

        # Compute safety indicator
        too_close = (distance_to_obstacles < self.min_distance_to_obstacle).any(dim=1)
        self.safety_indicator[too_close] = 0

        # Get wheel velocities
        left_wheel_velocity = self.joint_vel[:, self._left_wheel_idx].squeeze(-1).unsqueeze(-1)
        right_wheel_velocity = self.joint_vel[:, self._right_wheel_idx].squeeze(-1).unsqueeze(-1)

        # Compute distance to goal and yaw difference to goal using LOS check
        # Pass robot_pos (shape [num_envs, 2]) and yaw (shape [num_envs])
        distance_to_goal_los, yaw_difference_los = self._get_distance_and_angle_from_goal_with_los_check(position, yaw, check_los=False)
        # print("Position: ", position.shape)
        # print("Distance to goal: ", distance_to_goal_los.shape)
        # print("Yaw difference: ", yaw_difference_los.shape)
        # print("Left wheel velocity: ", left_wheel_velocity.shape)
        # print("Right wheel velocity: ", right_wheel_velocity.shape)
        # print("Distance to obstacles: ", distance_to_obstacles.shape)
        # print("Obstacle angle difference: ", obstacle_angle_difference.shape)

        #print("Left wheel velocity observed: ", left_wheel_velocity)
        #print("Right wheel velocity observed: ", right_wheel_velocity)

        # Concatenate observations
        observation = torch.cat((
            position,                 # [num_envs, 2]
            distance_to_goal_los,     # [num_envs, 1]
            yaw_difference_los,       # [num_envs, 1]
            left_wheel_velocity,      # [num_envs, 1]
            right_wheel_velocity,     # [num_envs, 1]
            distance_to_obstacles,    # [num_envs, 3]
            obstacle_angle_difference,# [num_envs, 3]
        ), dim=-1)
        #print("Observation: ", observation[0,:])

        return {"policy": observation}
    
    def _get_rewards(self) -> torch.Tensor:
        self.i += 1
        # Compute distances to obstacles
        distances_to_obstacles = self.maze.check_collision(self.jetbot.data.root_pos_w[:, :2])

        # Compute the minimum distance to goal with line-of-sight check
        current_distance_to_goal = self._get_distance_from_goal_with_los_check(check_los=False)

        is_new_cell = None

        if self.reward_scale_exploration is not None:
            current_robot_positions_xy = self.jetbot.data.root_pos_w[:, :2]  # (num_envs, 2)
            grid_pos = self._get_cell(current_robot_positions_xy).squeeze(-1)  # (num_envs,)

            # Get current values at those grid cells
            grid_values = self.grids[torch.arange(self.cfg.scene.num_envs, device=self.device), grid_pos]  # (num_envs,)

            # Detect which envs are visiting a new cell (value was 0)
            is_new_cell = (grid_values == 0)

            # Set those grid cells to 1 (mark as visited)
            self.grids[torch.arange(self.cfg.scene.num_envs, device=self.device)[is_new_cell], grid_pos[is_new_cell]] = 1
            #print("Grid 0 new: ", self.grids[0,:])
            
            #print("Grid position: ", self.grids[0,:])

        # Compute individual reward components and total primary reward
        distance_reward, goal_reward, termination_reward, exploration_reward, primary_reward = compute_rewards(
            self.reward_scale_terminated,
            self.reward_scale_goal,
            self.reward_scale_distance,
            self.cfg.max_distance_from_goal,
            self.jetbot.data.root_pos_w[:, :2],  # Robot root position (x, y)
            self.previous_distance_to_goal,  # Previous distance to goal
            current_distance_to_goal,  # Current distance to goal
            self.reward_scale_exploration,
            self.penalty_scale_exploration,
            is_new_cell,  # New cell value for exploration
        )
        # Update previous distance to goal for the next step
        self.previous_distance_to_goal = current_distance_to_goal
        
        # Compute constraint cost based on distance to obstacles
        constraint_cost = compute_cost(
            self.cost_obstacle,
            self.min_distance_to_obstacle + self.cfg.wheelbase / 2.0,
            distances_to_obstacles,
        )
        self._computed_constraint_cost = constraint_cost

        # Update episode rewards and costs
        self.episode_rewards += primary_reward
        self.episode_costs += constraint_cost
        self.last_episode_costs = constraint_cost

        # Calculate mean and standard deviation of reward components and total reward/cost
        mean_distance_reward = distance_reward.mean().item()
        mean_goal_reward = goal_reward.mean().item()
        mean_termination_reward = termination_reward.mean().item()
        std_distance_reward = distance_reward.std().item()
        std_goal_reward = goal_reward.std().item()
        std_termination_reward = termination_reward.std().item()
        mean_exploration_reward = exploration_reward.mean().item() if exploration_reward is not None else 0.0
        std_exploration_reward = exploration_reward.std().item() if exploration_reward is not None else 0.0

        mean_total_reward = primary_reward.mean().item()
        std_total_reward = primary_reward.std().item()
        mean_total_cost = constraint_cost.mean().item()
        std_total_cost = constraint_cost.std().item()

        # Get safety probability
        safety_probability = self._get_safety_probability().item()

        # Log the reward and cost information
        total_lagrangian_reward = mean_total_reward - self.dual_multiplier * mean_total_cost
        #print("Total Lagrangian reward: ", total_lagrangian_reward)
        self.logger.record(self.dual_multiplier, mean_distance_reward, std_distance_reward,
                           mean_goal_reward, std_goal_reward, mean_termination_reward, std_termination_reward,
                           mean_exploration_reward, std_exploration_reward,
                           mean_total_reward, std_total_reward, mean_total_cost, std_total_cost,
                           total_lagrangian_reward, safety_probability)

        # Return the Lagrangian reward
        lag_rew = primary_reward - self.dual_multiplier * constraint_cost
        #print("Lagrangian reward: ", lag_rew)
        return lag_rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # check if the episode has run out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # check if the robot has gone out of bounds
        out_of_bounds = torch.any(torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal, dim=1)

        self.failed_robots += torch.sum(out_of_bounds).item()
        self.failed_robots += torch.sum(time_out).item()

        # check if the robot has reached the goal
        robot_pos = self.jetbot.data.root_pos_w[:, :2]
        # convert the list of goal positions into a tensor (shape: [num_goals, 2])
        all_goals = torch.tensor(self.cfg.goal_pos, device=self.jetbot.data.root_pos_w.device)  
        # pairwise differences between each robot and each goal
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        # Compute Euclidean distances along the last dimension: [num_envs, num_goals]
        dists = torch.norm(diff, dim=2)
        # Check if any goal is within 0.1 distance for each robot: [num_envs] boolean tensor
        reached_goal = torch.any(dists <= 0.3, dim=1)

        # episode is done if any of the conditions are met
        return out_of_bounds | reached_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.jetbot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.reward_scale_exploration is not None:
            # Reset the exploration tracker for the environments being reset
            # self.exploration_tracker.reset(env_ids)
            # Reset the grid for the environments being reset
            self.grids[env_ids] = 0


        # sample a new starting position within initial_pose_radius around the goal
        num_ids = len(env_ids)

        self.spawned_robots += num_ids

        # vectorized sample
        safe_positions = self.maze.sample_safe_positions(
            num_envs=num_ids,
            x_bounds=(-1.5,1.5),
            y_bounds=(-3.0,3.0),
            min_dist=self.min_distance_to_obstacle,
            max_attempts=50,
            safety_margin=0.01
        )
        # print(f"Sampled {num_ids} safe positions: {safe_positions}")
        # force safe position to be (0.1, 0.24) for all environments

        # set spawn height as defined in cfg (e.g., self.cfg.spawn_height)
        default_root_state = self.jetbot.data.default_root_state[env_ids]
        default_root_state[:, :2] = safe_positions
        default_root_state[:, 2] = self.cfg.spawn_height

        # sample a new starting yaw angle
        axis = torch.zeros(num_ids, 3, device=self.device)
        axis[:, 2] = 1.0 # rotation around z-axis
        angle = torch.rand(num_ids, device=self.device) * 2 * math.pi
        half_angle = angle / 2
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)
        qx = axis[:, 0] * sin_half_angle # already zeros
        qy = axis[:, 1] * sin_half_angle # already zeros
        qz = axis[:, 2] * sin_half_angle
        qw = cos_half_angle
        new_orientation = torch.stack([qw, qx, qy, qz], dim=1)

        self.previous_distance_to_goal[env_ids] = torch.zeros(num_ids, device=self.device)

        # update the root state
        default_root_state[:, 3:7] = new_orientation

        self.jetbot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # reset safety indicator for the reset environments
        self.safety_indicator[env_ids] = 1
    
    def _get_distance_from_goal_with_los_check(self, check_los=True) -> torch.Tensor:
        """
        Computes the minimum distance to goals, considering line-of-sight to maze walls.
        If line-of-sight to a goal is blocked by a wall, that goal's distance is treated as 10.0.
        Returns a tensor of shape [num_envs] with the minimum modified distance for each environment.
        """
        robot_pos = self.jetbot.data.root_pos_w[:, :2]  # Shape: [num_envs, 2]
        # Ensure all_goals is float32 and on the correct device
        all_goals = torch.tensor(self.cfg.goal_pos, device=robot_pos.device, dtype=torch.float32) # Shape: [num_goals, 2]

        num_envs = robot_pos.shape[0]
        num_goals = all_goals.shape[0]

        # If there are no goals, return a high distance for all environments
        if num_goals == 0:
            return torch.full((num_envs,), 10.0, device=robot_pos.device, dtype=torch.float32)

        # Calculate Euclidean distances from each robot to each goal
        # robot_pos_exp shape: [num_envs, 1, 2]
        # all_goals_exp shape: [1, num_goals, 2]
        # diff shape: [num_envs, num_goals, 2]
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        euclidean_dists = torch.norm(diff, dim=2)  # Shape: [num_envs, num_goals]

        if check_los:
            # Prepare line segments for LOS check: from each robot to each goal
            # line_starts_flat shape: [num_envs * num_goals, 2]
            line_starts_flat = robot_pos.unsqueeze(1).expand(-1, num_goals, -1).reshape(num_envs * num_goals, 2)
            # line_ends_flat shape: [num_envs * num_goals, 2]
            line_ends_flat = all_goals.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_goals, 2)

            # Check if these line segments are blocked by any maze walls
            # is_blocked_flat shape: [num_envs * num_goals] (boolean)
            if self.maze.walls_start_tensor is not None and self.maze.walls_start_tensor.shape[0] > 0:
                is_blocked_flat = self.maze.are_line_segments_blocked_by_walls(line_starts_flat, line_ends_flat)
            else: # No walls defined in the maze, so no segments are blocked
                is_blocked_flat = torch.zeros(num_envs * num_goals, dtype=torch.bool, device=robot_pos.device)
            
            # Reshape the blocked status to [num_envs, num_goals]
            is_blocked_matrix = is_blocked_flat.view(num_envs, num_goals)

            # Define the penalty distance for blocked LOS
            penalty_distance = torch.tensor(10.0, device=robot_pos.device, dtype=torch.float32)

            # Apply penalty: if LOS is blocked, use penalty_distance, otherwise use Euclidean distance
            modified_dists = torch.where(is_blocked_matrix, penalty_distance, euclidean_dists)

            # Find the minimum of these modified distances for each robot across all goals
            min_final_dists, _ = torch.min(modified_dists, dim=1)  # Shape: [num_envs]
        
        else:
            # Not checking LOS, find the minimum of direct Euclidean distances
            min_final_dists, _ = torch.min(euclidean_dists, dim=1) # Shape: [num_envs]

        return min_final_dists
    
    def _get_distance_and_angle_from_goal_with_los_check(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor, check_los=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the minimum distance to goals and yaw difference to that goal,
        considering line-of-sight to maze walls.
        If line-of-sight to a goal is blocked by a wall, that goal's distance is treated as 10.0.
        If all goals are blocked, yaw difference is returned as 0.0.

        Args:
            robot_pos (torch.Tensor): Current robot positions (x,y), shape [num_envs, 2].
            robot_yaw (torch.Tensor): Current robot yaw angles, shape [num_envs].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - min_modified_dists (torch.Tensor): Minimum modified distance for each env, shape [num_envs, 1].
                - yaw_diff_to_eff_goal (torch.Tensor): Yaw difference to the effective goal, shape [num_envs, 1].
        """
        # Ensure all_goals is float32 and on the correct device
        all_goals = torch.tensor(self.cfg.goal_pos, device=robot_pos.device, dtype=torch.float32) # Shape: [num_goals, 2]

        num_envs = robot_pos.shape[0]
        num_goals = all_goals.shape[0]
        penalty_distance_val = 10.0

        # If there are no goals, return penalty distance and 0 yaw difference
        if num_goals == 0:
            return (
                torch.full((num_envs, 1), penalty_distance_val, device=robot_pos.device, dtype=torch.float32),
                torch.zeros((num_envs, 1), device=robot_pos.device, dtype=torch.float32)
            )

        # Calculate Euclidean distances from each robot to each goal
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)  # Shape: [num_envs, num_goals, 2]
        euclidean_dists = torch.norm(diff, dim=2)               # Shape: [num_envs, num_goals]

        if check_los:
            # Prepare line segments for LOS check
            line_starts_flat = robot_pos.unsqueeze(1).expand(-1, num_goals, -1).reshape(num_envs * num_goals, 2)
            line_ends_flat = all_goals.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_goals, 2)

            if self.maze.walls_start_tensor is not None and self.maze.walls_start_tensor.shape[0] > 0:
                is_blocked_flat = self.maze.are_line_segments_blocked_by_walls(line_starts_flat, line_ends_flat)
            else:
                is_blocked_flat = torch.zeros(num_envs * num_goals, dtype=torch.bool, device=robot_pos.device)
            
            is_blocked_matrix = is_blocked_flat.view(num_envs, num_goals) # Shape: [num_envs, num_goals]

            penalty_tensor = torch.tensor(penalty_distance_val, device=robot_pos.device, dtype=torch.float32)
            modified_dists = torch.where(is_blocked_matrix, penalty_tensor, euclidean_dists) # Shape: [num_envs, num_goals]

            # Find the minimum of these modified distances and the indices of these goals
            min_final_dists, min_dist_indices = torch.min(modified_dists, dim=1)  # Shapes: [num_envs], [num_envs]

            # Get the XY coordinates of the effective closest goals
            # Need to gather based on min_dist_indices. all_goals shape [num_goals, 2]
            # min_dist_indices shape [num_envs]
            effective_closest_goal_xy = all_goals[min_dist_indices] # Shape: [num_envs, 2]

            # Calculate vector and angle to the effective closest goal
            vector_to_effective_goal = effective_closest_goal_xy - robot_pos # Shape: [num_envs, 2]
            angle_to_effective_goal = torch.atan2(vector_to_effective_goal[:, 1], vector_to_effective_goal[:, 0]) # Shape: [num_envs]

            # Compute yaw difference to the effective goal
            # robot_yaw is expected to be shape [num_envs]
            yaw_diff_to_final_goal = (angle_to_effective_goal - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi # Shape: [num_envs]

            # If the minimum distance is the penalty distance, set yaw_difference to 0.0
            is_penalized = (min_final_dists >= penalty_distance_val - 1e-5) # Add tolerance for float comparison
            yaw_diff_to_final_goal = torch.where(is_penalized, torch.zeros_like(yaw_diff_to_final_goal), yaw_diff_to_final_goal)

        else:
            min_final_dists, min_dist_indices = torch.min(euclidean_dists, dim=1) # Shapes: [num_envs], [num_envs]
            
            # Get the XY coordinates of the closest goals (no LOS check)
            closest_goal_xy_no_los = all_goals[min_dist_indices] # Shape: [num_envs, 2]

            # Calculate vector and angle to the closest goal (no LOS check)
            vector_to_closest_goal_no_los = closest_goal_xy_no_los - robot_pos # Shape: [num_envs, 2]
            angle_to_closest_goal_no_los = torch.atan2(vector_to_closest_goal_no_los[:, 1], vector_to_closest_goal_no_los[:, 0]) # Shape: [num_envs]
            
            # Compute yaw difference to the closest goal (no LOS check)
            yaw_diff_to_final_goal = (angle_to_closest_goal_no_los - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi # Shape: [num_envs]

        return min_final_dists.unsqueeze(1), yaw_diff_to_final_goal.unsqueeze(1)
    
    def _get_cell(self, robot_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the grid cell index for each robot position.
        The grid is defined by the boundaries and cell size.
        The grid position is returned as a single index, since the grid is linearized.

        Args:
            robot_pos (torch.Tensor): Robot positions, shape [num_envs, 2].

        Returns:
            torch.Tensor: Grid cell indices, shape [num_envs, 1].
        """
        x_min, x_max, y_min, y_max = self.cfg.grid_boundaries
        cell_size = self.cfg.grid_cell_size

        max_x = int((x_max - x_min) / cell_size)

        # Compute grid indices
        grid_x = ((robot_pos[:, 0] - x_min) / cell_size).long()
        grid_y = ((robot_pos[:, 1] - y_min) / cell_size).long()

        # Ensure indices are within bounds
        grid_x = torch.clamp(grid_x, 0, int((x_max - x_min) / cell_size) - 1)
        grid_y = torch.clamp(grid_y, 0, int((y_max - y_min) / cell_size) - 1)

        grid_pos = grid_x + max_x * grid_y

        return grid_pos.unsqueeze(-1)  # Shape: [num_envs, 1] 

    def _compute_yaw(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions [w, x, y, z] to yaw angles.

        Args:
            quaternions (torch.Tensor): Tensor of quaternions with shape [num_envs, 4].

        Returns:
            torch.Tensor: Tensor of yaw angles with shape [num_envs, 1].
        """
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        # Calculate yaw using the standard formula
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw.unsqueeze(-1)
    
    def _set_dual_multiplier(self, value: float) -> None:
        self.dual_multiplier = value

    def _get_constraint_cost(self):
        return self._computed_constraint_cost
    
    def _get_last_episode_costs(self):
        return self.last_episode_costs
    
    def _get_safety_probability(self):
        return torch.mean(self.safety_indicator)
    
    def _reset_reward_and_cost(self):
        self.episode_rewards = torch.zeros(self.n_envs, device=self.device)
        self.episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.n_envs, device=self.device)
    
    def _get_reward_and_cost(self):
        return self.episode_rewards, self.episode_costs
    
    def _set_distance_rew(self, value):
        self.reward_scale_distance = value

    def _get_success_rate(self) -> float:
        return (1 - (self.failed_robots / self.spawned_robots) if self.spawned_robots > 0 else 0.0)


# decorator to compile the function to TorchScript
# https://pytorch.org/docs/stable/jit.html
# useful for performance optimization
@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_scale_goal: float,
    rew_scale_distance: float,
    max_distance_from_goal: float,
    robot_pos: torch.Tensor,
    previous_distance_to_goal: torch.Tensor,
    current_distance_to_goal: torch.Tensor,
    rew_scale_exploration: float | None,
    penalty_scale_exploration: float | None,
    new_cell: torch.Tensor | None,
):
    # reward for termination
    terminated = torch.any(torch.abs(robot_pos) > max_distance_from_goal, dim=1).float()
    rew_termination = rew_scale_terminated * terminated

    # reward for reaching the goal
    rew_goal = rew_scale_goal * (current_distance_to_goal <= 0.3).float()
    
    # reward for moving towards the goal
    rew_distance = torch.where(previous_distance_to_goal == 0.0, 
                           torch.zeros_like(current_distance_to_goal), 
                           previous_distance_to_goal - current_distance_to_goal)
    rew_distance = rew_scale_distance * rew_distance
    # clip distance reward to 0 as minimum
    rew_distance = torch.clamp(rew_distance, min=0.0)

    # reward for exploration
    if rew_scale_exploration is not None and penalty_scale_exploration is not None and new_cell is not None:
        # reward for exploration
        rew_exploration = rew_scale_exploration * new_cell.float()
        # penalty for exploration
        penalty_exploration = penalty_scale_exploration * (1 - new_cell.float())
        rew_exploration = rew_exploration + penalty_exploration
        total_reward = rew_distance + rew_goal + rew_termination + rew_exploration
        
    else:
        rew_exploration = None
        total_reward = rew_distance + rew_goal + rew_termination
  
    return rew_distance, rew_goal, rew_termination, rew_exploration, total_reward

@torch.jit.script
def compute_cost(
    cost_obstacle: float,
    min_dist: float,
    dist_to_obs: torch.Tensor,
):
    
    cost_obstacles = torch.where(dist_to_obs <= min_dist, cost_obstacle*(1 - dist_to_obs/min_dist), torch.tensor(0.0)).sum(dim=1)
    return cost_obstacles


# define the Maze class

class Maze:
    def __init__(self):
        self.walls = []
        self.walls_start_tensor = None
        self.walls_end_tensor = None
        self.obstacle_prims = []

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "maze_cfg.yaml")

        with open(self.config_path, "r") as f:
            maze_config = yaml.safe_load(f)
        self.walls_config = maze_config["maze"]["walls"]
    
        if self.walls_config:
             cpu_device = torch.device("cpu") # Default to CPU for initial tensor creation
             self.walls_start_tensor = torch.tensor([wall["start"] for wall in self.walls_config],
                                                 device=cpu_device, # Will be moved to GPU in spawn_maze if needed
                                                 dtype=torch.float32)
             self.walls_end_tensor = torch.tensor([wall["end"] for wall in self.walls_config],
                                               device=cpu_device, # Will be moved to GPU in spawn_maze if needed
                                               dtype=torch.float32)
        else: # Handle case with no walls defined
            self.walls_start_tensor = torch.empty((0,2), device=torch.device("cpu"), dtype=torch.float32)
            self.walls_end_tensor = torch.empty((0,2), device=torch.device("cpu"), dtype=torch.float32)

    def spawn_maze(self, width=0.1, height=0.5, walls_config=None):
        if walls_config is None:
            walls_config = self.walls_config

        self.width = width
        self.height = height


        self.walls_start_tensor = torch.tensor([wall["start"] for wall in walls_config],
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                 dtype=torch.float)
        self.walls_end_tensor = torch.tensor([wall["end"] for wall in walls_config],
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               dtype=torch.float)
        
        for i, wall in enumerate(walls_config):
            prim_path = f"/World/Maze/wall_{i}"
            # print("Add wall: ", prim_path)
            self.spawn_wall_cube(prim_path, wall["start"], wall["end"])

    def spawn_wall_cube(self, prim_path: str, start: list[float], end: list[float]):
        # compute length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        # compute midpoint
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        mid_z = self.height / 2.0  # place it on the ground
        translation = (mid_x, mid_y, mid_z)

        size = (length, self.width, self.height)

        # compute orientation angle
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        theta_rad = math.radians(angle_deg)
        w = math.cos(theta_rad / 2)
        x = 0.0
        y = 0.0
        z = math.sin(theta_rad / 2)
        orientation = (w, x, y, z)

        # create the cuboid configuration using sim_utils
        cfg_wall_cube = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.14, 0.35))
        )

        cfg_wall_cube.func(
            prim_path,
            cfg_wall_cube,
            translation=translation,
            orientation=orientation
        )
        # print("Spawn wall cube: ", prim_path)
        
        return cfg_wall_cube

    def sample_safe_positions(self,
                            num_envs: int,
                            x_bounds: Tuple[float, float] = (-1.4, 1.4),
                            y_bounds: Tuple[float, float] = (-2.9, 2.9),
                            min_dist: float = 0.15,
                            max_attempts: int = 100,
                            safety_margin: float = 0.01):
        """
        Vectorized sampling of one safe (x,y) per env.
        """

        device = self.walls_start_tensor.device

        # 1) generate all candidates at once: shape (num_envs, max_attempts, 2)
        u = torch.rand((num_envs, max_attempts), device=device)
        v = torch.rand((num_envs, max_attempts), device=device)
        xs = x_bounds[0] + u * (x_bounds[1] - x_bounds[0])
        ys = y_bounds[0] + v * (y_bounds[1] - y_bounds[0])
        candidates = torch.stack((xs, ys), dim=-1)  # (num_envs, max_attempts, 2)

        # 2) compute distances for *all* these points in one shot.
        #    We’ll reshape to (num_envs*max_attempts, 2) to call your check_collision,
        #    then reshape back to (num_envs, max_attempts, k)
        flat = candidates.view(-1, 2)
        dists_flat = self.check_collision(flat)    # returns (n*m, k)
        k = dists_flat.shape[-1]
        dists = dists_flat.view(num_envs, max_attempts, k)

        # 3) build a “safe” mask per candidate:
        threshold = min_dist + safety_margin
        safe_mask = (dists >= threshold).all(dim=-1)   # shape (num_envs, max_attempts)

        # 4) for each env, find the first safe index
        #    if none is safe, we’ll just pick the last candidate
        #    torch.argmax will give us the first True in each row, but if
        #    there are no Trues it returns 0 — so we need a fallback.
        any_safe = safe_mask.any(dim=-1)               # (num_envs,)
        first_idx = torch.argmax(safe_mask.int(), dim=-1)  # (num_envs,) — first True or 0
        fallback_idx = torch.full_like(first_idx, max_attempts-1)
        pick_idx = torch.where(any_safe, first_idx, fallback_idx)  # (num_envs,)

        # 5) gather the chosen (x,y) per env
        env_idx = torch.arange(num_envs, device=device)
        chosen = candidates[env_idx, pick_idx]        # (num_envs, 2)

        return chosen  # tensor of shape [num_envs,2]
        
    def distance_from_line(self, point, start, end):
        # compute perpendicular distance from point to the line defined by start and end.
        if isinstance(point, list):
            point = torch.tensor(point, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if point.is_cuda:  # check if the tensor is on a GPU
            p = np.array(point.cpu())  # move to CPU before converting to NumPy
        else:
            p = np.array(point) 
        s = np.array(start)
        e = np.array(end)
        if np.allclose(s, e):
            return np.linalg.norm(p - s)
        return np.abs(np.cross(e-s, s-p)) / np.linalg.norm(e-s)
    
    def check_collision(self, pos):
        """
        Compute the distances from each robot (given by their positions pos)
        to each wall defined by its start and end points.
        
        Args:
            pos (torch.Tensor): Robot positions, shape [num_envs, 2].
            min_dist (float): Minimum allowed distance from the wall.
            
        Returns:
            torch.Tensor: A tensor of shape [num_envs, num_walls] with the computed distances.
                        (You can then choose how to aggregate these, e.g. taking the minimum.)
        """
        # Adjust the minimum distance by adding half the wall width if needed.
        #min_dist_adjusted = min_dist + self.width / 2.0
        
        if self.walls_start_tensor is None or self.walls_end_tensor is None:
            return torch.full((pos.shape[0], 3), float('inf'), device=pos.device)

        # walls_start_tensor and walls_end_tensor are assumed to have shape [num_walls, 2]
        walls_start = self.walls_start_tensor  # shape: [num_walls, 2]
        walls_end = self.walls_end_tensor      # shape: [num_walls, 2]
        
        # Compute wall direction vectors and their squared lengths
        wall_dirs = walls_end - walls_start            # shape: [num_walls, 2]
        norm_wall_dirs_sq = (wall_dirs ** 2).sum(dim=1)  # shape: [num_walls]
        norm_wall_dirs_sq = torch.where(norm_wall_dirs_sq == 0, torch.tensor(1e-6, device=pos.device), norm_wall_dirs_sq)
        
        # Expand dimensions so that we can compute pairwise differences:
        # pos: [num_envs, 2] -> [num_envs, 1, 2]
        # walls_start: [num_walls, 2] -> [1, num_walls, 2]
        pos_exp = pos.unsqueeze(1)              # shape: [num_envs, 1, 2]
        walls_start_exp = walls_start.unsqueeze(0)  # shape: [1, num_walls, 2]
        wall_dirs_exp = wall_dirs.unsqueeze(0)      # shape: [1, num_walls, 2]
        walls_end_exp = walls_end.unsqueeze(0)      # shape: [1, num_walls, 2]

        # Compute vector from wall start to each robot position
        vec = pos_exp - walls_start_exp          # shape: [num_envs, num_walls, 2]
        
        # Compute the projection parameter t for each robot-wall pair:
        # t = ((P - A) dot (B - A)) / ||B-A||^2
        dot = (vec * wall_dirs_exp).sum(dim=2)     # shape: [num_envs, num_walls]
        norm_sq = norm_wall_dirs_sq.unsqueeze(0)    # shape: [1, num_walls]
        t = dot / norm_sq                          # shape: [num_envs, num_walls]
        
        # Compute the projection point on the infinite line: P_proj = A + t*(B-A)
        p_proj = walls_start_exp + t.unsqueeze(-1) * wall_dirs_exp  # shape: [num_envs, num_walls, 2]
    
        # Compute perpendicular distance (d_perp) to the infinite line
        d_perp = torch.norm(pos_exp - p_proj, dim=2)           # shape: [num_envs, num_walls]

        # --- Case 1: When projection falls on the segment (0 <= t <= 1) ---
        mask_inside = (t >= 0.0) & (t <= 1.0)  # shape: [num_envs, num_walls]
        d_on_segment = torch.clamp(d_perp - self.width/2.0, min=0.0)  # adjusted distance when on segment

        # For outside cases, determine the closest endpoint.
        # For t < 0, use walls_start; for t > 1, use walls_end.
        # Compute distance from projection to A and B.
        d_ep_A = torch.norm(p_proj - walls_start_exp, dim=2)  # distance from projection to start, shape: [num_envs, num_walls]
        d_ep_B = torch.norm(p_proj - walls_end_exp, dim=2)      # distance from projection to end, shape: [num_envs, num_walls]
        # Choose the correct endpoint distance based on t.
        d_ep = torch.where(t < 0, d_ep_A, d_ep_B)  # shape: [num_envs, num_walls]
        
        # Now, for outside cases, further decide:
        # If d_perp < w/2, robot is "beside" the wall; use d_ep.
        # Else, robot is near a corner; use corner distance computed as:
        # d_corner = sqrt( d_ep^2 + (d_perp - w/2)^2 ).
        corner_distance = torch.sqrt(d_ep**2 + (d_perp - self.width/2.0)**2)
        # Choose: if d_perp < w/2, then use d_ep; else, use corner_distance.
        d_outside = torch.where(d_perp < self.width/2.0, d_ep, corner_distance)

        # For outside, use d_outside; for inside, use d_on_segment.
        final_dists = torch.where(mask_inside, d_on_segment, d_outside)  # shape: [num_envs, num_walls]

        #capped_dist = torch.where(final_dists <= 0.4, final_dists, torch.tensor(0.4, device=final_dists.device))
        
        # Optionally sort and return the smallest few distances.
        sorted_dists, _ = torch.sort(final_dists, dim=1)
        #sorted_dists, _ = torch.sort(capped_dist, dim=1)
        smallest3 = sorted_dists[:, :3]
        #print("Distances: ", smallest3)
        
        # print("OTHER SHAPE: ", torch.full((pos.shape[0], 3), float('inf'), device=pos.device).shape)
        # print("SHAPE: ", smallest3.shape)

        return smallest3
    
    def check_collision_ang(self, pos, yaw):
        """
        Compute the distances from each robot (given by their positions pos)
        to each wall defined by its start and end points.
        Also compute the angle difference between the robot's yaw and the wall direction.
        
        Args:
            pos (torch.Tensor): Robot positions, shape [num_envs, 2].
            min_dist (float): Minimum allowed distance from the wall.
            
        Returns:
            torch.Tensor: A tensor of shape [num_envs, 3] with the computed distances to the 3 closest walls.
                        (You can then choose how to aggregate these, e.g. taking the minimum.)
            torch.Tensor: A tensor of shape [num_envs, 3] with the angle differences to the 3 closest walls.
        """
        # Adjust the minimum distance by adding half the wall width if needed.
        #min_dist_adjusted = min_dist + self.width / 2.0
        

        if self.walls_start_tensor is None or self.walls_end_tensor is None:
            ret = torch.full((pos.shape[0], 3), float('inf'), device=pos.device), torch.zeros((pos.shape[0], 3), device=pos.device)
            print("SHAPE: ", ret[0].shape)
            print("SHAPE: ", ret[1].shape)
            return ret

        yaw.unsqueeze_(1)  # shape: [num_envs, 1]

        # walls_start_tensor and walls_end_tensor are assumed to have shape [num_walls, 2]
        walls_start = self.walls_start_tensor  # shape: [num_walls, 2]
        walls_end = self.walls_end_tensor      # shape: [num_walls, 2]
        
        # Compute wall direction vectors and their squared lengths
        wall_dirs = walls_end - walls_start            # shape: [num_walls, 2]
        norm_wall_dirs_sq = (wall_dirs ** 2).sum(dim=1)  # shape: [num_walls]
        norm_wall_dirs_sq = torch.where(norm_wall_dirs_sq == 0, torch.tensor(1e-6, device=pos.device), norm_wall_dirs_sq)
        
        # Expand dimensions so that we can compute pairwise differences:
        # pos: [num_envs, 2] -> [num_envs, 1, 2]
        # walls_start: [num_walls, 2] -> [1, num_walls, 2]
        pos_exp = pos.unsqueeze(1)              # shape: [num_envs, 1, 2]
        walls_start_exp = walls_start.unsqueeze(0)  # shape: [1, num_walls, 2]
        wall_dirs_exp = wall_dirs.unsqueeze(0)      # shape: [1, num_walls, 2]
        walls_end_exp = walls_end.unsqueeze(0)      # shape: [1, num_walls, 2]

        # Compute vector from wall start to each robot position
        vec = pos_exp - walls_start_exp          # shape: [num_envs, num_walls, 2]
        
        # Compute the projection parameter t for each robot-wall pair:
        # t = ((P - A) dot (B - A)) / ||B-A||^2
        dot = (vec * wall_dirs_exp).sum(dim=2)     # shape: [num_envs, num_walls]
        norm_sq = norm_wall_dirs_sq.unsqueeze(0)    # shape: [1, num_walls]
        t = dot / norm_sq                          # shape: [num_envs, num_walls]
        
        # Compute the projection point on the infinite line: P_proj = A + t*(B-A)
        p_proj = walls_start_exp + t.unsqueeze(-1) * wall_dirs_exp  # shape: [num_envs, num_walls, 2]
    
        # Compute perpendicular distance (d_perp) to the infinite line
        d_perp = torch.norm(pos_exp - p_proj, dim=2)           # shape: [num_envs, num_walls]

        # --- Case 1: When projection falls on the segment (0 <= t <= 1) ---
        mask_inside = (t >= 0.0) & (t <= 1.0)  # shape: [num_envs, num_walls]
        d_on_segment = torch.clamp(d_perp - self.width/2.0, min=0.0)  # adjusted distance when on segment

        # For outside cases, determine the closest endpoint.
        # For t < 0, use walls_start; for t > 1, use walls_end.
        # Compute distance from projection to A and B.
        d_ep_A = torch.norm(p_proj - walls_start_exp, dim=2)  # distance from projection to start, shape: [num_envs, num_walls]
        d_ep_B = torch.norm(p_proj - walls_end_exp, dim=2)      # distance from projection to end, shape: [num_envs, num_walls]
        # Choose the correct endpoint distance based on t.
        d_ep = torch.where(t < 0, d_ep_A, d_ep_B)  # shape: [num_envs, num_walls]
        
        # Now, for outside cases, further decide:
        # If d_perp < w/2, robot is "beside" the wall; use d_ep.
        # Else, robot is near a corner; use corner distance computed as:
        # d_corner = sqrt( d_ep^2 + (d_perp - w/2)^2 ).
        corner_distance = torch.sqrt(d_ep**2 + (d_perp - self.width/2.0)**2)
        # Choose: if d_perp < w/2, then use d_ep; else, use corner_distance.
        d_outside = torch.where(d_perp < self.width/2.0, d_ep, corner_distance)

        # For outside, use d_outside; for inside, use d_on_segment.
        final_dists = torch.where(mask_inside, d_on_segment, d_outside)  # shape: [num_envs, num_walls]

        #capped_dist = torch.where(final_dists <= 0.4, final_dists, torch.tensor(0.4, device=final_dists.device))
        
        # Optionally sort and return the smallest few distances.
        sorted_dists, sorted_indices = torch.sort(final_dists, dim=1)
        #sorted_dists, _ = torch.sort(capped_dist, dim=1)
        smallest3 = sorted_dists[:, :3]
        #print("Distances: ", smallest3)
        top3_indices = sorted_indices[:, :3]

        # Extract the vectors from robot to the 3 closest projection points
        # p_proj: [num_envs, num_walls, 2] → pick top3
        # Use gather with proper indexing
        batch_indices = torch.arange(pos.shape[0], device=pos.device).unsqueeze(1).expand(-1, 3)
        p_proj_closest = p_proj[batch_indices, top3_indices]          # shape: [num_envs, 3, 2]
        vectors_to_obs = p_proj_closest - pos.unsqueeze(1)            # [num_envs, 3, 2]

        # Compute direction angle of each vector: atan2(dy, dx)
        obs_angles = torch.atan2(vectors_to_obs[:, :, 1], vectors_to_obs[:, :, 0])  # [num_envs, 3]

        # Compute yaw difference: wrap to (-π, π]
        angle_diffs = obs_angles - yaw                                # [num_envs, 3]
        angle_diffs = (angle_diffs + torch.pi) % (2 * torch.pi) - torch.pi

        return smallest3, angle_diffs
    
    @staticmethod
    def _orientation_static(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Computes the orientation of ordered triplet (p, q, r).
        Returns:
            torch.Tensor: 0 if p, q, r are collinear.
                          1 if (p, q, r) is clockwise.
                         -1 if (p, q, r) is counterclockwise.
        Shape of p, q, r: [..., 2]
        Output shape: [...]
        """
        val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - \
              (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
        return torch.sign(val)

    @staticmethod
    def _on_segment_static(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Given three collinear points p, q, r, the function checks if point q lies on segment 'pr'.
        Shape of p, q, r: [..., 2]
        Output shape: [...] (boolean)
        """
        # Add a small epsilon for floating point comparisons
        eps = 1e-6
        return (q[..., 0] <= torch.maximum(p[..., 0], r[..., 0]) + eps) & \
               (q[..., 0] >= torch.minimum(p[..., 0], r[..., 0]) - eps) & \
               (q[..., 1] <= torch.maximum(p[..., 1], r[..., 1]) + eps) & \
               (q[..., 1] >= torch.minimum(p[..., 1], r[..., 1]) - eps)

    def check_line_segment_intersections(self,
                                         seg1_starts: torch.Tensor,  # [N, 2]
                                         seg1_ends: torch.Tensor,    # [N, 2]
                                         seg2_starts: torch.Tensor,  # [M, 2]
                                         seg2_ends: torch.Tensor     # [M, 2]
                                         ) -> torch.Tensor:          # [N, M] boolean, True if intersection
        """
        Checks for intersections between N segments (seg1) and M segments (seg2)
        using the orientation method.
        """
        N = seg1_starts.shape[0]
        M = seg2_starts.shape[0]
        device = seg1_starts.device

        if N == 0 or M == 0:
            return torch.zeros((N, M), dtype=torch.bool, device=device)

        # Expand dimensions for broadcasting
        p1 = seg1_starts.unsqueeze(1)  # [N, 1, 2]
        q1 = seg1_ends.unsqueeze(1)    # [N, 1, 2]
        p2 = seg2_starts.unsqueeze(0)  # [1, M, 2]
        q2 = seg2_ends.unsqueeze(0)    # [1, M, 2]

        # Calculate orientations
        o1 = Maze._orientation_static(p1, q1, p2)  # Orientation(p1, q1, p2)
        o2 = Maze._orientation_static(p1, q1, q2)  # Orientation(p1, q1, q2)
        o3 = Maze._orientation_static(p2, q2, p1)  # Orientation(p2, q2, p1)
        o4 = Maze._orientation_static(p2, q2, q1)  # Orientation(p2, q2, q1)

        # General case: Segments cross each other
        # This happens if (p1, q1, p2) and (p1, q1, q2) have different orientations,
        # AND (p2, q2, p1) and (p2, q2, q1) have different orientations.
        # o1 * o2 < 0 means different non-zero orientations.
        general_intersects = (o1 * o2 < 0) & (o3 * o4 < 0)

        # Special Cases: Collinear intersections
        # Case 1: p1, q1, p2 are collinear and p2 lies on segment p1q1
        collinear_case1 = (o1 == 0) & Maze._on_segment_static(p1, p2, q1)
        # Case 2: p1, q1, q2 are collinear and q2 lies on segment p1q1
        collinear_case2 = (o2 == 0) & Maze._on_segment_static(p1, q2, q1)
        # Case 3: p2, q2, p1 are collinear and p1 lies on segment p2q2
        collinear_case3 = (o3 == 0) & Maze._on_segment_static(p2, p1, q2)
        # Case 4: p2, q2, q1 are collinear and q1 lies on segment p2q2
        collinear_case4 = (o4 == 0) & Maze._on_segment_static(p2, q1, q2)
        
        collinear_intersects = collinear_case1 | collinear_case2 | collinear_case3 | collinear_case4
        
        return general_intersects | collinear_intersects

    def are_line_segments_blocked_by_walls(self, seg_starts: torch.Tensor, seg_ends: torch.Tensor) -> torch.Tensor:
        """
        Checks if multiple line segments are blocked by any wall in the maze.
        Args:
            seg_starts (torch.Tensor): Start points of segments, shape [K, 2].
            seg_ends (torch.Tensor): End points of segments, shape [K, 2].
        Returns:
            torch.Tensor: Boolean tensor of shape [K]. True if segment k is blocked by any wall.
        """
        if self.walls_start_tensor is None or self.walls_start_tensor.shape[0] == 0:
            return torch.zeros(seg_starts.shape[0], dtype=torch.bool, device=seg_starts.device)
        
        # Ensure wall tensors are on the same device as segments
        wall_starts_device = self.walls_start_tensor.to(seg_starts.device)
        wall_ends_device = self.walls_end_tensor.to(seg_starts.device)

        # Check intersections between each segment and all walls
        # Resulting shape: [K, num_walls]
        intersections_matrix = self.check_line_segment_intersections(
            seg_starts, seg_ends,
            wall_starts_device, wall_ends_device
        )
        # A segment is blocked if it intersects with ANY wall
        return intersections_matrix.any(dim=1)



def direct_diff_drive(linear_velocity: float, angular_velocity: float, wheelbase: float, wheel_radius: float) -> tuple[float, float]:
    """
    Converts base linear and angular velocity commands into left and right wheel velocities for a differential drive robot.

    Args:
        linear_velocity (float): The desired linear velocity of the robot's center.
        angular_velocity (float): The desired angular velocity of the robot (yaw rate).
        wheelbase (float): The distance between the left and right wheels.
        wheel_radius (float): The radius of the robot's wheels.

    Returns:
        tuple[float, float]: A tuple containing the left wheel angular velocity and the right wheel angular velocity.
    """
    # Compute left and right wheel linear velocities
    v_left = linear_velocity - (angular_velocity * wheelbase / 2)
    v_right = linear_velocity + (angular_velocity * wheelbase / 2)

    # Convert linear velocities to angular velocities
    omega_left = v_left / wheel_radius
    omega_right = v_right / wheel_radius

    return omega_left, omega_right

def inverse_diff_drive(omega_left: float, omega_right: float, wheelbase: float, wheel_radius: float) -> tuple[float, float]:
    """
    Converts left and right wheel angular velocities into base linear and angular velocity commands for a differential drive robot.

    Args:
        omega_left (float): The angular velocity of the left wheel.
        omega_right (float): The angular velocity of the right wheel.
        wheelbase (float): The distance between the left and right wheels.
        wheel_radius (float): The radius of the robot's wheels.

    Returns:
        tuple[float, float]: A tuple containing the robot's linear velocity and angular velocity.
    """
    # Compute linear velocity
    linear_velocity = (wheel_radius / 2) * (omega_left + omega_right)

    # Compute angular velocity
    angular_velocity = (wheel_radius / wheelbase) * (omega_right - omega_left)

    return linear_velocity, angular_velocity