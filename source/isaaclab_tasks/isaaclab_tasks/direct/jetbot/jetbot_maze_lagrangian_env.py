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
import numpy as np
from typing import Tuple
import csv




# utility class to log reward and cost statistics to a CSV file
class CustomCSVLogger:
    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"iter_reward_log_{timestamp}.csv")
        self.header = [
            "dual", "dist_reward", "dr_std", "goal_reward", "gr_std", "term_reward", "tr_std", "expl_reward", "expl_std",
            "mean_reward", "std_reward", "mean_cost", "std_cost", "total_reward", "safety_prob"
        ]
        self._write_header()


    def _write_header(self):
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)


    def record(self,
               dual: float,
               dist_reward: float, dist_std: float,
               goal_reward: float, goal_std: float,
               term_reward: float, term_std: float,
               expl_reward: float, expl_std: float,
               mean_reward: float, std_reward: float,
               mean_cost: float, std_cost: float,
               total_reward: float, safety_prob: float) -> None:
        data = [
            dual, dist_reward, dist_std,
            goal_reward, goal_std,
            term_reward, term_std,
            expl_reward, expl_std,
            mean_reward, std_reward,
            mean_cost, std_cost,
            total_reward, safety_prob
        ]
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)




# utility function to recursively unwrap a wrapped environment
def get_unwrapped_env(env):
    current_env = env
    while hasattr(current_env, "env") or hasattr(current_env, "venv"):
        current_env = getattr(current_env, "env", getattr(current_env, "venv", current_env))
    return current_env




# define the configuration for the JetBot environment.
@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    # environment config
    decimation: int = 4 # how many physics steps to skip per action
    episode_length_s: float = 120.0 # maximum episode length in seconds
    action_scale: float = 1.0 # scale factor for the actions
    action_space: int = 2 # number of action outputs (left and right wheel velocities)
    observation_space: int = 12  # x, y, dist_to_goal, yaw_diff, lin_vel, ang_vel, dist_to_obs (x3), angle_diff (x3)
    state_space: int = 0

    # simulation config
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot config
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/JetBot")
    left_wheel_joint_name = "left_wheel_joint"
    right_wheel_joint_name = "right_wheel_joint"

    # scene setup
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)

    # reset parameters
    max_distance_from_goal: float = 3.5
    spawn_height: float = 0.0437

    # spatial discretization for exploration tracking
    grid_boundaries = (-1.5, 1.5, -3.0, 3.0)  # (x_min, x_max, y_min, y_max)
    grid_cell_size: float = 0.5

    # maze walls width
    wall_width: float = 0.05

    # goal positions in XY
    goal_pos = [[-2.0, 1.0], [-2.0, 0.0], [2.0, 0.0], [0.5, 3.5], [-0.5, -3.5]]

    # robot geometry
    wheelbase: float = 0.12
    wheelradius: float = 0.032




# main environment class for the JetBot maze task.
class JetBotEnv(DirectRLEnv):
    cfg: JetBotEnvCfg

    def __init__(self, cfg: JetBotEnvCfg, render_mode: str | None = None, **kwargs):
        # instantiate maze geometry utility
        self.maze = Maze()
        super().__init__(cfg, render_mode, **kwargs)

        # retrieve joint indices for motorized wheels from articulation model
        self._left_wheel_idx, _ = self.jetbot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_wheel_idx, _ = self.jetbot.find_joints(self.cfg.right_wheel_joint_name)

        # scale applied to raw actions before execution
        self.action_scale = self.cfg.action_scale

        # alias joint position and velocity tensors for efficient access
        self.joint_pos = self.jetbot.data.joint_pos
        self.joint_vel = self.jetbot.data.joint_vel

        # initialize previous distance to goal (used in shaping reward)
        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        # load reward parameters from external YAML config file
        config_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_cfg_maze.yaml")
        print("Config path: ", config_path)

        try:
            with open(config_path) as stream:
                data = yaml.safe_load(stream)
                # extract reward and cost scales
                self.gamma = data["agent"]["discount_factor"]
                self.reward_scale_goal = data["rewards"]["goal"]
                self.reward_scale_distance = data["rewards"]["distance"]
                self.reward_scale_terminated = data["rewards"]["termination"]
                self.cost_obstacle = data["costs"]["obstacle"]
                self.min_distance_to_obstacle = data["min_dist"]
                self.reward_scale_exploration = data["rewards"].get("exploration_reward")
                self.penalty_scale_exploration = data["rewards"].get("exploration_penalty")
        except yaml.YAMLError as exc:
            # fallback values if YAML is missing or malformed
            print(f"Error loading YAML config: {exc}. Using default reward/cost parameters.")
            self.gamma = 0.99
            self.reward_scale_goal = 60.0
            self.reward_scale_distance = 10.0
            self.reward_scale_terminated = -50.0
            self.cost_obstacle = 6.0
            self.min_distance_to_obstacle = 0.15
            self.reward_scale_exploration = None
            self.penalty_scale_exploration = 0.0

        # print loaded reward/cost parameters for debugging
        print("Gamma: ", self.gamma)
        print("Reward scale goal: ", self.reward_scale_goal)
        print("Reward scale distance: ", self.reward_scale_distance)
        print("Reward scale terminated: ", self.reward_scale_terminated)
        print("Cost obstacle: ", self.cost_obstacle)
        print("Min dist to obstacle: ", self.min_distance_to_obstacle)
        print("Exploration reward: ", self.reward_scale_exploration)
        print("Exploration penalty: ", self.penalty_scale_exploration)

        # lagrangian dual variable for cost shaping
        self.dual_multiplier = 0.0

        # alias number of environments for convenience
        self.n_envs = self.cfg.scene.num_envs

        # initialize reward and cost buffers
        self.episode_rewards = torch.zeros(self.n_envs, device=self.device)
        self.episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.n_envs, device=self.device)

        # binary indicator of whether the agent was safe in each env during the step
        self.safety_indicator = torch.ones(self.n_envs, device=self.device)

        # initialize grid-based exploration tracking if enabled
        if self.reward_scale_exploration is not None:
            num_cells_x = int((self.cfg.grid_boundaries[1] - self.cfg.grid_boundaries[0]) / self.cfg.grid_cell_size)
            num_cells_y = int((self.cfg.grid_boundaries[3] - self.cfg.grid_boundaries[2]) / self.cfg.grid_cell_size)
            num_cells = num_cells_x * num_cells_y
            # binary grid indicating whether each cell has been visited in each env
            self.grids = torch.zeros(self.n_envs, num_cells, device=self.device)

        # initialize reward logger
        self.logger = CustomCSVLogger()

        # count robots that were spawned and failed (used for success rate estimation)
        self.spawned_robots = 0
        self.failed_robots = 0

        # track cumulative actions to compute statistics
        self.action_accumulation = torch.zeros(self.n_envs, self.cfg.action_space, device=self.device)
        self.action_count = 0


    # initializes the simulation environment by populating the scene with all necessary components:
    # the jetbot robot articulation, ground plane, visual goal markers, ambient lighting, and the maze walls.
    # also registers the articulation into the simulation backend and replicates the environment across instances.
    def _setup_scene(self):
        # create robot articulation and assign to scene
        self.jetbot = Articulation(self.cfg.robot_cfg)

        # add flat ground plane to simulation
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add visual goal cubes at specified positions
        goal_size = (0.5, 0.5, 0.25)
        for i, g in enumerate(self.cfg.goal_pos):
            # compute 3D position of goal cube based on xy and height offset
            goal_pos = list(g) + [goal_size[2] / 2.0]
            cfg_goal_cube = sim_utils.CuboidCfg(
                size=goal_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.2, 0.92)),
            )
            # instantiate the visual goal object in the USD scene
            cfg_goal_cube.func(f"/World/Objects/goal_{i}", cfg_goal_cube, translation=tuple(goal_pos))

        # duplicate the environment template across all envs
        self.scene.clone_environments(copy_from_source=False)

        # attach articulation object to the scene for simulation tracking
        self.scene.articulations["jetbot"] = self.jetbot

        # configure ambient lighting (dome light simulates sky illumination)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # spawn maze walls from the Maze utility
        self.maze.spawn_maze(width=self.cfg.wall_width, height=0.5)


    # prepares the incoming actions before the physics step.
    # clones and stores the raw actions, accumulates them for statistical tracking (mean/std),
    # and defers actual actuation to the _apply_action method.
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # clone actions to prevent in-place modification from downstream
        self.actions = actions.clone()

        # apply action scaling
        # self.actions = torch.tanh(self.actions) * 6.0  # Ensure actions are in [-6, 6] range

        """
        # accumulate actions for computing statistics (mean and std) over time
        self.action_accumulation += self.actions
        self.action_count += 1

        # compute average action across environments over the last `action_count` calls
        mean_actions = self.action_accumulation.mean(dim=0) / self.action_count

        # compute standard deviation of actions across environments
        std_actions = self.action_accumulation.std(dim=0) / self.action_count

        # optionally print action statistics for debugging
        print("Mean actions: ", mean_actions)
        print("Std actions: ", std_actions)
        """


    # applies the current policy action to the robot's wheel joints by setting target velocities.
    # the action tensor is expected to have shape [num_envs, 2] corresponding to left and right wheel commands.
    def _apply_action(self) -> None:
        # extract left and right wheel commands from action vector
        left_action = self.actions[:, 0].unsqueeze(-1)   # shape: [num_envs, 1]
        right_action = self.actions[:, 1].unsqueeze(-1)  # shape: [num_envs, 1]

        # optionally print raw wheel actions
        print("Left action: ", left_action, "\nRight action: ", right_action)

        # apply the velocity commands to the corresponding joints
        self.jetbot.set_joint_velocity_target(left_action, joint_ids=self._left_wheel_idx)
        self.jetbot.set_joint_velocity_target(right_action, joint_ids=self._right_wheel_idx)


    # constructs and returns the policy observation vector for each environment.
    # the observation includes kinematic quantities (position, velocity), task information (goal distance and yaw error),
    # and obstacle proximity and orientation, concatenated into a single tensor per environment.
    def _get_observations(self) -> dict:
        # get the (x, y) position of the robot base in world frame
        position = self.jetbot.data.root_pos_w[:, :2]  # shape: [num_envs, 2]

        # extract base orientation as yaw angle from quaternion
        orientation = self.jetbot.data.root_quat_w     # shape: [num_envs, 4]
        yaw = self._compute_yaw(orientation).squeeze(-1)  # shape: [num_envs]

        # compute distance and relative angle to nearby obstacles
        distance_to_obstacles, obstacle_angle_difference = self.maze.check_collision_ang(position, yaw.clone())

        # update safety indicator: mark environments with close obstacles as unsafe
        too_close = (distance_to_obstacles < self.min_distance_to_obstacle).any(dim=1)
        self.safety_indicator[too_close] = 0

        # extract joint (wheel) velocities
        left_wheel_velocity = self.joint_vel[:, self._left_wheel_idx].squeeze(-1).unsqueeze(-1)
        right_wheel_velocity = self.joint_vel[:, self._right_wheel_idx].squeeze(-1).unsqueeze(-1)

        # compute task-related features: distance and heading error to goal
        distance_to_goal_los, yaw_difference_los = self._get_distance_and_angle_from_goal_with_los_check(
            position, yaw, check_los=False
        )

        # concatenate all components to form the final observation vector
        observation = torch.cat((
            position,                   # robot base (x, y)
            distance_to_goal_los,       # scalar distance to nearest goal
            yaw_difference_los,         # angular error to nearest goal
            left_wheel_velocity,        # wheel speed
            right_wheel_velocity,
            distance_to_obstacles,      # distances to nearest 3 obstacles
            obstacle_angle_difference   # angles to nearest 3 obstacles
        ), dim=-1)

        return {"policy": observation}
    

    # computes the Lagrangian reward signal for each environment by combining task-related reward components,
    # exploration bonuses, and safety-related constraint costs. logs statistical summaries for diagnostics
    # and updates internal episode reward trackers.
    def _get_rewards(self) -> torch.Tensor:
        # compute distances to the three closest obstacles
        distances_to_obstacles = self.maze.check_collision(self.jetbot.data.root_pos_w[:, :2])

        # compute Euclidean distance to nearest goal with optional LOS filtering
        current_distance_to_goal = self._get_distance_from_goal_with_los_check(check_los=False)

        is_new_cell = None  # exploration bonus flag per environment

        if self.reward_scale_exploration is not None:
            # extract current (x, y) positions of each robot
            current_robot_positions_xy = self.jetbot.data.root_pos_w[:, :2]

            # map continuous positions to discrete grid cells
            grid_pos = self._get_cell(current_robot_positions_xy).squeeze(-1)

            # read visitation status from the exploration grid
            grid_values = self.grids[torch.arange(self.cfg.scene.num_envs, device=self.device), grid_pos]

            # identify environments entering a new grid cell
            is_new_cell = (grid_values == 0)

            # mark those cells as visited
            self.grids[torch.arange(self.cfg.scene.num_envs, device=self.device)[is_new_cell], grid_pos[is_new_cell]] = 1

        # compute reward components and total reward
        distance_reward, goal_reward, termination_reward, exploration_reward, primary_reward = compute_rewards(
            self.reward_scale_terminated,
            self.reward_scale_goal,
            self.reward_scale_distance,
            self.cfg.max_distance_from_goal,
            self.jetbot.data.root_pos_w[:, :2],
            self.previous_distance_to_goal,
            current_distance_to_goal,
            self.reward_scale_exploration,
            self.penalty_scale_exploration,
            is_new_cell,
        )

        # update distance tracker for next reward step
        self.previous_distance_to_goal = current_distance_to_goal

        # compute obstacle penalty based on proximity
        constraint_cost = compute_cost(
            self.cost_obstacle,
            self.min_distance_to_obstacle + self.cfg.wheelbase / 2.0,
            distances_to_obstacles,
        )
        self._computed_constraint_cost = constraint_cost

        # update episodic reward/cost accumulators
        self.episode_rewards += primary_reward
        self.episode_costs += constraint_cost
        self.last_episode_costs = constraint_cost

        # compute mean and std for logging purposes
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

        # compute estimated probability of safety under current policy
        safety_probability = self._get_safety_probability().item()

        # compute Lagrangian-adjusted reward
        total_lagrangian_reward = mean_total_reward - self.dual_multiplier * mean_total_cost

        # log all reward statistics
        self.logger.record(
            self.dual_multiplier, mean_distance_reward, std_distance_reward,
            mean_goal_reward, std_goal_reward, mean_termination_reward, std_termination_reward,
            mean_exploration_reward, std_exploration_reward,
            mean_total_reward, std_total_reward,
            mean_total_cost, std_total_cost,
            total_lagrangian_reward, safety_probability
        )

        # return final shaped reward used for optimization
        lag_rew = primary_reward - self.dual_multiplier * constraint_cost
        return lag_rew


    # determines termination conditions for each environment.
    # checks whether the robot timed out, went out of bounds, or reached a goal position.
    # returns two boolean tensors: (done_flag, timeout_flag).
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # check if episode time limit has been reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # check if robot has exited the allowable workspace
        out_of_bounds = torch.any(
            torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal,
            dim=1
        )

        # update failure statistics
        self.failed_robots += torch.sum(out_of_bounds).item()
        self.failed_robots += torch.sum(time_out).item()

        # get robot and goal positions
        robot_pos = self.jetbot.data.root_pos_w[:, :2]
        all_goals = torch.tensor(self.cfg.goal_pos, device=self.jetbot.data.root_pos_w.device)

        # compute Euclidean distance from each robot to each goal
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        dists = torch.norm(diff, dim=2)

        # check if any goal is within threshold distance
        reached_goal = torch.any(dists <= 0.3, dim=1)

        # done when robot has reached goal or gone out of bounds
        return out_of_bounds | reached_goal, time_out


    # resets the state of selected environments (by index) after termination.
    # resets position, orientation, safety buffers, goal distance memory, and exploration grid.
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.jetbot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self.reward_scale_exploration is not None:
            # reset exploration grid values for the specified environments
            self.grids[env_ids] = 0

        # sample collision-free spawn positions for the robots
        num_ids = len(env_ids)
        self.spawned_robots += num_ids

        safe_positions = self.maze.sample_safe_positions(
            num_envs=num_ids,
            x_bounds=(-1.5, 1.5),
            y_bounds=(-3.0, 3.0),
            min_dist=self.min_distance_to_obstacle,
            max_attempts=50,
            safety_margin=0.01
        )

        # access the default root state buffer
        default_root_state = self.jetbot.data.default_root_state[env_ids]
        default_root_state[:, :2] = safe_positions
        default_root_state[:, 2] = self.cfg.spawn_height

        # generate new random yaw orientations around z-axis
        axis = torch.zeros(num_ids, 3, device=self.device)
        axis[:, 2] = 1.0
        angle = torch.rand(num_ids, device=self.device) * 2 * math.pi
        half_angle = angle / 2
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)
        qx = axis[:, 0] * sin_half_angle
        qy = axis[:, 1] * sin_half_angle
        qz = axis[:, 2] * sin_half_angle
        qw = cos_half_angle
        new_orientation = torch.stack([qw, qx, qy, qz], dim=1)

        # set yaw and reset distance to goal
        self.previous_distance_to_goal[env_ids] = torch.zeros(num_ids, device=self.device)
        default_root_state[:, 3:7] = new_orientation

        # write updated pose into the simulator state
        self.jetbot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # reset safety status indicator
        self.safety_indicator[env_ids] = 1
    

    # computes the minimum distance from the robot to any goal, optionally considering line-of-sight (LOS) visibility.
    # if LOS is blocked by a maze wall, a penalty distance is assigned to that goal (default 10.0).
    # returns a tensor of shape [num_envs], each entry being the minimum modified distance across all goals.
    def _get_distance_from_goal_with_los_check(self, check_los=True) -> torch.Tensor:
        # extract 2D (x, y) positions of the robot across all environments
        robot_pos = self.jetbot.data.root_pos_w[:, :2]  # shape: [num_envs, 2]

        # load goal positions from config and convert to tensor on the same device
        all_goals = torch.tensor(self.cfg.goal_pos, device=robot_pos.device, dtype=torch.float32)  # shape: [num_goals, 2]

        # extract dimensions
        num_envs = robot_pos.shape[0]
        num_goals = all_goals.shape[0]

        # handle edge case where no goals are specified
        if num_goals == 0:
            return torch.full((num_envs,), 10.0, device=robot_pos.device, dtype=torch.float32)

        # compute pairwise difference vectors between robot positions and goal positions
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)  # shape: [num_envs, num_goals, 2]

        # compute pairwise euclidean distances
        euclidean_dists = torch.norm(diff, dim=2)  # shape: [num_envs, num_goals]

        if check_los:
            # flatten the start and end point arrays for all env-goal pairs
            line_starts_flat = robot_pos.unsqueeze(1).expand(-1, num_goals, -1).reshape(num_envs * num_goals, 2)
            line_ends_flat = all_goals.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_goals, 2)

            # check whether each robot-goal segment is blocked by any wall
            if self.maze.walls_start_tensor is not None and self.maze.walls_start_tensor.shape[0] > 0:
                is_blocked_flat = self.maze.are_line_segments_blocked_by_walls(line_starts_flat, line_ends_flat)
            else:
                # no walls present; assume all lines are unblocked
                is_blocked_flat = torch.zeros(num_envs * num_goals, dtype=torch.bool, device=robot_pos.device)

            # reshape the blocked information back into [num_envs, num_goals]
            is_blocked_matrix = is_blocked_flat.view(num_envs, num_goals)

            # define the penalty distance for blocked goals
            penalty_distance = torch.tensor(10.0, device=robot_pos.device, dtype=torch.float32)

            # replace distances to blocked goals with penalty
            modified_dists = torch.where(is_blocked_matrix, penalty_distance, euclidean_dists)

            # extract the minimum distance across all goals for each environment
            min_final_dists, _ = torch.min(modified_dists, dim=1)

        else:
            # if no LOS check, just take minimum Euclidean distance
            min_final_dists, _ = torch.min(euclidean_dists, dim=1)

        return min_final_dists

    
    # returns both the minimum distance to any goal and the angular yaw error relative to that goal.
    # goals behind obstacles (if LOS is enabled) are assigned a penalty distance and excluded from yaw computation.
    # used for reward shaping and navigation orientation.
    def _get_distance_and_angle_from_goal_with_los_check(
        self,
        robot_pos: torch.Tensor,
        robot_yaw: torch.Tensor,
        check_los=True
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # goal positions as [num_goals, 2] tensor on the same device as robot positions
        all_goals = torch.tensor(self.cfg.goal_pos, device=robot_pos.device, dtype=torch.float32)

        num_envs = robot_pos.shape[0]
        num_goals = all_goals.shape[0]
        penalty_distance_val = 10.0  # artificial distance assigned to unreachable goals

        if num_goals == 0:
            # no goals available: return penalty distance and zero angle for all environments
            return (
                torch.full((num_envs, 1), penalty_distance_val, device=robot_pos.device),
                torch.zeros((num_envs, 1), device=robot_pos.device)
            )

        # compute pairwise vector differences: shape [num_envs, num_goals, 2]
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        # compute pairwise Euclidean distances to all goals
        euclidean_dists = torch.norm(diff, dim=2)

        if check_los:
            # prepare flattened arrays of start and end points for line-of-sight checks
            line_starts_flat = robot_pos.unsqueeze(1).expand(-1, num_goals, -1).reshape(num_envs * num_goals, 2)
            line_ends_flat = all_goals.unsqueeze(0).expand(num_envs, -1, -1).reshape(num_envs * num_goals, 2)

            # if wall tensors are defined, check which robot-goal rays are blocked
            if self.maze.walls_start_tensor is not None and self.maze.walls_start_tensor.shape[0] > 0:
                is_blocked_flat = self.maze.are_line_segments_blocked_by_walls(line_starts_flat, line_ends_flat)
            else:
                # if no walls are defined, assume no occlusions
                is_blocked_flat = torch.zeros(num_envs * num_goals, dtype=torch.bool, device=robot_pos.device)

            # reshape into [num_envs, num_goals] for indexing
            is_blocked_matrix = is_blocked_flat.view(num_envs, num_goals)
            # replace distances to blocked goals with penalty value
            modified_dists = torch.where(is_blocked_matrix, penalty_distance_val, euclidean_dists)

            # for each environment, select the minimum (modified) distance and index of corresponding goal
            min_final_dists, min_dist_indices = torch.min(modified_dists, dim=1)

            # extract closest goal coordinates for each environment
            effective_closest_goal_xy = all_goals[min_dist_indices]

            # compute vector to closest goal and convert to angle in global frame
            vector_to_goal = effective_closest_goal_xy - robot_pos
            angle_to_goal = torch.atan2(vector_to_goal[:, 1], vector_to_goal[:, 0])

            # compute angular difference in [-pi, pi] between robot yaw and goal direction
            yaw_diff = (angle_to_goal - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi

            # if closest goal was unreachable, zero out yaw error (don't try to rotate)
            is_penalized = (min_final_dists >= penalty_distance_val - 1e-5)
            yaw_diff = torch.where(is_penalized, torch.zeros_like(yaw_diff), yaw_diff)

        else:
            # no LOS checking: simply pick the closest Euclidean goal
            min_final_dists, min_dist_indices = torch.min(euclidean_dists, dim=1)
            closest_goal_xy = all_goals[min_dist_indices]

            # compute angle to closest goal and yaw error
            vector_to_goal = closest_goal_xy - robot_pos
            angle_to_goal = torch.atan2(vector_to_goal[:, 1], vector_to_goal[:, 0])
            yaw_diff = (angle_to_goal - robot_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        # return both distance and yaw error, with shape [num_envs, 1]
        return min_final_dists.unsqueeze(1), yaw_diff.unsqueeze(1)
    

    # returns the 1D grid index of the robot’s current position in the discretized maze.
    # the grid is linearized as row-major from (x_min, y_min) to (x_max, y_max).
    def _get_cell(self, robot_pos: torch.Tensor) -> torch.Tensor:
        x_min, x_max, y_min, y_max = self.cfg.grid_boundaries
        cell_size = self.cfg.grid_cell_size
        max_x = int((x_max - x_min) / cell_size)

        # compute grid indices for each robot position
        # robot_pos is expected to be of shape [num_envs, 2] (x, y)
        grid_x = ((robot_pos[:, 0] - x_min) / cell_size).long()
        grid_y = ((robot_pos[:, 1] - y_min) / cell_size).long()

        # clamp indices to ensure they are within valid range
        grid_x = torch.clamp(grid_x, 0, max_x - 1)
        grid_y = torch.clamp(grid_y, 0, int((y_max - y_min) / cell_size) - 1)

        # compute linearized grid position
        grid_pos = grid_x + max_x * grid_y
        return grid_pos.unsqueeze(-1)


    # extracts the yaw angle (rotation around z-axis) from a batch of quaternions.
    def _compute_yaw(self, quaternions: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw.unsqueeze(-1)
    

    # sets the current value of the Lagrange dual multiplier
    def _set_dual_multiplier(self, value: float) -> None:
        self.dual_multiplier = value


    # returns the currently accumulated constraint cost
    def _get_constraint_cost(self):
        return self._computed_constraint_cost


    # returns the constraint costs from the last episode
    def _get_last_episode_costs(self):
        return self.last_episode_costs


    # returns the average safety indicator across environments
    def _get_safety_probability(self):
        return torch.mean(self.safety_indicator)


    # resets all episode-level reward and cost trackers
    def _reset_reward_and_cost(self):
        self.episode_rewards = torch.zeros(self.n_envs, device=self.device)
        self.episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.n_envs, device=self.device)


    # returns the episode-level rewards and costs
    def _get_reward_and_cost(self):
        return self.episode_rewards, self.episode_costs


    # sets the weight for the distance reward component
    def _set_distance_rew(self, value):
        self.reward_scale_distance = value


    # computes success rate as 1 - (failures / total)
    def _get_success_rate(self) -> float:
        return 1 - (self.failed_robots / self.spawned_robots) if self.spawned_robots > 0 else 0.0




# decorator to compile the function to TorchScript
# https://pytorch.org/docs/stable/jit.html
# useful for performance optimization
# computes reward components and the total reward for reinforcement learning agents navigating to a goal.
# includes components for reaching the goal, making progress toward it, staying within bounds, and exploring new states.
# designed for use with torch.jit.script for performance optimization.
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
    # reward for termination when robot exceeds maximum allowed distance from goal
    terminated = torch.any(torch.abs(robot_pos) > max_distance_from_goal, dim=1).float()
    rew_termination = rew_scale_terminated * terminated

    # reward for reaching the goal region (distance <= 0.3 units)
    rew_goal = rew_scale_goal * (current_distance_to_goal <= 0.3).float()

    # reward for making progress toward the goal (difference in distance)
    rew_distance = torch.where(
        previous_distance_to_goal == 0.0,
        torch.zeros_like(current_distance_to_goal),
        previous_distance_to_goal - current_distance_to_goal
    )
    rew_distance = rew_scale_distance * rew_distance

    # clip reward so that moving away from the goal does not result in penalty
    rew_distance = torch.clamp(rew_distance, min=0.0)

    # reward and penalty for exploration, if exploration tracking is enabled
    if (
        rew_scale_exploration is not None and
        penalty_scale_exploration is not None and
        new_cell is not None
    ):
        # reward for visiting a new cell (binary indicator)
        rew_exploration = rew_scale_exploration * new_cell.float()

        # penalty for revisiting an already seen cell
        penalty_exploration = penalty_scale_exploration * (1.0 - new_cell.float())

        # combined exploration term
        rew_exploration = rew_exploration + penalty_exploration

        # sum all reward components
        total_reward = rew_distance + rew_goal + rew_termination + rew_exploration
    else:
        # exploration term is disabled
        rew_exploration = None
        total_reward = rew_distance + rew_goal + rew_termination

    return rew_distance, rew_goal, rew_termination, rew_exploration, total_reward




# computes safety cost based on proximity to obstacles.
# agents incur a cost if they are closer than a threshold to any obstacle.
# cost increases linearly as the distance decreases, capped at cost_obstacle.
@torch.jit.script
def compute_cost(
    cost_obstacle: float,
    min_dist: float,
    dist_to_obs: torch.Tensor,
):
    # compute per-obstacle cost based on proximity
    cost_obstacles = torch.where(
        dist_to_obs <= min_dist,
        cost_obstacle * (1.0 - dist_to_obs / min_dist),
        torch.tensor(0.0)
    )

    # sum cost over all obstacles per environment
    return cost_obstacles.sum(dim=1)




# defines a Maze class for managing wall geometry, spawning walls in simulation, and sampling valid positions
class Maze:
    def __init__(self):
        # initialize wall data structures
        self.walls = []
        self.walls_start_tensor = None
        self.walls_end_tensor = None
        self.obstacle_prims = []

        # construct absolute path to maze configuration
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "maze_cfg.yaml")

        # load wall configurations from YAML
        with open(self.config_path, "r") as f:
            maze_config = yaml.safe_load(f)
        self.walls_config = maze_config["maze"]["walls"]

        # create tensors for wall start and end points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.walls_config:
            self.walls_start_tensor = torch.tensor(
                [wall["start"] for wall in self.walls_config],
                device=self.device,
                dtype=torch.float32
            )
            self.walls_end_tensor = torch.tensor(
                [wall["end"] for wall in self.walls_config],
                device=self.device,
                dtype=torch.float32
            )
        else:
            # handle empty configuration case with zero-length tensors
            self.walls_start_tensor = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            self.walls_end_tensor = torch.empty((0, 2), device=self.device, dtype=torch.float32)


    # gets the wall configuration as a list of dictionaries.
    def spawn_maze(self, width=0.1, height=0.5, walls_config=None):
        # spawns the maze walls as cuboids in simulation, using the specified wall configuration
        if walls_config is None:
            walls_config = self.walls_config

        self.width = width
        self.height = height

        # move wall tensors to device (CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.walls_start_tensor = torch.tensor(
            [wall["start"] for wall in walls_config],
            device=device,
            dtype=torch.float32
        )
        self.walls_end_tensor = torch.tensor(
            [wall["end"] for wall in walls_config],
            device=device,
            dtype=torch.float32
        )

        # spawn each wall as a cuboid in simulation
        for i, wall in enumerate(walls_config):
            prim_path = f"/World/Maze/wall_{i}"
            self.spawn_wall_cube(prim_path, wall["start"], wall["end"])


    # spawns a single cuboid wall segment between start and end points.
    # the wall is positioned at the midpoint of the segment and oriented along its length.
    def spawn_wall_cube(self, prim_path: str, start: list[float], end: list[float]):
        # spawns a single cuboid wall segment between start and end points

        # compute wall length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        # compute midpoint for positioning
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        mid_z = self.height / 2.0
        translation = (mid_x, mid_y, mid_z)

        size = (length, self.width, self.height)

        # compute yaw orientation (around z-axis) as quaternion
        angle_rad = math.atan2(dy, dx)
        theta_rad = angle_rad
        w = math.cos(theta_rad / 2)
        x = 0.0
        y = 0.0
        z = math.sin(theta_rad / 2)
        orientation = (w, x, y, z)

        # configure cuboid visual and collision properties
        cfg_wall_cube = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.14, 0.35))
        )

        # spawn the cuboid primitive in simulation
        cfg_wall_cube.func(
            prim_path,
            cfg_wall_cube,
            translation=translation,
            orientation=orientation
        )

        return cfg_wall_cube


    # samples a valid (x, y) position for each environment that lies at least `min_dist` from all walls.
    # returns a tensor of shape (num_envs, 2) containing the sampled positions.
    # if no valid position is found after `max_attempts`, returns the last candidate position.
    def sample_safe_positions(
        self,
        num_envs: int,
        x_bounds: Tuple[float, float] = (-1.4, 1.4),
        y_bounds: Tuple[float, float] = (-2.9, 2.9),
        min_dist: float = 0.15,
        max_attempts: int = 100,
        safety_margin: float = 0.01
    ) -> torch.Tensor:

        device = self.walls_start_tensor.device

        # 1) sample random candidate positions: shape (num_envs, max_attempts, 2)
        u = torch.rand((num_envs, max_attempts), device=device)
        v = torch.rand((num_envs, max_attempts), device=device)
        xs = x_bounds[0] + u * (x_bounds[1] - x_bounds[0])
        ys = y_bounds[0] + v * (y_bounds[1] - y_bounds[0])
        candidates = torch.stack((xs, ys), dim=-1)  # shape: (num_envs, max_attempts, 2)

        # 2) compute distances to all walls for all candidates
        flat = candidates.view(-1, 2)                       # shape: (num_envs * max_attempts, 2)
        dists_flat = self.check_collision(flat)             # shape: (num_envs * max_attempts, num_walls)
        k = dists_flat.shape[-1]
        dists = dists_flat.view(num_envs, max_attempts, k)  # shape: (num_envs, max_attempts, num_walls)

        # 3) build mask identifying "safe" candidate positions
        threshold = min_dist + safety_margin
        safe_mask = (dists >= threshold).all(dim=-1)        # shape: (num_envs, max_attempts)

        # 4) find the first safe candidate per environment, fallback to last if none are safe
        any_safe = safe_mask.any(dim=-1)                    # shape: (num_envs,)
        first_idx = torch.argmax(safe_mask.int(), dim=-1)   # shape: (num_envs,)
        fallback_idx = torch.full_like(first_idx, max_attempts - 1)
        pick_idx = torch.where(any_safe, first_idx, fallback_idx)

        # 5) index into candidate array to select the chosen safe (x, y) for each env
        env_idx = torch.arange(num_envs, device=device)
        chosen = candidates[env_idx, pick_idx]              # shape: (num_envs, 2)

        return chosen
        

    # compute the perpendicular distance from a point to the line segment defined by `start` and `end`
    # if `start` and `end` coincide, returns Euclidean distance to that single point
    def distance_from_line(self, point, start, end):
        # convert input point to a tensor if it is a list
        if isinstance(point, list):
            point = torch.tensor(point, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # move tensor to CPU and convert to numpy for vectorized computation
        if point.is_cuda:
            p = np.array(point.cpu())
        else:
            p = np.array(point)

        # convert start and end to numpy arrays
        s = np.array(start)
        e = np.array(end)

        # if start and end are the same point, return Euclidean distance to it
        if np.allclose(s, e):
            return np.linalg.norm(p - s)

        # compute perpendicular distance using vector cross product formula
        return np.abs(np.cross(e - s, s - p)) / np.linalg.norm(e - s)
    

    # compute the signed perpendicular distances from each robot position to each wall,
    # taking into account wall width and clamping to segment endpoints.
    # returns the three smallest distances for each robot.
    def check_collision(self, pos):
        # return inf distances if walls are not defined
        if self.walls_start_tensor is None or self.walls_end_tensor is None:
            return torch.full((pos.shape[0], 3), float('inf'), device=pos.device)

        # get wall endpoints
        walls_start = self.walls_start_tensor      # [num_walls, 2]
        walls_end = self.walls_end_tensor          # [num_walls, 2]
        
        # compute wall directions and their squared norms
        wall_dirs = walls_end - walls_start        # [num_walls, 2]
        norm_wall_dirs_sq = (wall_dirs ** 2).sum(dim=1)  # [num_walls]
        norm_wall_dirs_sq = torch.where(norm_wall_dirs_sq == 0,
                                         torch.tensor(1e-6, device=pos.device),
                                         norm_wall_dirs_sq)  # avoid divide-by-zero
        
        # expand tensors for broadcasting over robots and walls
        pos_exp = pos.unsqueeze(1)                 # [num_envs, 1, 2]
        walls_start_exp = walls_start.unsqueeze(0)  # [1, num_walls, 2]
        wall_dirs_exp = wall_dirs.unsqueeze(0)      # [1, num_walls, 2]
        walls_end_exp = walls_end.unsqueeze(0)      # [1, num_walls, 2]

        # compute vector from wall start to robot position
        vec = pos_exp - walls_start_exp            # [num_envs, num_walls, 2]
        
        # compute projection factor t along wall direction
        dot = (vec * wall_dirs_exp).sum(dim=2)     # [num_envs, num_walls]
        norm_sq = norm_wall_dirs_sq.unsqueeze(0)   # [1, num_walls]
        t = dot / norm_sq                          # [num_envs, num_walls]
        
        # get projected point on infinite line
        p_proj = walls_start_exp + t.unsqueeze(-1) * wall_dirs_exp  # [num_envs, num_walls, 2]
    
        # compute perpendicular distance from robot to line
        d_perp = torch.norm(pos_exp - p_proj, dim=2)  # [num_envs, num_walls]

        # case 1: projection is inside wall segment
        mask_inside = (t >= 0.0) & (t <= 1.0)  # [num_envs, num_walls]
        d_on_segment = torch.clamp(d_perp - self.width / 2.0, min=0.0)  # [num_envs, num_walls]

        # case 2: projection is outside wall segment -> use endpoint or corner
        d_ep_A = torch.norm(p_proj - walls_start_exp, dim=2)  # [num_envs, num_walls]
        d_ep_B = torch.norm(p_proj - walls_end_exp, dim=2)    # [num_envs, num_walls]
        d_ep = torch.where(t < 0, d_ep_A, d_ep_B)             # select appropriate endpoint
        
        # corner distance = sqrt( d_ep^2 + (d_perp - w/2)^2 )
        corner_distance = torch.sqrt(d_ep**2 + (d_perp - self.width / 2.0)**2)
        d_outside = torch.where(d_perp < self.width / 2.0, d_ep, corner_distance)

        # choose distances depending on projection location
        final_dists = torch.where(mask_inside, d_on_segment, d_outside)  # [num_envs, num_walls]

        # sort distances to get the 3 closest walls
        sorted_dists, _ = torch.sort(final_dists, dim=1)
        smallest3 = sorted_dists[:, :3]

        return smallest3  # [num_envs, 3]

    

    # computes the distance and angle to the three closest walls for each robot position.
    # returns a tuple of distances and angles, both tensors of shape [num_envs, 3].
    # distances are the signed perpendicular distances to the walls,
    # and angles are the yaw differences to the wall normals.
    def check_collision_ang(self, pos, yaw):
        # handle case when walls are not defined by returning +inf distances and zero angles
        if self.walls_start_tensor is None or self.walls_end_tensor is None:
            ret = torch.full((pos.shape[0], 3), float('inf'), device=pos.device), torch.zeros((pos.shape[0], 3), device=pos.device)
            return ret

        # add singleton dimension to yaw to broadcast over walls later
        yaw.unsqueeze_(1)  # shape: [num_envs, 1]

        # get wall endpoints [num_walls, 2]
        walls_start = self.walls_start_tensor
        walls_end = self.walls_end_tensor

        # compute direction vectors of walls [num_walls, 2]
        wall_dirs = walls_end - walls_start

        # compute squared norms of wall direction vectors [num_walls]
        norm_wall_dirs_sq = (wall_dirs ** 2).sum(dim=1)

        # avoid division by zero in degenerate walls
        norm_wall_dirs_sq = torch.where(norm_wall_dirs_sq == 0, torch.tensor(1e-6, device=pos.device), norm_wall_dirs_sq)

        # expand dimensions to compute pairwise robot-wall projections
        pos_exp = pos.unsqueeze(1)              # [num_envs, 1, 2]
        walls_start_exp = walls_start.unsqueeze(0)  # [1, num_walls, 2]
        wall_dirs_exp = wall_dirs.unsqueeze(0)      # [1, num_walls, 2]
        walls_end_exp = walls_end.unsqueeze(0)      # [1, num_walls, 2]

        # vector from wall start to robot position [num_envs, num_walls, 2]
        vec = pos_exp - walls_start_exp

        # scalar projection factor along wall direction: t = dot(P-A, B-A) / ||B-A||^2
        dot = (vec * wall_dirs_exp).sum(dim=2)       # [num_envs, num_walls]
        t = dot / norm_wall_dirs_sq.unsqueeze(0)     # [num_envs, num_walls]

        # project robot position onto the infinite extension of wall: A + t(B-A)
        p_proj = walls_start_exp + t.unsqueeze(-1) * wall_dirs_exp  # [num_envs, num_walls, 2]

        # perpendicular distance from robot to projected point [num_envs, num_walls]
        d_perp = torch.norm(pos_exp - p_proj, dim=2)

        # identify projections falling inside the wall segment (0 <= t <= 1)
        mask_inside = (t >= 0.0) & (t <= 1.0)

        # for inside projections, subtract half wall width to get actual clearance
        d_on_segment = torch.clamp(d_perp - self.width / 2.0, min=0.0)

        # compute distance to wall endpoints (used for t < 0 and t > 1 cases)
        d_ep_A = torch.norm(p_proj - walls_start_exp, dim=2)  # distance to start
        d_ep_B = torch.norm(p_proj - walls_end_exp, dim=2)    # distance to end

        # select which endpoint to use based on t
        d_ep = torch.where(t < 0, d_ep_A, d_ep_B)

        # compute corner distance: hypotenuse of d_ep and lateral offset from wall edge
        corner_distance = torch.sqrt(d_ep**2 + (d_perp - self.width / 2.0)**2)

        # if robot is beside the wall (close but not past ends), use d_ep; else use full corner distance
        d_outside = torch.where(d_perp < self.width / 2.0, d_ep, corner_distance)

        # combine distances based on whether projection falls inside segment or outside
        final_dists = torch.where(mask_inside, d_on_segment, d_outside)

        # sort all wall distances and select top-3 closest ones [num_envs, 3]
        sorted_dists, sorted_indices = torch.sort(final_dists, dim=1)
        smallest3 = sorted_dists[:, :3]
        top3_indices = sorted_indices[:, :3]

        # compute projection points corresponding to the top-3 closest walls
        batch_indices = torch.arange(pos.shape[0], device=pos.device).unsqueeze(1).expand(-1, 3)
        p_proj_closest = p_proj[batch_indices, top3_indices]       # [num_envs, 3, 2]

        # vectors from robot to projection points [num_envs, 3, 2]
        vectors_to_obs = p_proj_closest - pos.unsqueeze(1)

        # compute direction of these vectors as angles (global frame) [num_envs, 3]
        obs_angles = torch.atan2(vectors_to_obs[:, :, 1], vectors_to_obs[:, :, 0])

        # compute yaw difference, normalized to [-π, π] [num_envs, 3]
        angle_diffs = obs_angles - yaw
        angle_diffs = (angle_diffs + torch.pi) % (2 * torch.pi) - torch.pi

        return smallest3, angle_diffs

    
    # computes the orientation of the triplet (p, q, r).
    # returns 0 if collinear, 1 if clockwise, -1 if counterclockwise.
    @staticmethod
    def _orientation_static(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - \
              (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
        return torch.sign(val)


    # given three collinear points p, q, r, checks if point q lies on segment 'pr'.
    @staticmethod
    def _on_segment_static(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # add a small epsilon for floating point comparisons
        eps = 1e-6
        return (q[..., 0] <= torch.maximum(p[..., 0], r[..., 0]) + eps) & \
               (q[..., 0] >= torch.minimum(p[..., 0], r[..., 0]) - eps) & \
               (q[..., 1] <= torch.maximum(p[..., 1], r[..., 1]) + eps) & \
               (q[..., 1] >= torch.minimum(p[..., 1], r[..., 1]) - eps)


    # checks for intersections between two sets of line segments using the orientation method.
    # segments are defined by their start and end points.
    def check_line_segment_intersections(self,
                                         seg1_starts: torch.Tensor,  # [N, 2]
                                         seg1_ends: torch.Tensor,    # [N, 2]
                                         seg2_starts: torch.Tensor,  # [M, 2]
                                         seg2_ends: torch.Tensor     # [M, 2]
                                         ) -> torch.Tensor:          # [N, M] boolean, True if intersection
        N = seg1_starts.shape[0]
        M = seg2_starts.shape[0]
        device = seg1_starts.device

        if N == 0 or M == 0:
            return torch.zeros((N, M), dtype=torch.bool, device=device)

        # expand dimensions for broadcasting
        p1 = seg1_starts.unsqueeze(1)  # [N, 1, 2]
        q1 = seg1_ends.unsqueeze(1)    # [N, 1, 2]
        p2 = seg2_starts.unsqueeze(0)  # [1, M, 2]
        q2 = seg2_ends.unsqueeze(0)    # [1, M, 2]

        # calculate orientations
        o1 = Maze._orientation_static(p1, q1, p2)  # orientation(p1, q1, p2)
        o2 = Maze._orientation_static(p1, q1, q2)  # orientation(p1, q1, q2)
        o3 = Maze._orientation_static(p2, q2, p1)  # orientation(p2, q2, p1)
        o4 = Maze._orientation_static(p2, q2, q1)  # orientation(p2, q2, q1)

        # general case: segments cross each other
        # this happens if (p1, q1, p2) and (p1, q1, q2) have different orientations,
        # AND (p2, q2, p1) and (p2, q2, q1) have different orientations.
        # o1 * o2 < 0 means different non-zero orientations.
        general_intersects = (o1 * o2 < 0) & (o3 * o4 < 0)

        # special Cases: collinear intersections
        # case 1: p1, q1, p2 are collinear and p2 lies on segment p1q1
        collinear_case1 = (o1 == 0) & Maze._on_segment_static(p1, p2, q1)
        # case 2: p1, q1, q2 are collinear and q2 lies on segment p1q1
        collinear_case2 = (o2 == 0) & Maze._on_segment_static(p1, q2, q1)
        # case 3: p2, q2, p1 are collinear and p1 lies on segment p2q2
        collinear_case3 = (o3 == 0) & Maze._on_segment_static(p2, p1, q2)
        # case 4: p2, q2, q1 are collinear and q1 lies on segment p2q2
        collinear_case4 = (o4 == 0) & Maze._on_segment_static(p2, q1, q2)
        
        collinear_intersects = collinear_case1 | collinear_case2 | collinear_case3 | collinear_case4
        
        return general_intersects | collinear_intersects


    # checks if multiple line segments are blocked by any wall in the maze.
    def are_line_segments_blocked_by_walls(self, seg_starts: torch.Tensor, seg_ends: torch.Tensor) -> torch.Tensor:
        if self.walls_start_tensor is None or self.walls_start_tensor.shape[0] == 0:
            return torch.zeros(seg_starts.shape[0], dtype=torch.bool, device=seg_starts.device)
        
        # ensure wall tensors are on the same device as segments
        wall_starts_device = self.walls_start_tensor.to(seg_starts.device)
        wall_ends_device = self.walls_end_tensor.to(seg_starts.device)

        # check intersections between each segment and all walls
        # resulting shape: [K, num_walls]
        intersections_matrix = self.check_line_segment_intersections(
            seg_starts, seg_ends,
            wall_starts_device, wall_ends_device
        )
        # a segment is blocked if it intersects with ANY wall
        return intersections_matrix.any(dim=1)




# converts base linear and angular velocity commands into left and right wheel velocities for a differential drive robot.
def direct_diff_drive(linear_velocity: float, angular_velocity: float, wheelbase: float, wheel_radius: float) -> tuple[float, float]:
    # compute left and right wheel linear velocities
    v_left = linear_velocity - (angular_velocity * wheelbase / 2)
    v_right = linear_velocity + (angular_velocity * wheelbase / 2)

    # convert linear velocities to angular velocities
    omega_left = v_left / wheel_radius
    omega_right = v_right / wheel_radius

    return omega_left, omega_right




# converts left and right wheel angular velocities into base linear and angular velocity commands for a differential drive robot.
def inverse_diff_drive(omega_left: float, omega_right: float, wheelbase: float, wheel_radius: float) -> tuple[float, float]:
    # compute linear velocity
    linear_velocity = (wheel_radius / 2) * (omega_left + omega_right)

    # compute angular velocity
    angular_velocity = (wheel_radius / wheelbase) * (omega_right - omega_left)

    return linear_velocity, angular_velocity