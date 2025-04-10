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

import csv


class CustomCSVLogger:
    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"iter_reward_log_{timestamp}.csv")
        self.header = ["dual", "dist_rewards", "dr_std", "goal_reward", "gr_std", "term_reward", "tr_std",
                       "move_reward", "mr_std", "mean_reward", "std_reward", "mean_cost", "std_cost",
                       "total_reward", "safety_prob"]
        self._write_header()

    def _write_header(self):
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def record(self, dual: float, dist_reward: float, dist_std: float, goal_reward: float, goal_std: float,
               term_reward: float, term_std: float, move_reward: float, move_std: float,
               mean_reward: float, std_reward: float, mean_cost: float, std_cost: float,
               total_reward: float, safety_prob: float) -> None:
        data = [dual, dist_reward, dist_std, goal_reward, goal_std, term_reward, term_std, move_reward, move_std,
                mean_reward, std_reward, mean_cost, std_cost, total_reward, safety_prob]
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
    # Environment
    decimation: int = 1
    episode_length_s: float = 20.0
    action_scale: float = 1.0
    action_space: int = 2
    # observation_space: int = 10 # x, y, distance_to_goal_x, distance_to_goal_y, yaw_diff, base_lin_vel, base_ang_vel, 3*distance_to_obstacle
    #observation_space: int = 9 # x, y, distance_to_goal, yaw_diff, base_lin_vel, base_ang_vel, 3*distance_to_obstacle
    observation_space: int = 12  # x, y, distance_to_goal, yaw_diff, base_lin_vel, base_ang_vel, 3*distance_to_obstacle, 3*angle_diff
    state_space: int = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Robot
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/JetBot")
    left_wheel_joint_name: str = "left_wheel_joint"
    right_wheel_joint_name: str = "right_wheel_joint"

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)

    # Reset
    max_distance_from_goal: float = 7.0
    initial_pose_min_distance: float = 2.8
    initial_pose_radius: float = 5.0
    spawn_height: float = 0.0437

    # World
    wall_width: float = 0.8
    goal_pos: list[float] = [0.0, 0.0]

    # Robot Geometry
    wheelbase: float = 0.12
    wheelradius: float = 0.032


class JetBotEnv(DirectRLEnv):
    cfg: JetBotEnvCfg

    def __init__(self, cfg: JetBotEnvCfg, render_mode: str | None = None, **kwargs):
        self.maze = Maze()
        super().__init__(cfg, render_mode, **kwargs)

        self._left_wheel_idx, _ = self.jetbot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_wheel_idx, _ = self.jetbot.find_joints(self.cfg.right_wheel_joint_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.jetbot.data.joint_pos
        self.joint_vel = self.jetbot.data.joint_vel

        self.previous_distance_to_goal = torch.zeros(self.num_envs, device=self.device)

        self.previous_position = torch.zeros(self.num_envs, 2, device=self.device)

        config_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_lagrangian_cfg.yaml")
        print("Config path: ", config_path)
        try:
            with open(config_path) as stream:
                data = yaml.safe_load(stream)
                self.gamma = data["agent"]["discount_factor"]
                self.reward_scale_goal = data["rewards"]["goal"]
                self.reward_scale_distance = data["rewards"]["distance"]
                self.reward_scale_terminated = data["rewards"]["termination"]
                self.reward_scale_time = data["rewards"].get("time")  # Use .get() for optional key
                self.reward_scale_movement = data["rewards"].get("movement")
                self.movement_threshold = data["rewards"].get("movement_threshold")
                self.cost_obstacle = data["costs"]["obstacle"]
                self.min_distance_to_obstacle = data["min_dist"]
        except yaml.YAMLError as exc:
            print(f"Error loading YAML config: {exc}. Using default reward/cost parameters.")
            self.gamma = 0.99
            self.reward_scale_goal = 30.0
            self.reward_scale_distance = 50.0
            self.reward_scale_terminated = -80.0
            self.reward_scale_time = None
            self.reward_scale_movement = None
            self.movement_threshold = None
            self.cost_obstacle = 1.0
            self.min_distance_to_obstacle = 0.18

        print("Gamma: ", self.gamma)
        print("Reward scale goal: ", self.reward_scale_goal)
        print("Reward scale distance: ", self.reward_scale_distance)
        print("Reward scale terminated: ", self.reward_scale_terminated)
        print("Cost obstacle: ", self.cost_obstacle)
        print("Min dist to obstacle: ", self.min_distance_to_obstacle)

        self.dual_multiplier = 0.0
        self.n_envs = self.cfg.scene.num_envs  # Use num_envs from config

        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.episode_costs = torch.zeros(self.num_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.num_envs, device=self.device)
        self.safety_indicator = torch.ones(self.num_envs, device=self.device)

        self.logger = CustomCSVLogger()

    def _setup_scene(self):
        self.jetbot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add a cube as a goal
        goal_size = (0.5, 0.5, 0.5)
        goal_pos = self.cfg.goal_pos + [goal_size[2] / 2.0]
        cfg_goal_cube = sim_utils.CuboidCfg(
            size=goal_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76 , 0.2, 0.92)),
        )
        cfg_goal_cube.func(f"/World/Objects/goal", cfg_goal_cube, translation=tuple(goal_pos))

                # add spawn area and bound are circles
        spawn_area_color = (0.1, 0.8, 0.1)  # green
        bound_area_color = (0.8, 0.1, 0.1)  # red
        
        cfg_spawn_area = sim_utils.CylinderCfg(
            radius=self.cfg.initial_pose_radius,
            height=0.006,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spawn_area_color),
        )
        cfg_spawn_area.func("/World/Decals/spawn_area", cfg_spawn_area, translation=(0, 0, 0.003))

        cfg_bound_area = sim_utils.CylinderCfg(
            radius=self.cfg.max_distance_from_goal,
            height=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bound_area_color),
        )
        cfg_bound_area.func("/World/Decals/bound_area", cfg_bound_area, translation=(0, 0, 0.0025))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["jetbot"] = self.jetbot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.maze.spawn_maze(width=self.cfg.wall_width)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self) -> None:
        left_action = self.actions[:, 0].unsqueeze(-1)
        right_action = self.actions[:, 1].unsqueeze(-1)
        self.jetbot.set_joint_velocity_target(left_action, joint_ids=self._left_wheel_idx)
        self.jetbot.set_joint_velocity_target(right_action, joint_ids=self._right_wheel_idx)

    def _get_observations(self) -> dict:
        # Get robot position (x, y)
        position = self.jetbot.data.root_pos_w[:, :2]

        # Compute robot yaw
        orientation = self.jetbot.data.root_quat_w
        yaw = self._compute_yaw(orientation).squeeze(-1)

        # Check collisions and get distances and angles to obstacles
        distance_to_obstacles, obstacle_angle_difference = self.maze.check_collision_ang(position, yaw)

        # Compute safety indicator
        too_close = (distance_to_obstacles < self.min_distance_to_obstacle).any(dim=1)
        self.safety_indicator[too_close] = 0

        # Get wheel velocities
        left_wheel_velocity = self.joint_vel[:, self._left_wheel_idx].squeeze(-1).unsqueeze(-1)
        right_wheel_velocity = self.joint_vel[:, self._right_wheel_idx].squeeze(-1).unsqueeze(-1)

        # Compute distance to goal
        goal_xy = torch.tensor(self.cfg.goal_pos[:2], device=position.device).unsqueeze(0).expand(self.num_envs, -1)
        vector_to_goal = goal_xy - position
        distance_to_goal = torch.norm(vector_to_goal, dim=1, keepdim=True)

        # Compute angle to goal
        angle_to_goal = torch.atan2(vector_to_goal[:, 1], vector_to_goal[:, 0]).unsqueeze(-1) # Shape [4096, 1]

        # Compute yaw difference to goal
        yaw_difference = (angle_to_goal - yaw + torch.pi) % (2 * torch.pi) - torch.pi

        # print("Position: ", position.shape)
        # print("Distance to goal: ", distance_to_goal.shape)
        # print("Yaw difference: ", yaw_difference.shape)
        # print("Left wheel velocity: ", left_wheel_velocity.shape)
        # print("Right wheel velocity: ", right_wheel_velocity.shape)
        # print("Distance to obstacles: ", distance_to_obstacles.shape)
        # print("Obstacle angle difference: ", obstacle_angle_difference.shape)

        # Concatenate observations
        observation = torch.cat((
            position,
            distance_to_goal,
            yaw_difference,
            left_wheel_velocity,
            right_wheel_velocity,
            distance_to_obstacles,
            obstacle_angle_difference,
        ), dim=-1)

        return {"policy": observation}
    
    def _get_rewards(self) -> torch.Tensor:
        # Compute distances to obstacles
        distances_to_obstacles = self.maze.check_collision(self.jetbot.data.root_pos_w[:, :2])

        # Compute individual reward components and total primary reward
        distance_reward, goal_reward, termination_reward, movement_reward, primary_reward, current_distance_to_goal = compute_rewards(
            self.reward_scale_terminated,
            self.reward_scale_goal,
            self.reward_scale_distance,
            self.reward_scale_time,
            self.reward_scale_movement,
            self.movement_threshold,
            self.jetbot.data.root_pos_w[:, :2],  # Robot root position (x, y)
            self.cfg.goal_pos[:2],  # Goal position (x, y)
            self.cfg.initial_pose_radius,  # Max distance for distance penalty
            self.previous_position,  # Previous position
            self.previous_distance_to_goal,  # Previous distance to goal
            self.episode_length_buf,  # Episode length buffer
            self.gamma  # Discount factor
        )

        # Update previous distance to goal for the next step
        self.previous_distance_to_goal = current_distance_to_goal
        
        self.previous_position = self.jetbot.data.root_pos_w[:, :2].clone()

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
        mean_movement_reward = movement_reward.mean().item()
        std_distance_reward = distance_reward.std().item()
        std_goal_reward = goal_reward.std().item()
        std_termination_reward = termination_reward.std().item()
        std_movement_reward = movement_reward.std().item()

        mean_total_reward = primary_reward.mean().item()
        std_total_reward = primary_reward.std().item()
        mean_total_cost = constraint_cost.mean().item()
        std_total_cost = constraint_cost.std().item()

        # Get safety probability
        safety_probability = self._get_safety_probability().item()

        # Log the reward and cost information
        total_lagrangian_reward = mean_total_reward - self.dual_multiplier * mean_total_cost
        self.logger.record(self.dual_multiplier, mean_distance_reward, std_distance_reward,
                           mean_goal_reward, std_goal_reward, mean_termination_reward, std_termination_reward,
                           mean_movement_reward, std_movement_reward,
                           mean_total_reward, std_total_reward, mean_total_cost, std_total_cost,
                           total_lagrangian_reward, safety_probability)

        # Return the Lagrangian reward
        return primary_reward - self.dual_multiplier * constraint_cost
        # return primary_reward # Uncomment to return only the primary reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check if the episode has run out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if the robot has gone out of bounds
        out_of_bounds = torch.any(torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal, dim=1)

        # Check if the robot has reached the goal
        reached_goal = torch.norm(self.jetbot.data.root_pos_w[:, :2]
                                  - torch.tensor(self.cfg.goal_pos[:2], device=self.jetbot.data.root_pos_w.device), dim=1) <= 0.1

        # Episode is done if any of the conditions are met
        dones = out_of_bounds | reached_goal
        return dones, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.jetbot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Sample a new starting position within initial_pose_radius around the goal
        num_ids = len(env_ids)

        # Random angle and radius around goal
        theta = torch.rand(num_ids, device=self.device) * 2 * math.pi
        r = torch.sqrt(
            torch.rand(num_ids, device=self.device) *
            (self.cfg.initial_pose_radius**2 - self.cfg.initial_pose_min_distance**2) +
            self.cfg.initial_pose_min_distance**2
        )

        # Transform polar coordinates to Cartesian coordinates
        new_x = self.cfg.goal_pos[0] + r * torch.cos(theta)
        new_y = self.cfg.goal_pos[1] + r * torch.sin(theta)
        new_z = torch.ones_like(new_x) * self.cfg.spawn_height
        new_positions = torch.stack([new_x, new_y, new_z], dim=1)

        # Reset previous distance
        self.previous_distance_to_goal[env_ids] = torch.zeros(num_ids, device=self.device)

        # Reset previous position
        self.previous_position[env_ids] = torch.zeros(num_ids, 2, device=self.device)

        # Sample a new starting yaw angle
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

        # Apply new poses to the robots
        default_root_state = self.jetbot.data.default_root_state[env_ids].clone() # Use clone to avoid modifying original
        default_root_state[:, :3] = new_positions
        default_root_state[:, 3:7] = new_orientation

        self.jetbot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # reset safety indicator for the reset environments
        self.safety_indicator[env_ids] = 1

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
        return self.episode_costs
    
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

    def _domain_randomize(self):
        self.maze.move_walls(0.1)
    
    def _set_obstacle_size(self, size: float):
        self.maze.set_obstacle_size(size)

    def _increase_obstacle_size(self, percentage: float):
        self.maze.increase_obstacle_size(percentage)
    
    def _change_obstacle_size(self, size: float):
        self.maze.change_obstacle_size(size)


# decorator to compile the function to TorchScript
# https://pytorch.org/docs/stable/jit.html
# useful for performance optimization
@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_scale_goal: float,
    rew_scale_distance: float,
    rew_scale_time: float | None,
    rew_scale_movement: float | None,
    movement_threshold: float | None,
    robot_pos: torch.Tensor,
    goal_pos: list[float],
    initial_pose_radius: float,
    prec_pos: torch.Tensor,
    prec_dist: torch.Tensor,
    episode_length_buf: torch.Tensor,
    gamma: float
):
    # compute the distance to the goal
    dist_to_goal = torch.norm(robot_pos - torch.tensor(goal_pos, device=robot_pos.device), dim=1) 

    # compute the rewards
    terminated = (dist_to_goal >= initial_pose_radius).float()
    rew_termination = rew_scale_terminated * terminated

    rew_goal = rew_scale_goal * (dist_to_goal <= 0.1).float() #* (gamma ** episode_length_buf)/(1.0 - gamma)

    rew_distance = torch.where((prec_dist == 0.0), torch.zeros_like(dist_to_goal), prec_dist - dist_to_goal)

    rew_distance = rew_scale_distance * rew_distance

    total_reward = rew_termination + rew_goal + rew_distance

    if rew_scale_movement is not None and movement_threshold is not None:
        mask = (prec_pos == 0).all(dim=1, keepdim=True)
        movement = torch.where(mask, torch.zeros_like(prec_pos[:, :1]), torch.norm(robot_pos - prec_pos, dim=1, keepdim=True))
        rew_movement = torch.where(movement < movement_threshold, rew_scale_movement * (movement - movement_threshold),
                                   torch.zeros_like(movement))
        total_reward += rew_movement.view(-1)
        return rew_distance, rew_goal, rew_termination, rew_movement, total_reward, dist_to_goal

    if rew_scale_time is not None:
        rew_time = dist_to_goal * torch.exp(rew_scale_time * episode_length_buf)
        total_reward = rew_termination + rew_goal + rew_distance
        return rew_distance, rew_goal, rew_termination, rew_time, total_reward, dist_to_goal

    total_reward = rew_termination + rew_goal + rew_distance
    return rew_distance, rew_goal, rew_termination, torch.zeros_like(rew_distance), total_reward, dist_to_goal

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
        self.walls_start_base = None  # store the original start points
        self.walls_end_base = None 
        self.walls_start_tensor = None # store the scaled start points
        self.walls_end_tensor = None
        self.obstacle_prims = []

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "obs_cfg.yaml")

        with open(self.config_path, "r") as f:
            maze_config = yaml.safe_load(f)
        self.walls_config = maze_config["maze"]["walls"]
        
        self.obstacle_scale_factor = 1.0
    
    def _scale_wall(self, start, end, scale_factor):
        """Scales the wall line by moving both start and end towards the center."""
        start_np = np.array(start)
        end_np = np.array(end)
        center = (start_np + end_np) / 2.0
        original_vector_half = (end_np - center)
        original_half_length = np.linalg.norm(original_vector_half)

        if original_half_length < 1e-6:  # Avoid division by zero for very short walls
            return start_np, end_np

        scaled_half_length = original_half_length * scale_factor
        scaled_vector_half = original_vector_half * (scaled_half_length / original_half_length)

        scaled_start = center - scaled_vector_half
        scaled_end = center + scaled_vector_half
        return scaled_start, scaled_end

    def spawn_maze(self, width=0.1, height=0.5, walls_config=None):
        if walls_config is None:
            walls_config = self.walls_config

        self.width = width
        self.height = height

        self.walls_start_base = torch.tensor([wall["start"] for wall in walls_config],
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                 dtype=torch.float)
        self.walls_end_base = torch.tensor([wall["end"] for wall in walls_config],
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               dtype=torch.float)

        self._update_scaled_walls()

        for i, wall_config in enumerate(walls_config):
            prim_path = f"/World/Maze/wall_{i}"
            cfg = self.spawn_wall_cube(prim_path, wall_config["start"], wall_config["end"])
            if cfg:
                self.obstacle_prims.append(prim_path) # Store the prim path
        
    def _update_scaled_walls(self):
        """Updates the scaled start and end points of the walls."""
        if self.walls_start_base is None or self.walls_end_base is None:
            return

        scaled_starts = []
        scaled_ends = []
        for i in range(len(self.walls_start_base)):
            start_np = self.walls_start_base[i].cpu().numpy()
            end_np = self.walls_end_base[i].cpu().numpy()
            scaled_start_np, scaled_end_np = self._scale_wall(start_np, end_np, self.obstacle_scale_factor)
            scaled_starts.append(scaled_start_np.tolist())
            scaled_ends.append(scaled_end_np.tolist())

        self.walls_start_tensor = torch.tensor(scaled_starts,
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                 dtype=torch.float)
        self.walls_end_tensor = torch.tensor(scaled_ends,
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               dtype=torch.float)

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
        return cfg_wall_cube
        
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

        if self.obstacle_scale_factor <= 0.0:
            return torch.full((pos.shape[0], 3), float('inf'), device=pos.device)
        
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

        if self.obstacle_scale_factor <= 0.0:
            ret = torch.full((pos.shape[0], 3), float('inf'), device=pos.device), torch.zeros((pos.shape[0], 3), device=pos.device)
            print("SHAPE: ", ret[0].shape)
            print("SHAPE: ", ret[1].shape)
            return ret
        

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
    
    def remove_walls(self):
        """Removes the spawned obstacle prims from the scene."""
        for prim_path in self.obstacle_prims:
            sim_utils.remove_prim(prim_path)
        self.obstacle_prims = []
        self.walls_start_tensor = None
        self.walls_end_tensor = None
    
    def move_walls(self, percent):
        new_walls = []
        for wall in self.walls:
            start = wall["start"]
            end = wall["end"]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            new_start = [start[0] + dx * percent, start[1] + dy * percent]
            new_end = [end[0] + dx * percent, end[1] + dy * percent]
            new_walls.append({"start": new_start, "end": new_end})
        
    def set_obstacle_size(self, size):
        self.obstacle_scale_factor = size

    def change_obstacle_size(self, size):
        self.obstacle_scale_factor = size
        # self.remove_walls()
        # self.spawn_maze(self.obstacle_scale_factor)
        self._update_scaled_walls()
        print(f"Obstacle size changed to {self.obstacle_scale_factor}")
        
    def increase_obstacle_size(self, percent):
        self.obstacle_scale_factor += percent
        # self.remove_walls()
        # self.spawn_maze(self.obstacle_scale_factor)
        self._update_scaled_walls()
        print(f"Obstacle size increased to {self.obstacle_scale_factor}")

    



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
