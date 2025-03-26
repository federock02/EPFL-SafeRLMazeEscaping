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

@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 20.0
    action_scale = 1.0
    action_space = 2
    observation_space = 10 # x, y, yaw_diff, left_wheel_vel, right_wheel_vel, distance_to_goal_x, distance_to_goal_y, 3*distance_to_obstacle
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    # the path represent the position of the robot prim in the hierarchical representation of the scene
    # using regex to match the robot prim in all environments for multi-environment training
    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/JetBot")
    left_wheel_joint_name = "left_wheel_joint"
    right_wheel_joint_name = "right_wheel_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)
    
    # reset
    max_distance_from_goal = 7.0 # jetbot moved too far from goal
    initial_pose_min_distance = 2.8  # the minimum distance from the goal
    initial_pose_radius = 5.0  # the radius around goal in which the jetbot initial position is sampled

    wall_width = 0.8

    # goal
    goal_pos = [0.0, 0.0]

    # spawn height
    spawn_height = 0.0437

    # robot geometry
    wheelbase = 0.12
    wheelradius = 0.032


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
        self.prec_dist = torch.zeros(self.jetbot.data.root_pos_w.shape[0], device=self.device)

        # gamma for reward computation
        config_path = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_lagrangian_cfg.yaml")
        print("Config path: ", config_path)
        with open(config_path) as stream:
            try:
                data = yaml.safe_load(stream)
                self.gamma = data["agent"]["discount_factor"]
                self.rew_scale_goal = data["rewards"]["goal"]
                self.rew_scale_distance = data["rewards"]["distance"]
                self.rew_scale_terminated = data["rewards"]["termination"]
                self.rew_scale_time = data["rewards"]["time"]
                self.cost_obstacle = data["costs"]["obstacle"]
                self.min_dist_to_obstacle = data["min_dist"]
            except yaml.YAMLError as exc:
                self.gamma = 0.99
                self.rew_scale_goal = 30.0
                self.rew_scale_distance = 50.0
                self.rew_scale_terminated = -80.0
                self.rew_scale_time = None
                self.cost_obstacle = 1.0
                self.min_dist_to_obstacle = 0.18
        
        print("Gamma: ", self.gamma)
        print("Rew scale goal: ", self.rew_scale_goal)
        print("Rew scale distance: ", self.rew_scale_distance)
        print("Rew scale terminated: ", self.rew_scale_terminated)
        print("Cost obstacle: ", self.cost_obstacle)
        print("Min dist to obstacle: ", self.min_dist_to_obstacle)

        self.dual_multiplier = 0.0
        self.n_envs = 16

        # logging
        self.episode_rewards = torch.zeros(self.n_envs, device=self.device)
        self.episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.safety_indicator = torch.ones(self.n_envs, device=self.device)

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
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # assume actions are the wheel velocities
        left_action = self.actions[:, 0].unsqueeze(-1)
        right_action = self.actions[:, 1].unsqueeze(-1)
        #print("Left action: ", left_action)
        #print("Right action: ", right_action)
        # set the wheel velocities
        self.jetbot.set_joint_velocity_target(left_action, joint_ids=self._left_wheel_idx)
        self.jetbot.set_joint_velocity_target(right_action, joint_ids=self._right_wheel_idx)

    def _get_observations(self) -> dict:
        # gethering observations
        # position: x, y from the root position
        pos = self.jetbot.data.root_pos_w[:, :2]
        #print("Robot pos: ", pos)

        dist_to_obs = self.maze.check_collision(pos)

        # compute safety indicator
        too_close_mask = (dist_to_obs < self.min_dist_to_obstacle).any(dim=1)
        self.safety_indicator[too_close_mask] = 0

        # orientation: compute yaw from the root orientation
        orient = self.jetbot.data.root_quat_w
        yaw = self._compute_yaw(orient).squeeze(-1)  # helper function defined below.
        #print("Robot yaw: ", yaw)

        # wheel velocities
        left_wheel_vel = self.joint_vel[:, self._left_wheel_idx].squeeze(-1).unsqueeze(-1)
        right_wheel_vel = self.joint_vel[:, self._right_wheel_idx].squeeze(-1).unsqueeze(-1)

        base_v, base_omega = inverse_diff_drive(left_wheel_vel, right_wheel_vel, self.cfg.wheelbase, self.cfg.wheelradius)

        # distance to goal (use only x, y coordinates)
        goal_xy = torch.tensor(self.cfg.goal_pos[:2], device=pos.device)
        goal_xy = goal_xy.unsqueeze(0).expand(pos.shape[0], -1)

        # in order to compute the distance to the goal, we need to convert the goal position to a tensor
        # the goal position tensor needs to be on the same device as the observations
        # compute distance to goal per robot
        dist_to_goal = pos - goal_xy  # shape: [256, 2]
        # extract x and y differences and ensure each has shape [256, 1]
        dist_to_goal_x = dist_to_goal[:, 0].unsqueeze(-1)
        dist_to_goal_y = dist_to_goal[:, 1].unsqueeze(-1)
        # print("Distance to goal: ", dist_to_goal_x, dist_to_goal_y)

        # compute the angle to the goal using atan2
        goal_direction = torch.stack((dist_to_goal_x.squeeze(-1), dist_to_goal_y.squeeze(-1)), dim=1)
        goal_angle = torch.atan2(goal_direction[:, 1], goal_direction[:, 0])

        # compute the angle difference (yaw error)
        angle_diff = goal_angle - yaw
        # normalize to range [-π, π] using modulo operation
        angle_diff = (angle_diff + torch.pi) % (2 * torch.pi) - torch.pi
        angle_diff = angle_diff.unsqueeze(-1)

        # concatenate into a single observation vector
        obs = torch.cat((pos,
                         angle_diff,
                         base_v,
                         base_omega,
                         dist_to_goal_x,
                         dist_to_goal_y,
                         dist_to_obs,
                        ), dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # compute rewards
        # combination of reaching the goal, moving towards the goal, and penalizing termination

        dist_to_obs = self.maze.check_collision(self.jetbot.data.root_pos_w[:, :2])
        #print("Distance to obstacles: ", dist_to_obs)

        # compute rewards
        # combination of reaching the goal, moving towards the goal, and penalizing termination
        primary_reward, dist = compute_rewards(
            self.rew_scale_terminated,
            self.rew_scale_goal,
            self.rew_scale_distance,
            self.rew_scale_time,
            self.jetbot.data.root_pos_w[:, :2], # shape [batch_size, 2] # root position
            self.cfg.goal_pos[:2], # shape [1, 2] # goal position
            self.cfg.max_distance_from_goal, # max distance from goal, for termination
            self.prec_dist,
            self.episode_length_buf,
            self.gamma
        )

        self.prec_dist = dist

        constraint_cost = compute_cost(
            self.cost_obstacle,
            self.min_dist_to_obstacle + self.cfg.wall_width / 2.0,
            dist_to_obs,
        )
        self._computed_constraint_cost = constraint_cost

        self.episode_rewards += primary_reward
        self.episode_costs += constraint_cost
        self.last_episode_costs = constraint_cost

        return primary_reward - self.dual_multiplier * constraint_cost
        #return primary_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # check if the episode has run out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # check if the robot has gone out of bounds
        out_of_bounds = torch.any(torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal, dim=1)

        # check if the robot has reached the goal
        reached_goal = torch.norm(self.jetbot.data.root_pos_w[:, :2] 
                                  - torch.tensor(self.cfg.goal_pos[:2], device=self.jetbot.data.root_pos_w.device), dim=1) <= 0.1
            
        # episode is done if any of the conditions are met
        return out_of_bounds | reached_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.jetbot._ALL_INDICES
        super()._reset_idx(env_ids)

        # sample a new starting position within initial_pose_radius around the goal
        num_ids = len(env_ids)

        # random angle and radius around goal
        theta = torch.rand(num_ids, device=self.device) * 2 * math.pi
        r = torch.sqrt(
            torch.rand(num_ids, device=self.device) * 
            (self.cfg.initial_pose_radius**2 - self.cfg.initial_pose_min_distance**2) + 
            self.cfg.initial_pose_min_distance**2
        )

        # transform polar coordinates to cartesian coordinates
        new_x = self.cfg.goal_pos[0] + r * torch.cos(theta)
        new_y = self.cfg.goal_pos[1] + r * torch.sin(theta)
        new_z = torch.ones_like(new_x) * self.cfg.spawn_height
        new_positions = torch.stack([new_x, new_y, new_z], dim=1)

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

        self.prec_dist = torch.zeros(self.jetbot.data.root_pos_w.shape[0], device=self.device)

        # set position and orientation
        default_root_state = self.jetbot.data.default_root_state[env_ids]
        default_root_state[:, :3] = new_positions
        default_root_state[:, 3:7] = new_orientation

        self.jetbot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

        # reset safety indicator for the reset environments
        self.safety_indicator[env_ids] = 1

    def _compute_yaw(self, quaternions: torch.Tensor) -> torch.Tensor:
        # convert quaternion [w, x, y, z] to yaw angle
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        # calculate yaw using the standard formula
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    
    def _set_dual_multiplier(self, value: float) -> None:
        self.dual_multiplier = value

    def _get_constraint_cost(self):
        return self.last_episode_costs
    
    def _get_safety_probability(self):
        return torch.mean(self.safety_indicator)
    
    def _reset_reward_and_cost(self):
        self.episode_rewards = torch.zeros(self.n_envs, device=self.device)
        self.episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.last_episode_costs = torch.zeros(self.n_envs, device=self.device)
        self.safety_indicator = torch.ones(self.n_envs, device=self.device)
    
    def _get_reward_and_cost(self):
        return self.episode_rewards, self.episode_costs
    
    def _set_num_envs(self, num_envs):
        self.n_envs = num_envs

    def _domain_randomize(self):
        self.maze.move_walls(0.1)


# decorator to compile the function to TorchScript
# https://pytorch.org/docs/stable/jit.html
# useful for performance optimization
@torch.jit.script
def compute_rewards(
    rew_scale_terminated: float,
    rew_scale_goal: float,
    rew_scale_distance: float,
    rew_scale_time: float | None,
    robot_pos: torch.Tensor,
    goal_pos: list[float],
    max_distance_from_goal: float,
    prec_dist: torch.Tensor,
    episode_length_buf: torch.Tensor,
    gamma: float
):
    # compute the distance to the goal
    dist_to_goal = torch.norm(robot_pos - torch.tensor(goal_pos, device=robot_pos.device), dim=1) 

    # compute the rewards
    terminated = (dist_to_goal >= max_distance_from_goal).float()
    rew_termination = rew_scale_terminated * terminated

    rew_goal = rew_scale_goal * (dist_to_goal <= 0.1).float() * (gamma ** episode_length_buf)/(1.0 - gamma)

    rew_distance = torch.where(
        (prec_dist == 0.0) | torch.isclose(prec_dist - dist_to_goal, torch.tensor(0.0, device=prec_dist.device), atol=1e-6), 
        torch.zeros_like(dist_to_goal), 
        prec_dist - dist_to_goal
    )

    rew_distance = rew_scale_distance * rew_distance

    if rew_scale_time is not None:
        rew_time = dist_to_goal * torch.exp(rew_scale_time * episode_length_buf)
        total_reward = rew_termination + rew_goal + rew_distance
        return total_reward, dist_to_goal
    
    total_reward = rew_termination + rew_goal + rew_distance
    return total_reward, dist_to_goal

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

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_path, "obs_cfg.yaml")

        with open(self.config_path, "r") as f:
            maze_config = yaml.safe_load(f)
        self.walls = maze_config["maze"]["walls"]

    def spawn_maze(self, width=0.1, height=0.5, walls=None):
        if walls is None:
            walls = self.walls

        self.width = width
        self.height = height

        self.walls_start_tensor = torch.tensor([wall["start"] for wall in walls],
                                                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                 dtype=torch.float)
        self.walls_end_tensor = torch.tensor([wall["end"] for wall in walls],
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               dtype=torch.float)
        
        for i, wall in enumerate(walls):
            prim_path = f"/World/Maze/wall_{i}"
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
        
        return cfg_wall_cube

    def sample_safe_position(self, r_min=1.0, r_max=5.0, robot_radius=0.6, max_attempts=100, safety_margin=0.05):

        device = self.walls_start_tensor.device

        # generate max_attempts candidate positions uniformly within the specified bounds.
        # candidates: shape (max_attempts, 2)
        candidates = torch.empty((max_attempts, 2), device=device)
        candidates[:, 0] = torch.rand(max_attempts, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        candidates[:, 1] = torch.rand(max_attempts, device=device) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
        
        # precomputed wall tensors: shape (M, 2) each
        # expand dimensions to allow broadcasting:
        # candidates: (A, 2) -> (A, 1, 2)
        # walls_start_tensor: (M, 2) -> (1, M, 2)
        cand_exp = candidates.unsqueeze(1)         # (A, 1, 2)
        walls_start = self.walls_start_tensor.unsqueeze(0)  # (1, M, 2)
        walls_end   = self.walls_end_tensor.unsqueeze(0)    # (1, M, 2)

        # compute wall directions and their norms (avoid division by zero):
        wall_dirs = walls_end - walls_start          # (1, M, 2)
        norm_wall_dirs = torch.norm(wall_dirs, dim=2)  # (1, M)
        norm_wall_dirs = torch.where(norm_wall_dirs == 0, 
                                    torch.tensor(1e-6, device=device), 
                                    norm_wall_dirs)

        # compute the vector from each wall's start to each candidate:
        vec = cand_exp - walls_start                # (A, M, 2)
        # in 2D, the magnitude of the cross product for vectors (a, b) is:
        # |a_x * b_y - a_y * b_x|
        cross = torch.abs(vec[:, :, 0] * wall_dirs[:, :, 1] - vec[:, :, 1] * wall_dirs[:, :, 0])  # (A, M)
        
        # distance from candidate to each wall:
        distances = cross / norm_wall_dirs           # (A, M)

        # a candidate is safe if its distance to every wall is at least (robot_radius + safety_margin)
        safe_threshold = robot_radius + safety_margin
        safe_mask = (distances >= safe_threshold).all(dim=1)  # (A,) True for candidates that are safe

        # select the first safe candidate if any exist, otherwise return the last candidate.
        safe_candidates = candidates[safe_mask]
        if safe_candidates.shape[0] > 0:
            return safe_candidates[0].tolist()
        else:
            return candidates[-1].tolist()
        
    def distance_from_line(self, point, start, end):
        # compute perpendicular distance from point to the line defined by start and end.
        import numpy as np
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

        # --- Case 2: When projection falls outside the segment (t < 0 or t > 1) ---
        mask_outside = ~mask_inside

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

        return smallest3
    
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


def direct_diff_drive(v, omega, L, r):
    # convert base velocity commands (v, ω) into left and right wheel velocities

    # compute left & right wheel velocities using differential drive formula
    v_left = v - (omega * L / 2)
    w_left = v_left / r
    v_right = v + (omega * L / 2)
    w_right = v_right / r

    return w_left, w_right

def inverse_diff_drive(w_left, w_right, L, r):
    # convert left and right wheel velocities into base velocity commands (v, ω)

    # compute linear and angular velocities using differential drive formula
    v = (r / 2) * (w_left + w_right)
    omega = (r / L) * (w_right - w_left)

    return v, omega

