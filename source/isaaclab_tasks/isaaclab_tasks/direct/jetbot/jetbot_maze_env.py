from __future__ import annotations

import math
import torch
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
import random

@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 20.0
    action_scale = 1.0  # TODO
    # 2 wheel control
    action_space = 2
    observation_space = 7 # x, y, yaw_diff, left_wheel_vel, right_wheel_vel, distance_to_goal_x, distance_to_goal_y
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
    initial_pose_radius = 5.0  # the radius around goal in which the jetbot initial position is sampled

    # reward scales
    rew_scale_goal = 20.0 # reward for reaching the goal
    rew_scale_distance = 5.0 # reward for moving towards the goal
    # penalty for moving away from the goal # using r_d = 1 - exp(-rew_scalse_distance*dist)
    # rew_scale_velocity = 0.1 # reward for moving towards the goal with high velocity
       # might be emerging behavior from high reward for reaching the goal
    # rew_scale_alive = 0.1 # reward for staying alive
    rew_scale_terminated = -20.0 # penalty for terminating the episode (went too far)
    # https://arxiv.org/pdf/2002.04109 (r_reached, r_crashed, 1 - exp(decay_rate*dist))

    # goal
    goals = [[-2.0,1.0], [-2.0,0.0], [2.0,0.0], [0.5,3.5], [-0.5,-3.5]]

    # spawn height
    spawn_height = 0.0437

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

    def _setup_scene(self):
        self.jetbot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add a cube as a goal
        goal_size = (0.25, 0.25, 0.25)
        for i, g in enumerate(self.cfg.goals):
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

        self.maze.spawn_maze()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

        """
        # convert base velocity commands (v, ω) into left and right wheel velocities

        # scale and extract linear & angular velocity from actions
        self.actions = self.action_scale * actions.clone()
        v = self.actions[:, 0]  # linear velocity (m/s)
        omega = self.actions[:, 1]  # angular velocity (rad/s)

        # get JetBot's wheelbase
        L = self.wheelbase

        # compute left & right wheel velocities using differential drive formula
        v_left = v - (omega * L / 2)
        v_right = v + (omega * L / 2)

        # apply velocities to the robot's wheels
        self.jetbot.data.joint_vel[:, self.left_wheel_idx] = v_left
        self.jetbot.data.joint_vel[:, self.right_wheel_idx] = v_right
        """

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

        # orientation: compute yaw from the root orientation
        orient = self.jetbot.data.root_quat_w
        yaw = self._compute_yaw(orient).squeeze(-1)  # helper function defined below.
        #print("Robot yaw: ", yaw)

        # wheel velocities
        left_wheel_vel = self.joint_vel[:, self._left_wheel_idx].squeeze(-1).unsqueeze(-1)
        right_wheel_vel = self.joint_vel[:, self._right_wheel_idx].squeeze(-1).unsqueeze(-1)

        # distance to goal (use only x, y coordinates)
        # picking the closest goal to the robot
        all_goals = torch.tensor(self.cfg.goals, device=pos.device)
        diff = pos.unsqueeze(1) - all_goals.unsqueeze(0)
        dists = torch.norm(diff, dim=-1)
        closest_idx = torch.argmin(dists, dim=1)
        goal_xy = all_goals[closest_idx]

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
                         left_wheel_vel,
                         right_wheel_vel,
                         dist_to_goal_x,
                         dist_to_goal_y
                        ), dim=-1)
        #print("Observations shape: ", obs.shape)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # compute rewards
        # combination of reaching the goal, moving towards the goal, and penalizing termination
        total_reward, dist = compute_rewards(
            #self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_goal,
            self.cfg.rew_scale_distance,
            self.jetbot.data.root_pos_w[:, :2], # shape [batch_size, 2] # root position
            self.cfg.goals, # shape [num_goals, 2] # goals positions
            self.cfg.max_distance_from_goal, # max distance from goal, for termination
            self.prec_dist
        )
        self.prec_dist = dist
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # check if the episode has run out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # check if the robot has gone out of bounds
        out_of_bounds = torch.any(torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal, dim=1)

        # check if the robot has reached the goal
        robot_pos = self.jetbot.data.root_pos_w[:, :2]
        # convert the list of goal positions into a tensor (shape: [num_goals, 2])
        all_goals = torch.tensor(self.cfg.goals, device=self.jetbot.data.root_pos_w.device)  
        # pairwise differences between each robot and each goal
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        # Compute Euclidean distances along the last dimension: [num_envs, num_goals]
        dists = torch.norm(diff, dim=2)
        # Check if any goal is within 0.1 distance for each robot: [num_envs] boolean tensor
        reached_goal = torch.any(dists <= 0.1, dim=1)

        # episode is done if any of the conditions are met
        return out_of_bounds | reached_goal, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.jetbot._ALL_INDICES
        super()._reset_idx(env_ids)

        # sample a new starting position within initial_pose_radius around the goal
        num_ids = len(env_ids)

        safe_positions = []
        for _ in env_ids:
            pos_xy = self.maze.sample_safe_position()
            safe_positions.append(pos_xy)
        safe_positions = torch.tensor(safe_positions, device=self.device, dtype=torch.float)
        
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

        robot_pos = self.jetbot.data.root_pos_w[:, :2]
        all_goals = torch.tensor(self.cfg.goals, device=robot_pos.device)
        diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
        dists = torch.norm(diff, dim=2)
        self.prec_dist = torch.min(dists, dim=1).values

        # update the root state
        default_root_state[:, 3:7] = new_orientation

        self.jetbot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)

    def _compute_yaw(self, quaternions: torch.Tensor) -> torch.Tensor:
        # convert quaternion [w, x, y, z] to yaw angle
        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]
        # calculate yaw using the standard formula
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw

# decorator to compile the function to TorchScript
# https://pytorch.org/docs/stable/jit.html
# useful for performance optimization
@torch.jit.script
def compute_rewards(
    #rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_goal: float,
    rew_scale_distance: float,
    robot_pos: torch.Tensor,
    goal_pos: list[list[float]],
    max_distance_from_goal: float,
    prec_dist: torch.Tensor
):
    # compute the distance to the goal
    all_goals = torch.tensor(goal_pos, device=robot_pos.device)
    diff = robot_pos.unsqueeze(1) - all_goals.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    dist_to_goal, _ = torch.min(dists, dim=1)

    # compute the rewards
    terminated = (dist_to_goal >= max_distance_from_goal).float()
    rew_termination = rew_scale_terminated * terminated
    #rew_alive = rew_scale_alive * (1.0 - terminated)
    rew_goal = rew_scale_goal * (dist_to_goal <= 0.11).float()
    #rew_distance = 1.15 - math.exp(rew_scale_distance * dist_to_goal)
    rew_distance = torch.where(prec_dist == 0.0, 
                           torch.zeros_like(dist_to_goal), 
                           prec_dist - dist_to_goal)
    rew_distance = rew_scale_distance * rew_distance
    total_reward = rew_termination + rew_goal + rew_distance
    #print("Total reward: ", total_reward)
    return total_reward, dist_to_goal

class Maze:
    def __init__(self):
        self.walls = []

    def spawn_maze(self):
        with open("/home/federico/isaaclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/jetbot/maze_confi.yaml", "r") as f:
            maze_config = yaml.safe_load(f)
        self.walls = maze_config["maze"]["walls"]

        for i, wall in enumerate(self.walls):
            start = wall["start"]
            end = wall["end"]
            prim_path = f"/World/Maze/wall_{i}"
            self.spawn_wall_cube(prim_path, start, end)

    def spawn_wall_cube(self, prim_path: str, start: list[float], end: list[float], height: float = 0.5, width: float = 0.1):
        # compute length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        
        # compute midpoint
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        mid_z = height / 2.0  # place it on the ground
        translation = (mid_x, mid_y, mid_z)

        size = (length, width, height)

        # compute orientation angle
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        theta_rad = math.radians(angle_deg)
        w = math.cos(theta_rad / 2)
        x = 0.0
        y = 0.0
        z = math.sin(theta_rad / 2)
        orientation = (w, x, y, z)

        # Create the cuboid configuration using sim_utils
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

    def sample_safe_position(self, x_bounds=(-1.5,1.5), y_bounds=(-3.0,3.0), robot_radius=0.15, max_attempts=100, safety_margin=0.1):
        for _ in range(max_attempts):
            candidate = [random.uniform(*x_bounds), random.uniform(*y_bounds)]
            safe = True
            for wall in self.walls:
                # for each wall, compute if candidate is within a safety margin.
                # you might use the distance from the candidate point to the line defined by wall start and end.
                if self.distance_from_line(candidate, wall["start"], wall["end"]) < robot_radius + safety_margin:
                    safe = False
                    break
            if safe:
                return candidate
        return candidate
    
    def distance_from_line(self, point, start, end):
        # compute perpendicular distance from point to the line defined by start and end.
        import numpy as np
        p = np.array(point)
        s = np.array(start)
        e = np.array(end)
        if np.allclose(s, e):
            return np.linalg.norm(p - s)
        return np.abs(np.cross(e-s, s-p)) / np.linalg.norm(e-s)
