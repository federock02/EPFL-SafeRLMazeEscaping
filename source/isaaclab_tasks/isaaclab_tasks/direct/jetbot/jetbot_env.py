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
from isaaclab.utils.math import sample_uniform

@configclass
class JetBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 20.0
    action_scale = 1.0  # TODO
    # 2 wheel control
    action_space = 2
    observation_space = 6 # x, y, yaw, left_wheel_vel, right_wheel_vel, distance_to_goal
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
    max_distance_from_goal = 10.0 # jetbot moved too far from goal
    initial_pose_radius = 6.0  # the radius around goal in which the jetbot initial position is sampled

    # reward scales
    rew_scale_goal = 4.0 # reward for reaching the goal
    rew_scale_distance = -0.1 # penalty for moving away from the goal
    # rew_scale_velocity = 0.1 # reward for moving towards the goal with high velocity
       # might be emerging behavior from high reward for reaching the goal
    # rew_scale_alive = 0.1 # reward for staying alive
    rew_scale_terminated = -2.0 # penalty for terminating the episode (went too far)
    # https://arxiv.org/pdf/2002.04109 (r_reached, r_crashed, 1 - exp(decay_rate*dist))

    # goal
    goal_pos = [0.0, 0.0]  # x, y

    # spawn height
    spawn_height = 0.0437

class JetBotEnv(DirectRLEnv):
    cfg: JetBotEnvCfg

    def __init__(self, cfg: JetBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # use the joint names from the configuration to find indices in the articulation
        self._left_wheel_idx, _ = self.jetbot.find_joints(self.cfg.left_wheel_joint_name)
        self._right_wheel_idx, _ = self.jetbot.find_joints(self.cfg.right_wheel_joint_name)
        self.action_scale = self.cfg.action_scale

        # cache joint positions and velocities data for efficiency
        self.joint_pos = self.jetbot.data.joint_pos
        self.joint_vel = self.jetbot.data.joint_vel

    def _setup_scene(self):
        self.jetbot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add a cube as a goal
        goal_pos = self.cfg.goal_pos + [0.25]
        goal_size = (0.5, 0.5, 0.5)
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
        print("Left action: ", left_action)
        print("Right action: ", right_action)
        # set the wheel velocities
        self.jetbot.set_joint_velocity_target(left_action, joint_ids=self._left_wheel_idx)
        self.jetbot.set_joint_velocity_target(right_action, joint_ids=self._right_wheel_idx)

    def _get_observations(self) -> dict:
        # gethering observations
        # position: x, y from the root position
        pos = self.jetbot.data.root_pos_w[:, :2]
        print("Robot pos: ", pos)
        # orientation: compute yaw from the root orientation
        orient = self.jetbot.data.root_quat_w
        yaw = self._compute_yaw(orient)  # helper function defined below.
        print("Robot yaw: ", yaw)
        # wheel velocities
        left_wheel_vel = self.joint_vel[:, self._left_wheel_idx].squeeze(-1)
        right_wheel_vel = self.joint_vel[:, self._right_wheel_idx].squeeze(-1)
        # distance to goal (use only x, y coordinates)
        goal_xy = self.cfg.goal_pos[:2]
        # in order to compute the distance to the goal, we need to convert the goal position to a tensor
        # the goal position tensor needs to be on the same device as the observations
        dist_to_goal = torch.norm(pos - torch.tensor(goal_xy, device=pos.device), dim=1, keepdim=True)

        # concatenate into a single observation vector
        obs = torch.cat((pos, yaw.unsqueeze(1), left_wheel_vel.unsqueeze(1), right_wheel_vel.unsqueeze(1), dist_to_goal), dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # compute rewards
        # combination of reaching the goal, moving towards the goal, and penalizing termination
        total_reward = compute_rewards(
            #self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_goal,
            self.cfg.rew_scale_distance,
            self.jetbot.data.root_pos_w[:, :2], # shape [batch_size, 2] # root position
            self.cfg.goal_pos[:2], # shape [1, 2] # goal position
            self.cfg.max_distance_from_goal # max distance from goal, for termination
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # check if the episode must be terminated
        # check if the episode has run out of time
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # check if the robot has gone out of bounds
        out_of_bounds = torch.any(torch.abs(self.jetbot.data.root_pos_w[:, :2]) > self.cfg.max_distance_from_goal, dim=1)
        # check if the robot has reached the goal
        reached_goal = torch.norm(self.jetbot.data.root_pos_w[:, :2] - torch.tensor(self.cfg.goal_pos[:2], device=self.jetbot.data.root_pos_w.device), dim=1) <= 0.05
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
        r = torch.sqrt(torch.rand(num_ids, device=self.device)) * self.cfg.initial_pose_radius

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
        new_orientation = torch.stack([qx, qy, qz, qw], dim=1)

        # set position and orientation
        default_root_state = self.jetbot.data.default_root_state[env_ids]
        default_root_state[:, :3] = new_positions
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
    goal_pos: list[float],
    max_distance_from_goal: float
):
    # compute the distance to the goal
    dist_to_goal = torch.norm(robot_pos - torch.tensor(goal_pos, device=robot_pos.device), dim=1)

    # compute the rewards
    terminated = (dist_to_goal > max_distance_from_goal).float()
    rew_termination = rew_scale_terminated * terminated
    #rew_alive = rew_scale_alive * (1.0 - terminated)
    rew_goal = rew_scale_goal * (dist_to_goal <= 0.05).float()
    rew_distance = rew_scale_distance * dist_to_goal
    total_reward = rew_termination + rew_goal + rew_distance
    return total_reward
