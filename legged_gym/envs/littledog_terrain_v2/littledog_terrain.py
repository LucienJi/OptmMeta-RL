from time import time
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .littledog_terrain_config import LittledogTerrainCfg
"""
description:
1. action: (6 * 3,) i: i+3 , mu,omega,phi
"""
TRIPLE_PHASE = [0, 0.5, 0, 0.5, 0, 0.5]
NUM_LEG = 6

class LittledogTerrain(LeggedRobot):
    cfg : LittledogTerrainCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super(LeggedRobot,self).__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.normal = torch.zeros(self.num_envs, 6, dtype = torch.double, device=self.device, requires_grad=False) 

        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #! 这里我们专门 reset 额外设定的 buffer
        self._reset_contact(env_ids)
        self.history_desired_foot_pos[env_ids] = 0.
        self.history_dof_vel[env_ids] = 0.
        self.history_joint_pos_error[env_ids] = 0.
        self.ftg_phase[env_ids] = 0.
        self.ftg_freq[env_ids] = 0.
        self.contacts[env_ids] =1 ## 这里我假设初始在 stand 姿势

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        #! 我们初始化一些需要的 buffer
        #! 我们假设 history = 2
        self.history_desired_foot_pos = torch.zeros(self.num_envs, 3*NUM_LEG * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.history_dof_vel = torch.zeros(self.num_envs, self.num_dofs * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.history_joint_pos_error = torch.zeros(self.num_envs, self.num_dofs * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.ftg_phase = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.float, device=self.device, requires_grad=False)
        self.ftg_freq = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.bool, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        #! 这是在 step 中执行的， 已经 refresh 过 dof_state_tensor 了
        #! 现在（我猜的）通过 dof 的变化，计算 base 和 force 的变化
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        #! root_states 是 view of root， 接下来根据 quat 旋转 vel 和 omega 
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        #! 注意, 这里的 reward 计算，是使用 step 之后的状态计算的
        #! 注意，我们在这里 reset 和 计算 obs
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        
        return super()._compute_torques(actions)

    def compute_observations(self):
        """ Computes observations
        """
        #! 我们将 observation = basic observation + current observation 
        """
        basic observation:
            cmd; joint pos, joint vel ,gravity, last action, FTG internal state 
        current observation:
            contact boolean, history_joint_vel, history_joint_pos_error, desired_foot_pos
        """
        
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    def _reset_contact(self,env_ids):
        #! 假设触地力 大于 1.0 为触地
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contacts[env_ids] = contact[env_ids]

    def inverse_kinematics(self, x, y, z, bendInfo:bool):
        len_thigh = 0.25
        len_calf = 0.25
        
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1) # num_envs * num_dof(18)

        # t1 > 0 , t2 < 0 for bend in. bend_info=True
        # t1 < 0 , t2 > 0 for bend out. bend_info=False
        theta0 = torch.atan(-y/z)

        z_tip_sag = torch.sqrt(z*z+y*y)
        cos_shank = (z_tip_sag**2 + x**2 - len_thigh**2 - len_calf**2)/(2*len_thigh*len_calf)
        cos_shank = torch.clamp(cos_shank, -1.0, 1.0)

        if bendInfo == True:
            theta2 = - torch.acos(cos_shank)
        else:
            theta2 = torch.acos(cos_shank)

        cos_beta = (z_tip_sag**2 + x**2 + len_thigh**2 - len_calf**2)/(2*len_thigh*torch.sqrt(z_tip_sag**2 + x**2))
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)

        alpha = torch.atan(x/z_tip_sag)

        if bendInfo == False:
            theta1 = -alpha - beta
        else:
            theta1 = -alpha + beta

        pos_action[:,[0,3,6, 9,12,15]] = theta0
        pos_action[:,[1,4,7,10,13,16]] = theta1
        pos_action[:,[2,5,8,11,14,17]] = theta2

        return pos_action



    def __init__(self, omega=5*torch.pi, mu=1, phase=TRIPLE_PHASE, alpha=100, beta=100, k=0.1):
        self.omega = omega
        self.mu = mu
        self.phase = phase

        self.alpha = alpha
        self.beta = beta
        self.k = k

    def hopf(self, x, y, steps=0.001, epsilon=1e-6):
        r2 = x**2 + y**2
        dx = self.alpha * (self.mu**2 -r2) * x - self.omega * y
        dy = self.beta  * (self.mu**2 -r2) * y + self.omega * x

        for i in range(NUM_LEG):
            for j in range(NUM_LEG):
                theta = (self.phase[i] - self.phase[j]) * 2 * math.pi
                coupling_term_x = -math.sin(theta) * (x[j]+y[j]) / torch.sqrt(x[j]**2+y[j]**2+epsilon)
                coupling_term_y =  math.cos(theta) * (x[j]+y[j]) / torch.sqrt(x[j]**2+y[j]**2+epsilon)
                dx[i] += self.k * coupling_term_x
                dy[i] += self.k * coupling_term_y
        
        return x+dx*steps, y+dy*steps

    def mapping(self, x, y, action, k2=0.1):
        h = 0.36
        # phi = torch.zeros_like(action[:,NUM_LEG].reshape((-1,1)))
        # d_step = 0.2*torch.ones_like(action[:,-1].reshape((-1,1))) 

        phi = action[:,NUM_LEG].reshape((-1,1)) # torch.zeros_like(action[:,NUM_LEG].reshape((-1,1)))
        d_step = action[:,-1].reshape((-1,1)) # 0.2*torch.ones_like(action[:,-1].reshape((-1,1))) 

        foot_x = - d_step * x * torch.cos(phi)
        foot_y = - d_step * x * torch.sin(phi)
        # foot_y[:,0:3] += -0.08025
        # foot_y[:,3:6] += 0.08025

        foot_z = torch.zeros_like(foot_x)
        for i in range(NUM_LEG):
            # k1 = 0.1
            k1 = action[:,i] # 0.1
            foot_z[:,i] = torch.where(y[:,i] >= 0, -h + k1 * y[:,i], -h + k2 * y[:,i]) 

        return foot_x, foot_y, foot_z