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
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
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

        #! 计算正向动力学
        self._abadLength = 0.0802
        self._thighLength = 0.249 # thigh
        self._shankLength = 0.24532 # shank
        self._bodyHeight = 0.41
        self._abadLocation = torch.tensor([
                [0.33, -0.05, 0.0],   # rf
                [0.0, -0.19025, 0.0], # rm
                [-0.33, -0.05, 0.0],  # rb
                [-0.33, 0.05, 0.0],   # lb
                [0.0, 0.19025, 0.0],  # lm
                [0.33, 0.05, 0.0]],  # lf, 
            dtype=torch.double, device=self.device, requires_grad=False)


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_leg_per_env)
        """
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_desired_footpositions[:] = self._dof2pos(self.last_dof_pos)
        #! 这里计算

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques= self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print('feet_contact!!!!!',torch.norm(self.contact_forces[0, self.feet_indices-1, :], dim=-1)>0.1)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    
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
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_desired_footpositions[env_ids] = self._dof2pos(self.last_dof_pos[env_ids])     
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        #! 这里我们专门 reset 额外设定的 buffer
        self.history_desired_foot_pos[env_ids,:3*NUM_LEG] =self.last_desired_footpositions[env_ids]
        self.history_desired_foot_pos[env_ids,3*NUM_LEG:] =self.last_desired_footpositions[env_ids]
        self.history_dof_vel[env_ids] = 0.
        self.history_joint_pos_error[env_ids] = 0.
        self.ftg_phase[env_ids] = 0.
        self.ftg_freq[env_ids] = 0.
        self.contacts[env_ids] =1 ## 这里我假设初始在 stand 姿势
        self._reset_contact(env_ids)

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
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_desired_footpositions = torch.zeros(self.num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self.device, requires_grad=False)
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
        #! init foot pos
        self.foot_height_points  = self._init_foot_height_points()
        self.measured_foot_positions = torch.zeros(self.num_envs, NUM_LEG * self.num_foot_height_points,3, dtype=torch.float, device=self.device, requires_grad=False)
        #! 我们初始化一些需要的 buffer
        #! 我们假设 history = 2

        self.history_desired_foot_pos = torch.zeros(self.num_envs, 3*NUM_LEG * 2, dtype=torch.float, device=self.device, requires_grad=False) #! 这也是放在 body frmae 中的, 也可以考虑 放在 hip frame 中
        self.history_dof_vel = torch.zeros(self.num_envs, self.num_dofs * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.history_joint_pos_error = torch.zeros(self.num_envs, self.num_dofs * 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.ftg_phase = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.float, device=self.device, requires_grad=False)
        self.ftg_freq = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts = torch.zeros(self.num_envs, NUM_LEG, dtype=torch.bool, device=self.device, requires_grad=False)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device, requires_grad=False)
        
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
        self.history_dof_vel = torch.cat((self.history_dof_vel[:, self.num_dofs:], self.last_dof_vel), dim=-1)
        self.history_joint_pos_error = torch.cat((self.history_joint_pos_error[:, self.num_dofs:], 
                                                  self.last_dof_pos - self.default_dof_pos), dim=-1)
        self.history_desired_foot_pos = torch.cat((self.history_desired_foot_pos[:, 3*NUM_LEG:], self.last_desired_footpositions), dim=-1)
        self._reset_contact()#Set Contact Boolean

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        
        self.reset_idx(env_ids) #! 上面是正常流程, 但是 reset 也需要重新走一边



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
    def _compute_basic_observation(self):
        """
        1. dim = NUM_LEG * 3 + NUM_LEG * 3 + 3 + NUM_LEG + NUM_LEG + NUM_LEG + NUM_Action + NUM_LEG
           dim = 6 * 3 + 6 * 3 + 3 + 6 + 6 + 6 + 18 + 6 = 81
        """
        self.basic_obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.projected_gravity,
            torch.cos(self.ftg_phase),
            torch.sin(self.ftg_phase),
            self.ftg_freq,
            self.actions,
            self.contacts,
        ), dim = -1)
                                        
    def _compute_other_observation(self):
        """
        1. dim = 3 + 3 + 2 * NUM_LEG * 3 + 2 * NUM_LEG * 3 + 2 * NUM_LEG * 3 
        2. dim = 3 + 3 + 2 * 6 * 3 + 2 * 6 * 3 + 2 * 6 * 3 = 114
        """
        self.other_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.history_dof_vel,
            self.history_joint_pos_error,
            self.history_desired_foot_pos,
        ),dim = -1)
    def _compute_privileged_observation(self):
        """
        privileged observation:
            1. contact force : legs * 3
            2. height sampled
        
        为了减少计算量, 我们只在 compute observation 的时候计算 privileged observation, reset 的时候并不计算,

        dim = NUM_LEG * 3 + NUM_LEG * N_Height_Samples = 6 * 3 + 6 * 4 * 4 = 114
        """
        measured_foot_height = self.get_foot_position()
        contact_force = self.contact_forces[:, self.feet_indices, :3].reshape(self.num_envs, NUM_LEG * 3)
        # print("Shape Check: ", measured_foot_height.shape, contact_force.shape)
        self.privileged_obs_buf = torch.cat((
            measured_foot_height,
            contact_force
        ), dim = -1)

        
    def compute_observations(self):
        """ Computes observations
        """
        #! 我们将 observation = basic observation + current observation 
        #! 首先, 我们计算 basic observation, basic observation 是要叠加做 history的
        """
        basic observation:
            cmd; joint pos, joint vel ,gravity, last action, FTG internal state 
        current observation:
            contact boolean, history_joint_vel, history_joint_pos_error, desired_foot_pos
        """
        self._compute_basic_observation()
        self._compute_other_observation()
        self._compute_privileged_observation()
        self.obs_buf = torch.cat((  self.basic_obs_buf,self.other_obs_buf
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    def _reset_contact(self,env_ids = None):
        #! 假设触地力 大于 1.0 为触地
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        if env_ids is None:
            self.contacts[:] = contact[:]
        else:
            self.contacts[env_ids] = contact[env_ids]

    def inverse_kinematics(self, x, y, z, bendInfo:bool = True):
        ## x,y,z 是相对于 hip frame 
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1) # num_envs * num_dof(18)
        
        yz_square = y**2 + z**2 - self._abadLength **2 
        index = yz_square < 0
        
        y[index] = y[index] / torch.sqrt(yz_square[index] + self._abadLength **2 )  * self._abadLength + 1e-5
        z[index] = z[index] / torch.sqrt(yz_square[index] + self._abadLength **2 )  * self._abadLength + 1e-5
        yz_square[index] = 0.0 

        length_square = yz_square + x**2 
        yz = torch.sqrt(yz_square)
        length = torch.sqrt(length_square)

        tmp1 =(self._thighLength ** 2 - self._shankLength ** 2 + length_square)/2.0/self._thighLength/length
        tmp1 = torch.clip(tmp1, -1.0, 1.0)

        tmp2 = (self._thighLength ** 2 + self._shankLength ** 2 - length_square)/2.0/self._thighLength/self._shankLength
        tmp2 = torch.clip(tmp2, -1.0, 1.0)
        
        if bendInfo:
            theta1 = -torch.atan2(x,yz) - torch.acos(tmp1)
            theta2 =  math.pi - torch.acos(tmp2) 
        else:
            theta1 = - torch.atan2(x,yz)+ torch.acos(tmp1)
            theta2 = -math.pi + torch.acos(tmp2)

        _abadLength = torch.ones_like(y[:,:3]) * self._abadLength
        theta0_r = -torch.atan2(z[:,:3],-y[:,:3]) - torch.atan2(yz[:,:3],_abadLength)
        theta0_l = torch.atan2(z[:,3:],y[:,3:]) + torch.atan2(yz[:,3:],_abadLength)
        theta0 = torch.cat((theta0_r, theta0_l), dim=-1)
        

        pos_action[:,[0,3,6, 9,12,15]] = theta0
        pos_action[:,[1,4,7,10,13,16]] = theta1
        pos_action[:,[2,5,8,11,14,17]] = theta2

        return pos_action

    def forward_kinematics(self):
        """
        计算再 hip frame 下的 x,y,z
        """
        pos = self.dof_pos.squeeze(-1)
        #! 从 pos 中提取出 theta0, theta1, theta2
        theta0 = pos[:,[0,3,6, 9,12,15]] # num_envs, 6 
        theta1 = pos[:,[1,4,7,10,13,16]]
        theta2 = pos[:,[2,5,8,11,14,17]]

        #! 计算 x:  计算 thigh 和 shank 的转动
        x = - torch.sin(theta1)*self._thighLength - torch.sin(theta1+theta2)*self._shankLength

        #! j计算 y:
        y_r = -torch.cos(theta0[:,:3])*self._abadLength + torch.sin(theta0[:,:3]) * (self._thighLength*torch.cos(theta1[:,:3]) + self._shankLength*torch.cos(theta1[:,:3]+theta2[:,:3]))
        y_l = torch.cos(theta0[:,3:])*self._abadLength + torch.sin(theta0[:,3:]) * (self._thighLength*torch.cos(theta1[:,3:]) + self._shankLength*torch.cos(theta1[:,3:]+theta2[:,3:]))
        y = torch.cat((y_r, y_l), dim=-1)
        #! 计算 z: 
        z_r = -torch.sin(theta0[:,:3])*self._abadLength - torch.cos(theta0[:,:3]) * (self._thighLength*torch.cos(theta1[:,:3]) + self._shankLength*torch.cos(theta1[:,:3]+theta2[:,:3]))
        z_l = torch.sin(theta0[:,3:])*self._abadLength - torch.cos(theta0[:,3:]) * (self._thighLength*torch.cos(theta1[:,3:]) + self._shankLength*torch.cos(theta1[:,3:]+theta2[:,3:]))
        z = torch.cat((z_r, z_l), dim=-1)

        return x, y, z

    def _dof2pos(self, dof):
        """
        1. 计算, num_envs, n_dof -> num_envs, num_leg * 3
        2. 输出的 position 是在 base frame 下的, 而不是在 hip frame 下的
        """
        pos = dof.squeeze(-1)
        #! 从 pos 中提取出 theta0, theta1, theta2
        theta0 = pos[:,[0,3,6, 9,12,15]] # num_envs, 6 
        theta1 = pos[:,[1,4,7,10,13,16]]
        theta2 = pos[:,[2,5,8,11,14,17]]

        #! 计算 x:  计算 thigh 和 shank 的转动
        x = - torch.sin(theta1)*self._thighLength - torch.sin(theta1+theta2)*self._shankLength

        #! j计算 y:
        y_r = -torch.cos(theta0[:,:3])*self._abadLength + torch.sin(theta0[:,:3]) * (self._thighLength*torch.cos(theta1[:,:3]) + self._shankLength*torch.cos(theta1[:,:3]+theta2[:,:3]))
        y_l = torch.cos(theta0[:,3:])*self._abadLength + torch.sin(theta0[:,3:]) * (self._thighLength*torch.cos(theta1[:,3:]) + self._shankLength*torch.cos(theta1[:,3:]+theta2[:,3:]))
        y = torch.cat((y_r, y_l), dim=-1)
        #! 计算 z: 
        z_r = -torch.sin(theta0[:,:3])*self._abadLength - torch.cos(theta0[:,:3]) * (self._thighLength*torch.cos(theta1[:,:3]) + self._shankLength*torch.cos(theta1[:,:3]+theta2[:,:3]))
        z_l = torch.sin(theta0[:,3:])*self._abadLength - torch.cos(theta0[:,3:]) * (self._thighLength*torch.cos(theta1[:,3:]) + self._shankLength*torch.cos(theta1[:,3:]+theta2[:,3:]))
        z = torch.cat((z_r, z_l), dim=-1)

        foot_position = torch.stack((x,y,z),dim=-1) # (num_envs, 6, 3)
        foot_position += self._abadLocation
        foot_position = foot_position.reshape(-1,3 * NUM_LEG)
        return foot_position

        

    def get_foot_position(self):
        """
        计算脚在 world frame 下的位置
        """
        x, y, z = self.forward_kinematics() # (num_envs, 6)
        foot_position = torch.stack((x,y,z),dim=-1) # (num_envs, 6, 3)
        foot_position += self._abadLocation
        n_foot = foot_position.shape[1]
        
        foot_position = foot_position.reshape(-1,3)

        foot_position = quat_apply(self.base_quat.repeat(1,n_foot), foot_position).reshape(self.num_envs, n_foot, 3)
        #! 这里的 footpos  (num_envs* 6, 3)
        foot_position = foot_position + self.root_states[:, :3].unsqueeze(1) # (num_envs, 6, 3) foot pos in world frame 

        self.measured_foot_positions[:] = foot_position.repeat(1,self.num_foot_height_points,1) + self.foot_height_points.repeat(n_foot,1).unsqueeze(0)

        #! get foot position in world frame
        points = self.measured_foot_positions + self.terrain.cfg.border_size 
        points = (points/ self.terrain.cfg.horizontal_scale).long()
        px = points[:,:,0].view(-1)
        py = points[:,:,1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        height = self.height_samples[px, py]
        height = height.view(self.num_envs, n_foot,self.num_foot_height_points) * self.terrain.cfg.vertical_scale
        self.measured_foot_positions[:,:,2] = height.view(self.num_envs, n_foot * self.num_foot_height_points) #! 这是世界坐标系下 footheight position 
        #! get measured_foot_position in body frame, measured_foot_position (num_envs, n_foot * self.num_foot_height_points, 3)
        self.measured_foot_positions = self.measured_foot_positions - self.root_states[:, :3].unsqueeze(1) # (num_envs, 6, 3) foot pos in body frame
        self.measured_foot_positions = quat_rotate_inverse(self.base_quat.repeat(n_foot * self.num_foot_height_points,1), 
                                                           self.measured_foot_positions.view(-1,3)).reshape(self.num_envs, n_foot * self.num_foot_height_points, 3)
        
        return self.measured_foot_positions[:,:,2]

    def _init_foot_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        measured_foot_points_x = [-0.1,-0.05,0.05,0.1]
        measured_foot_points_y = [-0.1,-0.05,0.05,0.1]

        y = torch.tensor(measured_foot_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(measured_foot_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_foot_height_points = grid_x.numel()
        points = torch.zeros(self.num_foot_height_points, 3, device=self.device, requires_grad=False)
        points[:, 0] = grid_x.flatten()
        points[:, 1] = grid_y.flatten()
        return points
