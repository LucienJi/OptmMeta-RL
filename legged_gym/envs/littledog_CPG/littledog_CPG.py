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
from .littledog_CPG_config import LittledogCPGRoughCfg

class LittledogCPG(LeggedRobot):
    cfg : LittledogCPGRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.hopf_oscillator = Hopf_oscillator()
        # initiallize CPG params
        self.x = torch.zeros(self.num_envs, 6, dtype = torch.float, device=self.device, requires_grad=False)
        self.y = torch.tensor([1.,-1.,1.,-1.,1.,-1.], dtype = torch.float, device=self.device, requires_grad=False).repeat(self.num_envs,1)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.
        self.x[env_ids,:] = torch.zeros(6, dtype = torch.float, device=self.device, requires_grad=False)
        self.y[env_ids,:] = torch.tensor([1.,-1.,1.,-1.,1.,-1.], dtype = torch.float, device=self.device, requires_grad=False)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # actions[:,0:6] = 0.1 * actions[:,0:6] -0.35 # h in range[-0.25, -0.45]
        # actions[:,6] = math.pi/3 * actions[:,6] # phi in range[-pi/3, pi/3]
        actions[:,0:6] = 0.045 * actions[:,0:6] + 0.075 # k1 in range[0.03, 0.12]
        actions[:,6] = math.pi/3 * actions[:,6] # phi in range[-pi, pi]
        actions[:,-1] = actions[:,-1] * 0.08 + 0.2 # d_step in [0.12,0.28]

        # print('x:!!!!!!!!!!!!!!!!!!!',self.x[0,:])
        # print('y:!!!!!!!!!!!!!!!!!!!',self.y[0,:])

    
        self.x, self.y = self.hopf_oscillator.hopf(self.x, self.y, steps=0.01)
        foot_x, foot_y, foot_z = self.hopf_oscillator.mapping(self.x, self.y, actions)
        # print('z!!!!!!!!',foot_z[0,:])
        
        target_dof_pos = self.inverse_kinematics(foot_x, foot_y, foot_z, True)

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(target_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits) 

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

    def inverse_kinematics2(self, x, y, z, bendInfo:bool):
        l0 = 0.0802
        l1 = 0.25
        l2 = 0.25

        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1) # num_envs * num_dof(18)

        z_r = z[:,0:3] # right legs
        z_l = z[:,3:6] # left legs
        y_r = y[:,0:3] # right legs
        y_l = y[:,3:6] # left legs

        theta0_r = -torch.atan2(z_r,-y_r) - torch.atan2(torch.sqrt(y_r**2+z_r**2-l0**2),l0*torch.ones_like(z_r))
        theta0_l = torch.atan2(z_l,y_l) + torch.atan2(torch.sqrt(y_l**2+z_l**2-l0**2),l0*torch.ones_like(z_r))

        # yz_square = y**2 + z**2 - l0**2

        # length_square = yz_square + x**2
        # yz = torch.sqrt(yz_square)
        # length = torch.sqrt(length_square)

        # theta0 = torch.atan2(z, -y) + torch.atan2(yz, l0*torch.ones_like(yz))
        # q0 = torch.zeros_like(theta0)
        # q0[0:3,:] = - theta0[0:3,:] 
        # q0[3:6,:] = theta0[3:6,:] 

        # tmp = (l1**2+length_square-l2**2)/2/l1/length
        # tmp = torch.clamp(tmp,-1,1)
        # tmp2 = (l1**2+l2**2-length_square)/2/l1/l2
        # tmp2 = torch.clamp(tmp2,-1,1)
        L = torch.sqrt(y**2+z**2-l0**2)
        # if bendInfo == False:    
        #     theta1 = -torch.atan2(x,yz) - torch.acos(tmp)
        #     theta2 = torch.pi - torch.acos(tmp2)
        
        # else:
        theta1 = -torch.atan2(x,L) + torch.acos((l1**2+L**2+x**2-l2**2)/2/l1/torch.sqrt(L**2+x**2))
        theta2 = -torch.pi + torch.acos(-L**2+l1**2+l2**2-x**2/2/l1/l2)
        # theta1 = -torch.atan2(x,torch.sqrt(L**2-x**2)) + torch.acos((l1**2+L**2-l2**2)/2/l1/L)
        # theta2 = -torch.pi + torch.acos(-L**2+l1**2+l2**2/2/l1/l2)

        pos_action[:,[0,3,6]] = theta0_r
        pos_action[:,[9,12,15]] = theta0_l
        pos_action[:,[1,4,7,10,13,16]] = theta1
        pos_action[:,[2,5,8,11,14,17]] = theta2

        return pos_action

TRIPLE_PHASE = [0, 0.5, 0, 0.5, 0, 0.5]
NUM_LEG = 6

class Hopf_oscillator():
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