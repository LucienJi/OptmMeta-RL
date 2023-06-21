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
from .littledog_ik_config import LittledogIKRoughCfg

class LittledogIK(LeggedRobot):
    cfg : LittledogIKRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # # load actuator network
        # if self.cfg.control.use_actuator_network:
        #     actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        #     self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

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
        x = actions[:,0:6] * 0.2
        y = actions[:,6:12] * 0.1
        z = actions[:,12:18] * 0.1 - 0.35

        target_dof_pos = self.inverse_kinematics(x, y, z, True)

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
