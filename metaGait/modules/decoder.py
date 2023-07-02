import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import get_activation
from .utils import MLP
class WorldDecoder(nn.Module):
    def __init__(self, obs_dim,act_dim,rew_dim,emb_dim,hidden_dim,activation = 'elu') -> None:
        super(WorldDecoder,self).__init__()
        input_dim = obs_dim + act_dim  + emb_dim
        dims = [input_dim] + hidden_dim
        activation = get_activation(activation)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)
        self.obs_head = nn.Linear(hidden_dim[-1],obs_dim)
        self.rew_head = nn.Linear(hidden_dim[-1],rew_dim)
    def forward(self,obs,act,emb):
        x = torch.cat([obs,act,emb],dim=-1)
        x = self.mlp(x)
        obs = self.obs_head(x)
        rew = self.rew_head(x)
        return obs,rew

class VelDecoder(nn.Module):
    """
    1. 通过历史信息预测当前 base frame 中的 线速度和 角速度
    """
    def __init__(self, emb_dim, vel_dim,hidden_dims,activation = 'elu') -> None:
        super(VelDecoder,self).__init__()
        input_dim = emb_dim
        self.net = MLP(input_dim,vel_dim,hidden_dims,activation)
    def forward(self,emb):
        return self.net(emb)

class ObsDecoder(nn.Module):
    """
    1. 通过历史信息和 当前 base frame 中的 线速度和 角速度 预测当前:
        1. joint position, vel  
        2. cmd, gravity 
        3. 可能的 internal state 
    """
    def __init__(self, emb_dim, vel_dim, obs_dim) -> None:
        super(ObsDecoder,self).__init__()
        input_dim = emb_dim + vel_dim
        output_dim = obs_dim
        self.net = MLP(input_dim,output_dim)
    def forward(self,emb,vel):
        x = torch.cat([emb,vel],dim=-1)
        return self.net(x)
    

############### Height 和 Contact Force 
############### 需要使用 obs 和 privileged obs 
class HeightPredictor(nn.Module):
    """
    1. 通过历史信息,和当前的 partial observation (可以是 basic, 也可以是 all), 预测当前的 height
    """
    def __init__(self, emb_dim, obs_dim, num_footposition,num_leg,hidden_dims,activation = 'elu') -> None:
        super(HeightPredictor,).__init__()

        num_points_to_predict = num_footposition * num_leg
        self.net = MLP(emb_dim + obs_dim,128,hidden_dims,activation)
        self.mean = nn.Linear(128,num_points_to_predict)
        self.logstd = nn.Linear(128,num_points_to_predict)
    def _default_forward(self,emb,obs):
        x = torch.cat([emb,obs],dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        return mean,logstd

class ForcePredictor(nn.Module):
    """
    1. 通过历史信息,和当前的 partial observation (可以是 basic, 也可以是 all), 预测当前的 height
    """
    def __init__(self, emb_dim, obs_dim ,num_leg,hidden_dims,activation = 'elu') -> None:
        super(ForcePredictor,).__init__()
        num_points_to_predict = 3 * num_leg
        self.net = MLP(emb_dim + obs_dim,128,hidden_dims,activation)
        self.mean = nn.Linear(128,num_points_to_predict)
        self.logstd = nn.Linear(128,num_points_to_predict)
    def _default_forward(self,emb,obs):
        x = torch.cat([emb,obs],dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        logstd = self.logstd(x)
        return mean,logstd

    