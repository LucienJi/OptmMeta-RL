import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import get_activation

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