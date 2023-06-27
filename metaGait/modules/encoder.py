
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .utils import get_activation

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim,act_dim,reward_dim,
                 hidden_dim = [256,256],output_dim = 16,
                 activation = 'elu',) -> None:
        super(FeatureExtractor,self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reward_dim = reward_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        input_dim = obs_dim + act_dim + reward_dim + obs_dim
        dims = [input_dim] + hidden_dim + [output_dim ]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Identity())
        self.mlp = nn.Sequential(*layers)
    def forward(self,obs,act,reward,obs2):
        delta = obs2 - obs 
        x = torch.cat([obs,act,reward,delta],dim=-1)
        return self.mlp(x)

class StackEncoder(nn.Module):
    def __init__(self, obs_dim,act_dim,reward_dim,emb_dim = 16,
                 feature_hidden_dim = [256,256],
                 mid_emb_dim = 16,
                 max_step_len=10,activation = 'elu') -> None:
        super(StackEncoder,self).__init__()
        self.max_step_len = max_step_len

        self.feature_extractor = FeatureExtractor(obs_dim,act_dim,reward_dim,
                                                  hidden_dim=feature_hidden_dim,
                                                  output_dim=mid_emb_dim,
                                                  activation=activation)
        mid_dim = max_step_len * mid_emb_dim
        activation = get_activation(activation)
        self.encoder = nn.Sequential(
            nn.Linear(mid_dim,256),
            activation,
            nn.Linear(256,emb_dim),
        )
    
    def forward(self,obs,act,reward,obs2):
        #! only accept (bz,max_step_len,dim)
        feature = self.feature_extractor(obs,act,reward,obs2) #! (bz,max_step_len,mid_emb_dim)
        feature = torch.flatten(feature,start_dim=-2,end_dim=-1) #! (bz,max_step_len*mid_emb_dim)
        emb = self.encoder(feature) #! (bz,emb_dim)
        return emb 

class RnnEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        