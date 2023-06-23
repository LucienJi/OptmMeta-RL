import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import get_activation
class MLP(nn.Module):
    """
    MLP network
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 256], activation='elu', output_activation="identity") -> None:
        super(MLP,self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim ]
        layers = []
        activation = get_activation(activation)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class ActorCritic(nn.Module):
    def __init__(self, num_obs,num_actions,emb_dim,
                 feature_extractor_dims = [256,256],
                  actor_hidden_dims=[256],
                critic_hidden_dims=[256],
                activation='elu',
                init_noise_std=1.0,) -> None:
        super(ActorCritic,self).__init__()
        # print("Check: ", actor_hidden_dims, critic_hidden_dims,feature_extractor_dims)
        input_dim = num_obs + emb_dim
        self.feature_extractor = MLP(
            input_dim=input_dim,
            output_dim=256,
            hidden_dims=feature_extractor_dims,
            activation=activation,
        )
        self.actor = MLP(input_dim=256,output_dim=num_actions,hidden_dims=actor_hidden_dims,activation=activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.critic = MLP(input_dim=256,output_dim=1,hidden_dims=critic_hidden_dims,activation=activation)
        Normal.set_default_validate_args = False
        self.distribution = None 
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def update_distribution(self,obs,emb):
        features = self.feature_extractor(torch.cat([obs,emb],dim=-1))
        mean = self.actor(features)
        self.distribution = Normal(mean, mean* 0. + self.std)


    def evaluate(self,obs,emb):
        value = self.critic(self.feature_extractor(torch.cat([obs,emb],dim=-1)))
        return value 
    
    def get_actions_log_prob(self,action,gaussain_action):
        #! only accept action in [-1,1]
        #! gaussain_action = torch.atanh(action)
        log_prob = self.distribution.log_prob(gaussain_action)
        log_prob -= torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(-1)
        return log_prob

    def evaluate_actions_log_prob(self,action):
        #! only accept action in [-1,1] 
        action = torch.clamp(action,min=-0.999,max=0.999)
        gaussain_action = torch.atanh(action)
        log_prob = self.distribution.log_prob(gaussain_action)
        log_prob -= torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(-1)
        return log_prob
    
    def get_action_and_value(self, obs, emb):
        self.update_distribution(obs,emb)
        gaussain_action = self.distribution.rsample()
        action = torch.tanh(gaussain_action)
        log_prob = self.get_actions_log_prob(action,gaussain_action)
        entropy = log_prob.mean()
        return action, log_prob, entropy, self.evaluate(obs,emb)