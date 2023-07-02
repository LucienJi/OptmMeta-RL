
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .utils import get_activation
from .utils import split_and_pad_trajectories, unpad_trajectories
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim = [256,256],output_dim = 16,
                 activation = 'elu',) -> None:
        super(FeatureExtractor,self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        dims = [self.input_dim] + hidden_dim + [output_dim ]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Identity())
        self.mlp = nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class StackEncoder(nn.Module):
    def __init__(self, input_dim,emb_dim = 16,
                 feature_hidden_dim = [256,256],
                 mid_emb_dim = 16,
                 max_step_len=10,activation = 'elu') -> None:
        super(StackEncoder,self).__init__()
        self.max_step_len = max_step_len
        self.feature_extractor = FeatureExtractor(input_dim,
                                                  hidden_dim=feature_hidden_dim,
                                                  output_dim=mid_emb_dim,
                                                  activation=activation)
        mid_dim = max_step_len * mid_emb_dim
        activation = get_activation(activation)
        self.encoder = nn.Sequential(
            nn.Linear(mid_dim,512),
            activation,
            nn.Linear(256,128),
            activation,
            nn.Linear(128,emb_dim),
        )
    
    def forward(self,obs):
        #! only accept (bz,max_step_len,dim)
        feature = self.feature_extractor(obs) #! (bz,max_step_len,mid_emb_dim)
        feature = torch.flatten(feature,start_dim=-2,end_dim=-1) #! (bz,max_step_len*mid_emb_dim)
        emb = self.encoder(feature) #! (bz,emb_dim)
        return emb 


"""
RNN Encoder:
注意，现在应该只是 input sequence of observation， 但是， 需要重新定义 observation 
"""
class RnnEncoder(nn.Module):
    def __init__(self, input_dim, rnn_hidden_size, feature_hidden_dim = [256,128,64], emb_dim = 32, activation = 'elu'):
        super().__init__()
        self.rnn = nn.GRU(input_dim, rnn_hidden_size,num_layers = 1)
        dims = [input_dim + rnn_hidden_size] + feature_hidden_dim
        activation = get_activation(activation)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-1], emb_dim))
        layers.append(nn.Identity())
        self.mlp = nn.Sequential(*layers)

        ## For init hidden state
        self.hidden_states = None 
    
    def reset(self,dones):
        self.hidden_states[..., dones,:] = 0.0 
    
    def forward(self,input, masks = None, hidden_states = None ):
        batch_mode = masks is not None 
        # if batch_mode 我们就是在训练模式， 或者是说， 我们打算用 truncated RNN 
        if batch_mode:
            out,_ = self.rnn(input,hidden_states)


 