import torch 
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os 
import torch.nn.functional as F

from envencoder.actor import MLPValue,MLPPolicy,Actor,QNetwork
from parameter.optm_Params import Parameters
from utils.torch_utils import to_device

class SAC(object):
    def __init__(self,obs_dim,act_dim,parameters:Parameters) -> None:
        self.parameter = parameters
        self.policy_config = Actor.make_config_from_param(self.parameter)
        self.value_config = QNetwork.make_config_from_param(self.parameter)

        self.policy = Actor(obs_dim, act_dim, **self.policy_config)
        
        self.policy_parameters = list(self.policy.parameters())

        self.value1 = QNetwork(obs_dim, act_dim, **self.value_config)
        self.value2 = QNetwork(obs_dim, act_dim, **self.value_config)
        self.target_value1 = QNetwork(obs_dim, act_dim, **self.value_config)
        self.target_value2 = QNetwork(obs_dim, act_dim, **self.value_config)
        self.tau = self.parameter.sac_tau

        self.target_value1.load_state_dict(self.value1.state_dict())
        self.target_value2.load_state_dict(self.value2.state_dict())
        for para in self.target_value1.parameters():
            para.requires_grad = False
        for para in self.target_value2.parameters():
            para.requires_grad = False

        self.value_parameters = list(self.value1.parameters()) + list(self.value2.parameters())
        self.value_optimizer = optim.Adam(self.value_parameters,
                                        lr=self.parameter.value_learning_rate) 
        self.policy_optimizer = torch.optim.Adam(list(self.policy_parameters), lr=self.parameter.policy_learning_rate)
        self.device = parameters.device

        self.target_entropy = -torch.prod(torch.Tensor((act_dim,)).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.parameter.value_learning_rate)
        to_device(self.device,self.policy,self.value1, self.value2, self.target_value1, self.target_value2)

    def to(self,device = torch.device('cpu')):
        self.policy.to(device)
        self.value1.to(device)
        self.value2.to(device)
        self.target_value1.to(device) 
        self.target_value2.to(device)
    def compute_q_loss(self,o,a,o2,r,done,emb,emb2):
        with torch.no_grad():
            alpha = self.log_alpha.exp().detach()
            next_action,next_action_log,_ = self.policy.rsample(o2,emb2)
            q1_next_target = self.target_value1(o2,next_action,emb2)
            q2_next_target = self.target_value2(o2,next_action,emb2)
            minq_next_target = torch.min(q1_next_target,q2_next_target) - alpha * next_action_log
            next_q_value = r.flatten() + (1 - done.flatten()) * self.parameter.gamma* (minq_next_target).view(-1)
        
        qf1_a_values = self.value1(o, a,emb).view(-1)
        qf2_a_values = self.value2(o, a,emb).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        info = {
        'qf1_loss':qf1_loss.item(),
        'qf2_loss':qf2_loss.item(),
        "minq_next_target":next_q_value.mean().item(),
        'qf1_value':qf1_a_values.mean().item(),
        'qf2_value':qf2_a_values.mean().item()
        }
        return qf_loss,info

    def compute_policy_loss(self,o,emb = None):
        alpha = self.log_alpha.exp()
        action,action_log,_ = self.policy.rsample(o,emb)
        qf1_pi = self.value1(o, action,emb)
        qf2_pi = self.value2(o, action,emb)
        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
        actor_loss = ((alpha.detach() * action_log) - min_qf_pi).mean()

        info = {
            'Part1: log(alpah) * log(a)': (alpha * action_log).mean().item(),
            'Part2: min_qf':min_qf_pi.mean().item(),
            'act_log_pi':action_log.mean().item(),
            'actor_loss':actor_loss.item(),
        }
        
        return actor_loss,info
    def compute_alpha_loss(self,o,emb = None):
        with torch.no_grad():
            action,action_log,_ = self.policy.rsample(o,emb)
        alpha_loss = (-self.log_alpha * (action_log + self.target_entropy)).mean()
        info = {
            'alpha_loss':alpha_loss.item(),
            'alpha':self.log_alpha.exp().item()
        }
        return alpha_loss,info

    def update_value_function(self):
        self.target_value1.copy_weight_from(self.value1,self.tau)
        self.target_value2.copy_weight_from(self.value2,self.tau)

    def save(self,path):
        self.policy.save(path)
        self.value1.save(path,0)
        self.value2.save(path,1)
        self.target_value1.save(path, "target0")
        self.target_value2.save(path, "target1")
        torch.save(self.policy_optimizer.state_dict(), os.path.join(path, 'policy_optim.pt'))
        torch.save(self.value_optimizer.state_dict(), os.path.join(path, 'value_optim.pt'))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(path, 'alpha_optim.pt'))

    def load(self,path,**kwargs):
        self.policy.load(path,**kwargs)
        self.value1.load(path, 0, **kwargs)
        self.value2.load(path, 1, **kwargs)
        self.target_value1.load(path, "target0",**kwargs)
        self.target_value2.load(path, "target1",**kwargs)
        self.policy_optimizer.load_state_dict(torch.load(os.path.join(path, 'policy_optim.pt'),
                                                         map_location=self.device))
        self.value_optimizer.load_state_dict(torch.load(os.path.join(path, 'value_optim.pt'),
                                                        map_location=self.device))
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(path, 'alpha_optim.pt'),
                                                        map_location=self.device))
