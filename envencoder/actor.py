import torch
import os, sys
import numpy as np
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_base import RNNBase
import time

LOG_STD_MAX = 2
LOG_STD_MIN = -1.5

def mlp(sizes, activation, output_activation=nn.Identity):
	layers = []
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [layer_init(nn.Linear(sizes[j], sizes[j+1])), act()]
	return nn.Sequential(*layers)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.kaiming_normal_(layer.weight,a = 0,nonlinearity='relu')
	# torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer
class Base(nn.Module):
	def __init__(self) -> None:
		super().__init__()
	def info(self, info):
		print(info)
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		self.info(f'saving model to {path}..')
		torch.save(self.state_dict(), path)
	def load(self, path, **kwargs):
		self.info(f'loading from {path}..')
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

	def copy_weight_from(self, src_net, tau):
		"""I am target net, tau ~~ 1
			if tau = 0, self <--- src_net
			if tau = 1, self <--- self
		"""
		with torch.no_grad():
			if tau == 0.0:
				self.load_state_dict(src_net.state_dict())
				return
			elif tau == 1.0:
				return
			for param, target_param in zip(src_net.parameters(True), self.parameters(True)):
				target_param.data.copy_(target_param.data * tau + (1-tau) * param.data)
				

class QNetwork(Base):
	def __init__(self,obs_dim,act_dim,emb_dim,hidden_size):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.use_emb = True if emb_dim > 0 else False 
		self.emb_dim = emb_dim 
		input_dim = obs_dim + act_dim + emb_dim 
		self.hidden_size = hidden_size
		self.net = mlp([input_dim,]+hidden_size + [1,],nn.ReLU)

	def forward(self,o,a,emb = None):
		if self.use_emb and emb is not None:
			input = torch.cat((o,a,emb),dim = -1)
		else:
			input = torch.cat((o,a),dim = -1) 
		return self.net(input)
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.value_hidden_size,
			emb_dim=parameter.emb_dim
		)
	def save(self, path, index=0):
		super().save(os.path.join(path, f'value{index}.pt'))
	def load(self, path, index=0, **kwargs):
		super().load(os.path.join(path, f'value{index}.pt'), **kwargs)
	
class Actor(Base):
	def __init__(self,obs_dim,act_dim,emb_dim,hidden_size):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.use_emb = True if emb_dim > 0 else False 
		self.emb_dim = emb_dim
		self.hidden_size = hidden_size
		input_dim = obs_dim  + emb_dim 
		self.soft_plus = torch.nn.Softplus()
		self.net = mlp([input_dim,]+hidden_size,nn.ReLU)
		self.mu = nn.Linear(hidden_size[-1],self.act_dim)
		self.logstd = nn.Linear(hidden_size[-1],self.act_dim)
		
		self.device = torch.device('cpu') 
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def _default_forward(self,x,emb):
		input = torch.cat((x,emb),dim = -1)
		x = self.net(input)
		mean,logstd = self.mu(x),self.logstd(x)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
		log_std = torch.clamp(log_std,LOG_STD_MIN,LOG_STD_MAX)
		return mean, log_std
	def rsample(self,x,emb):
		mu, log_std = self._default_forward(x,emb)
		std = log_std.exp()
		normal = torch.distributions.Normal(mu, std)
		x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
		y_t = torch.tanh(x_t)
		log_prob =  normal.log_prob(x_t)
		# Enforcing Action Bound
		log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)
		mu = torch.tanh(mu)
		return y_t,log_prob,mu

	def act(self,x,emb,deterministic):
		action,log_prob,mu = self.rsample(x,emb)
		if deterministic:
			return mu 
		else:
			return action
	
	def get_prob(self,obs,act,emb):
		mu,log_std = self._default_forward(obs,emb)
		std = log_std.exp()
		normal = torch.distributions.Normal(mu,std)
		x = torch.atanh(act)
		log_prob = normal.log_prob(x)
		log_prob = log_prob.sum(-1, keepdim=True)
		return log_prob


	def save(self, path):
		super().save(os.path.join(path, 'mlppolicy.pt'))
	def load(self, path, **kwargs):
		super().load(os.path.join(path, 'mlppolicy.pt'), **kwargs)
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.policy_hidden_size,
			emb_dim=parameter.emb_dim,
		)

	

class MLPBase(RNNBase):
	def __init__(self, input_size, output_size, hidden_size_list, activation):
		super().__init__(input_size, output_size, hidden_size_list, activation, ['fc'] * len(activation))

	def meta_forward(self,x):
		_meta_start_time = time.time()
		for ind,layer in enumerate(self.layer_list):
			activation = self.activation_list[ind]
			layer_type = self.layer_type[ind]
			x = layer(x)
			if activation is not None:
				x = activation(x)
		self.cumulative_meta_forward_time += time.time() - _meta_start_time
		return x

class MLPPolicy(torch.nn.Module):
	def __init__(self, obs_dim, act_dim, hidden_size, activations,
				 emb_dim, stop_pg_for_ep=True,
				 bottle_neck=False, bottle_sigma=1e-4):
		super(MLPPolicy, self).__init__()
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		# stop the gradient from ep when inferring action.
		self.stop_pg_for_ep = stop_pg_for_ep
		self.bottle_neck = bottle_neck
		self.bottle_sigma = bottle_sigma
		# aux dim: we add ep to every layer inputs.
		self.emb_dim = emb_dim
		self.net = MLPBase(obs_dim + emb_dim, act_dim * 2, hidden_size, activations)
		# ep first, up second
		self.module_list = torch.nn.ModuleList(self.net.total_module_list)
		self.soft_plus = torch.nn.Softplus()
		self.min_log_std = -7.0
		self.max_log_std = 2.0
		self.ep_tensor = None
		self.allow_sample = True
		self.device = torch.device('cpu')
	def set_deterministic(self, deterministic):
		self.allow_sample = not deterministic
	def make_init_action(self, device=torch.device('cpu')):
		return torch.zeros((1, self.act_dim), device=device)
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def meta_forward(self,x,emb):
		if self.bottle_neck and self.allow_sample:
			emb = emb + torch.rand_like(emb) * self.bottle_sigma
		if self.stop_pg_for_ep:
			emb = emb.detach()
		input = torch.cat((x,emb),dim = -1)
		policy_out = self.net.meta_forward(input)
		return policy_out
	def forward(self,x,emb,require_log_std = False):
		policy_out = self.meta_forward(x,emb)
		mu, log_std = policy_out.chunk(2, dim=-1)
		log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
		std = log_std.exp()
		if require_log_std:
			return mu, std, log_std
		return mu, std
	def rsample(self,x,emb):
		mu, std, log_std = self.forward(x,emb, require_log_std=True)
		noise = torch.randn_like(mu).detach() * std
		sample = noise + mu
		log_prob = (- 0.5 * (noise / std).pow(2) - (log_std + 0.5 * np.log(2 * np.pi))).sum(-1, keepdim=True)
		log_prob = log_prob - (2 * (- sample - self.soft_plus(-2 * sample) + np.log(2))).sum(-1, keepdim=True)
		return torch.tanh(mu), std, torch.tanh(sample), log_prob
	def act(self,x,emb,deterministic):
		mu,std,action,log_prob = self.rsample(x,emb)
		if deterministic:
			return mu 
		else:
			return action
	def save(self, path):
		self.net.save(os.path.join(path, 'mlppolicy.pt'))

	def load(self, path, **kwargs):
		self.net.load(os.path.join(path, 'mlppolicy.pt'), **kwargs)

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.policy_hidden_size,
			activations=parameter.policy_activations,
			emb_dim=parameter.emb_dim,
			bottle_neck=parameter.bottle_neck,
			bottle_sigma=parameter.bottle_sigma,
			stop_pg_for_ep =parameter.p_stop_pg_for_ep
		)

	def inference_one_step(self, obs,emb, deterministic=True):
		self.set_deterministic_ep(deterministic)
		with torch.no_grad():
			mu, std, act, logp, self.sample_hidden_state = self.rsample(obs,emb)
		if deterministic:
			return mu
		return act

class MLPValue(torch.nn.Module):
	def __init__(self, obs_dim, act_dim, hidden_size, activations,
				 emb_dim, stop_pg_for_ep=True):
		super(MLPValue, self).__init__()
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		# stop the gradient from ep when inferring action.
		self.stop_pg_for_ep = stop_pg_for_ep
		self.net = MLPBase(obs_dim + emb_dim + act_dim, 1, hidden_size, activations)
		# ep first, up second
		self.module_list = torch.nn.ModuleList(self.net.total_module_list)
		self.min_log_std = -7.0
		self.max_log_std = 2.0
	def meta_forward(self,obs,act,emb):
		if self.stop_pg_for_ep:
			emb = emb.detach()
		input = torch.cat((obs,act,emb),dim = -1)
		v_out = self.net.meta_forward(input)
		return v_out 
	def forward(self,obs,act,emb):
		return self.meta_forward(obs,act,emb)
	def save(self, path, index=0):
		self.net.save(os.path.join(path, f'value{index}.pt'))
	def load(self, path, index=0, **kwargs):
		self.net.load(os.path.join(path, f'value{index}.pt'), **kwargs)
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.value_hidden_size,
			activations=parameter.value_activations,
			emb_dim=parameter.emb_dim,
			stop_pg_for_ep=parameter.v_stop_pg_for_ep
		)
	def copy_weight_from(self, src, tau):
		"""
		I am target net, tau ~~ 1
		if tau = 0, self <--- src_net
		if tau = 1, self <--- self
		"""
		self.net.copy_weight_from(src.net, tau)






if __name__ == '__main__':
	hidden_layers = [256, 128, 64]
	hidden_activates = ['leaky_relu'] * len(hidden_layers)
	hidden_activates.append('tanh')
	nn = MLPBase(64, 4, hidden_layers, hidden_activates)
	for _ in range(5):
		x = torch.randn((3, 64))
		y = nn.meta_forward(x)
		print(y)
	