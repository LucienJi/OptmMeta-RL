import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from utils.math import mlp
import torch.distributions as pyd
from .transition import SimpleTransition
MIN_LOGSTD = -7
MAX_LOGSTD = 2

class Encoder_ForwardModel(nn.Module):
	def __init__(self,obs_dim,act_dim,emb_dim,hidden_size = [64,64],deterministic = True) -> None:
		super().__init__()
		self.encoder = Base_Encoder(obs_dim,act_dim,emb_dim,hidden_size=[64,64],deterministic=True)
		self.transition = SimpleTransition(obs_dim,act_dim,emb_dim,hidden_size=[64,64],deterministic=True,device = torch.device('cpu'))
		self.device = torch.device('cpu')
	def _default_forward(self,obs,act,obs2):
		mean,logstd = self.encoder._default_forward(obs,act,obs2)
		mean = torch.clamp(mean,min = -5.0,max=5.0)
		return mean,logstd 

	def forward(self,obs,act,obs2,deterministic):
		mean,logstd = self._default_forward(obs,act,obs2)
		return mean

	def _compute_kl_divergence(self,obs,act,obs2):
		mean,logstd = self._default_forward(obs,act,obs2)
		prior = pyd.Normal(torch.zeros_like(mean,device=mean.device),torch.ones_like(logstd,device = logstd.device))
		posterior = pyd.Normal(mean,logstd.exp())
		kl_div = pyd.kl_divergence(posterior,prior)
		loss = kl_div.mean()
		info = {'kl_div':loss.item()}
		return loss,info
		
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.encoder_hidden_size,
			emb_dim=parameter.emb_dim,
			deterministic = parameter.encoder_deterministic,
		)


class Fixed_Feature(nn.Module):
	def __init__(self,obs_dim,act_dim,emb_dim) -> None:
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.embed_dim =emb_dim 
		self.emb = nn.Parameter(torch.zeros(self.embed_dim),requires_grad=True)
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def forward(self,obs,act,obs2,deterministic = False):
		if obs.ndim == 1:
			return self.emb.squeeze()
		else:
			# shape = (...,dim)
			size = obs.shape[0:-1]
			emb = self.emb.expand(size = (size + (self.embed_dim,)))
			return emb

	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim=parameter.emb_dim,
		)
	

class Base_Encoder(nn.Module):
	def __init__(self,obs_dim,act_dim,emb_dim,hidden_size = (256,256),deterministic = True):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.embed_dim =emb_dim 
		self.hidden_size = hidden_size 
		self.deterministic = deterministic
		input_dim , output_dim = 2 * obs_dim + act_dim,emb_dim
		self.net = mlp([input_dim,]+hidden_size,activation=nn.LeakyReLU,output_activation=nn.Tanh)
		if deterministic:
			self.output_layer = nn.Linear(hidden_size[-1],output_dim)
		else:
			self.output_layer = nn.Linear(hidden_size[-1],2 * output_dim)
		self.device = torch.device('cpu')
	
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def _default_forward(self,obs,act,obs2):
		delta_obs = obs2 - obs
		x = torch.cat((obs,delta_obs,act),dim = -1)
		res = self.net(x)
		if self.deterministic:
			mean,logstd = self.output_layer(res),None
		else:
			mean,logstd = self.output_layer(res).chunk(2,-1) 
			logstd = torch.clamp(logstd,min=MIN_LOGSTD,max=MAX_LOGSTD)
		return mean,logstd 

	def forward(self,obs,act,obs2,deterministic):
		mean,logstd = self._default_forward(obs,act,obs2)
		if self.deterministic or deterministic:
			return mean
		else:
			std = logstd.exp()
			dist = pyd.Normal(mean,std)
			sample = dist.rsample()
			return sample
	
	def _compute_kl_divergence(self,obs,act,obs2):
		mean,logstd = self._default_forward(obs,act,obs2)
		prior = pyd.Normal(torch.zeros_like(mean,device=mean.device),torch.ones_like(logstd,device = logstd.device))
		posterior = pyd.Normal(mean,logstd.exp())
		kl_div = pyd.kl_divergence(posterior,prior)
		loss = kl_div.mean()
		info = {'kl_div':loss.item()}
		return loss,info
		
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			hidden_size=parameter.encoder_hidden_size,
			emb_dim=parameter.emb_dim,
			deterministic = parameter.encoder_deterministic,
		)