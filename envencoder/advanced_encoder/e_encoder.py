
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from utils.math import mlp
import torch.distributions as pyd
from envencoder.transition import Transition

MIN_LOGSTD = -1.5
MAX_LOGSTD = 2


class E_Encoder(nn.Module):
	#! 根据单个 (s,a,s',r) 计算 feature
	def __init__(self,obs_dim,act_dim,feature_dim,hidden_size = (256,256),deterministic = True):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.hidden_size = hidden_size 
		self.deterministic = True
		input_dim , output_dim = 2 * obs_dim + act_dim + 1 ,feature_dim
		self.net = mlp([input_dim,]+hidden_size + [output_dim],activation=nn.LeakyReLU,output_activation=nn.LeakyReLU)
		
	def _default_forward(self,obs,act,obs2,rew):
		delta_obs = obs2 - obs
		x = torch.cat((obs,delta_obs,act,rew),dim = -1)
		feature = self.net(x)
		return feature 

	def compute_feature(self,obs,act,obs2,rew):
		feature = self._default_forward(obs,act,obs2,rew)
		return feature
	
		
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

class UDP_Encoder(E_Encoder):
	#! 首先将 transition 转化为 high-dim feature 
	def __init__(self, obs_dim, act_dim, emb_dim,max_env_num, hidden_size=[256, 256], learnable_length_scale=True,length_scale = 1.0,
	      tau = 0.95):
		feature_dim = 128
		super().__init__(obs_dim, act_dim, feature_dim, hidden_size,True)
		self.emb_dim = emb_dim 
		self.max_env_num = max_env_num
		self.W = nn.Parameter(torch.normal(torch.zeros(max_env_num, emb_dim, feature_dim), 0.05),requires_grad=True)
		self.register_buffer("n_env",torch.zeros(1,dtype = torch.long))   #! 记录当前有多少个 env
		self.register_buffer("N", torch.ones(max_env_num)) #! 每个 env 记录过的 embedding 数量
		self.register_buffer(
			"m", torch.normal(torch.zeros(max_env_num,emb_dim), 1) #! embedding 的累积和
		)
		self.register_buffer(
			"e", torch.normal(torch.zeros(max_env_num,emb_dim), 1) #! embedding 均值
		)
		self.learnable_length_scale = learnable_length_scale
		self.m = self.m * self.N.unsqueeze(1) ## ? for the computatin conveniencee

		if learnable_length_scale:
			self.sigma = nn.Parameter(torch.zeros(max_env_num) + length_scale)
		else:
			self.sigma = length_scale
		self.device = torch.device('cpu')
		self.tau = tau 
	
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def add_env(self,n = 1):
		self.n_env += torch.tensor(n,dtype = torch.long,device=self.device)

	def update_emb(self,embs,id):
		# emns.shape = bz,emb_dim
		bz = embs.shape[0]
		self.N[id] = self.N[id] * self.tau + torch.tensor(bz,dtype = torch.long,device=self.device) * (1 - self.tau)
		self.m[id] = self.m[id] * self.tau + embs.sum(0).detach() * (1 - self.tau)
		self.e[id] = self.m[id] / self.N[id].unsqueeze(-1)

	def _set_env(self,n):
		self.n_env = torch.tensor(n,dtype = torch.long)

	def _get_embedding(self):
		return self.e[0:self.n_env]
	
	def _get_sigma(self):
		if self.learnable_length_scale:
			return self.sigma[0:self.n_env]
		return self.sigma

	def calc_embedding(self,obs,act,obs2,rew,no_grad = False):
		"""
		compute the embedding of the given obs,act,obs2,rew
		only consider the first n_env envs
		output n_env,emb_dim
		"""
		#(bz,dim) either (1,dim) or (n_history,dim)
		feature = self.compute_feature(obs,act,obs2,rew)
		if no_grad:
			feature = feature.detach()
		w = self.W[0:self.n_env]
		f2e = torch.einsum("bf,nef->bne",feature,w)
		return f2e


	def calc_distance(self,obs,act,obs2,rew,no_grad = False):
		f2e = self.calc_embedding(obs,act,obs2,rew,no_grad) # bz,n_env,dim
		embedding = self._get_embedding() # n_env,dim
		sigma = self._get_sigma() # n_env
		diff = f2e - embedding.unsqueeze(0)
		distance = (-(diff**2).mean(-1)).div(2 * sigma**2).exp() ## 0 ~ 1.0, represent the prob of being class i 
		return distance # shape (bz,n_class,)

	def inference(self,obs,act,obs2,rew,id = None):
		"""
		if given id, then only consider the given id 
		else, consider all the envs, and infer from the distance
		"""
		assert self.n_env.item()> 0
		f2e = self.calc_embedding(obs,act,obs2,rew) #
		embedding = self._get_embedding() # n_env,dim
		sigma = self._get_sigma() # n_env
		diff = f2e - embedding.unsqueeze(0)
		distance = (-(diff**2).mean(-1)).div(2 * sigma**2).exp() ## 0 ~ 1.0, shae (bz,n_class,)
		if id is None or id >= self.n_env.item():
			idx = torch.argmin(distance,dim = 1,keepdim = True) ## bz,1
			chosen_mean = torch.gather(embedding,0,idx.expand((idx.shape[0],embedding.shape[-1]))).squeeze(1)
		else:
			idx = torch.tensor(id,dtype = torch.long,device = self.device).unsqueeze(0).expand((distance.shape[0],1))
			chosen_mean = embedding[id].unsqueeze(0).expand((distance.shape[0],-1))
		expanded_idx = idx.unsqueeze(-1).expand((-1,1,embedding.shape[-1]))
		chosen_embedding = torch.gather(f2e,-2,expanded_idx).squeeze(-2) 
		chosen_dist = torch.gather(distance,-1,idx)
		
		return chosen_embedding,chosen_dist,idx,chosen_mean,distance
	
	@property
	def num_envs(self):
		return self.n_env.item()
	
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
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.encoder_hidden_size,
			learnable_length_scale=True,
			length_scale = 1.0,
	      	tau = parameter.emb_tau
		)

	