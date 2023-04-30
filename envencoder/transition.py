import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from utils.math import mlp
MIN_LOGSTD = -7
MAX_LOGSTD = 2
def gaussian_nll(
	pred_mean: torch.Tensor,
	pred_logvar: torch.Tensor,
	target: torch.Tensor,
	reduce: bool = True,) -> torch.Tensor:
	l2 = F.mse_loss(pred_mean, target, reduction="none")
	inv_var = (-pred_logvar).exp()
	losses = l2 * inv_var + pred_logvar
	if reduce:
		return losses.sum(dim=1).mean()
	return losses

class Transition(nn.Module):
	def __init__(self,obs_dim,act_dim,emb_dim,deterministic,device):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.deterministic = deterministic
		self.emb_dim =emb_dim 
		self.use_embed = False if emb_dim < 1 else True 
		self.device = device 
	def _default_forward(self,obs,act,emb):
		pass 
	def sample(self,obs,act,emb,deterministic = True):
		
		mean,logstd = self._default_forward(obs,act,emb)
		if (not deterministic) and (not self.deterministic):
			std = logstd.exp()
			return torch.normal(mean, std)
		else:
			return mean
	def _compute_loss(self,obs,act,obs2,emb = None,reduction = True):
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = obs2.cpu().numpy() 
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).sum(-1).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean,logstd = self._default_forward(obs,act,emb)
		info = dict()
		if self.deterministic:
			loss = F.mse_loss(mean,obs2,reduction = 'none').mean(-1)
		else:
			logvar = 2 * logstd
			nll = (gaussian_nll(mean,logvar,obs2,reduce = False)).mean(-1)
			loss = nll
		info['dyanmic_loss'] = np.round(loss.mean().item(),4)
		info['dynamic_explained_variance'] = np.round(explained_var,4)
		info['prediction_error'] = np.round(prediction_error,4)
		if reduction:
			return loss.mean(),info 
		else:
			return loss,info

	def _compute_recons_loss(self,obs,act,emb,obs2):
		return 0.0,{}
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)
	def load(self, path, **kwargs):
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

class SimpleTransition(Transition):
	def __init__(self, obs_dim, act_dim, emb_dim,hidden_size, deterministic, device):
		super().__init__(obs_dim, act_dim, emb_dim, deterministic, device)
		self.net = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.LeakyReLU,nn.LeakyReLU)
		if deterministic:
			self.output_layer = nn.Linear(hidden_size[-1],obs_dim)
		else:
			self.output_layer = nn.Linear(hidden_size[-1],2 * obs_dim)
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def _default_forward(self,obs,act,emb = None):
		if self.use_embed:
			x = torch.cat((obs,act,emb),dim = -1)
		else:
			x = torch.cat((obs,act),dim = -1) 
		res = self.net(x) 
		if self.deterministic:
			mean,logstd = self.output_layer(res),None 
		else:
			mean,logstd = self.output_layer(res).chunk(2,-1) 
			logstd = torch.clamp(logstd,min=MIN_LOGSTD,max=MAX_LOGSTD)
		mean = obs + mean 
		return mean,logstd

	def _compute_recons_loss(self,obs,act,emb,obs2):
		return 0.0,{}
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			deterministic = parameter.transition_deterministic,
			device = parameter.device
		)

class LinearTransition(Transition):
	def __init__(self,obs_dim,act_dim,emb_dim,deterministic,hidden_size,device) -> None:
		super().__init__(obs_dim,act_dim,emb_dim,deterministic,device)
		input_dim = obs_dim + act_dim 
		self.base_net = mlp([input_dim,] + hidden_size + [obs_dim,],nn.ReLU)
		self.gen_matrix =  mlp([input_dim,] + hidden_size,nn.ReLU,nn.ReLU)
		self.v = nn.Linear(hidden_size[-1],emb_dim)
		self.r = nn.Linear(hidden_size[-1],emb_dim)
		self.delta_net = nn.Linear(emb_dim,obs_dim)
		self.deterministic = True 
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def _default_forward(self,obs,act,emb = None):
		input = torch.cat((obs,act),dim = -1)
		base = self.base_net(input)
		matrix = self.gen_matrix(input)
		v,r = self.v(matrix),self.r(matrix)
		A = torch.matmul(v.unsqueeze(-1),r.unsqueeze(-2))
		extra_delta = torch.matmul(A,emb.unsqueeze(-1)).squeeze(-1)
		extra_delta = self.delta_net(torch.tanh(extra_delta))
		delta = base + extra_delta
		return obs + delta ,None


	def _compute_recons_loss(self,obs,act,emb,obs2):
		input = torch.cat((obs,act),dim = -1)
		base = self.base_net(input)
		loss1 = F.mse_loss(base,obs2,reduction='none').sum(-1).mean()
		bz = obs.shape[0]
		bz_ = torch.randperm(bz)
		obs_,act_,obs2_ = obs[bz_],act[bz_],obs2[bz_]

		input_ = torch.cat((obs_,act_),dim=-1)
		base_ = self.base_net(input_)
		loss2 = F.mse_loss((base - base_ ),(obs2 - obs2_),reduction='none') .sum(-1).mean()
		
		loss = loss1 + loss2
		info = {
			'basenet_loss': loss1.item(),
			'additional_loss':loss2.item()
		}
		return loss, info
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			deterministic = parameter.transition_deterministic,
			device = parameter.device
		)
class LearnedLinearTransition(Transition):
	def __init__(self, obs_dim, act_dim, emb_dim, deterministic,hidden_size, device):
		super().__init__(obs_dim, act_dim, emb_dim, deterministic, device)
		## only bias
		input_dim = obs_dim + act_dim 
		self.base_net = mlp([input_dim,] + hidden_size + [obs_dim,],nn.Tanh)
		self.A = nn.Parameter(torch.randn(obs_dim,emb_dim),requires_grad=True)
		self.device = torch.device('cpu')
		self.deterministic = True 
	def _default_forward(self, obs, act, emb):
		input = torch.cat((obs,act),dim = -1)
		base_delta = self.base_net(input)
		extra_delta = torch.matmul(self.A,emb.unsqueeze(-1)).squeeze(-1)
		delta_obs = base_delta + extra_delta 
		next_obs = delta_obs + obs
		return next_obs,None

	def _compute_recons_loss(self,obs,act,emb,obs2):
		input = torch.cat((obs,act),dim = -1)
		base = self.base_net(input)
		loss1 = F.mse_loss(base,obs2,reduction='none').sum(-1).mean()
		bz = obs.shape[0]
		bz_ = torch.randperm(bz)
		obs_,act_,obs2_ = obs[bz_],act[bz_],obs2[bz_]

		input_ = torch.cat((obs_,act_),dim=-1)
		base_ = self.base_net(input_)
		loss2 = F.mse_loss((base - base_ ),(obs2 - obs2_),reduction='none') .sum(-1).mean()
		
		loss = loss1 + loss2
		info = {
			'basenet_loss': loss1.item(),
			'additional_loss':loss2.item()
		}
		return loss, info

	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			deterministic = parameter.transition_deterministic,
			device = parameter.device
		)