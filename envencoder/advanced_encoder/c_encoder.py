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

#LOG_STD_MAX = 2
# LOG_STD_MIN = -1.5

class UdpReward(nn.Module):
	def __init__(self,obs_dim, act_dim, emb_dim,hidden_size, device) -> None:
		super().__init__()
		self.net1 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.mean = nn.Linear(hidden_size[-1],1)
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def _default_forward(self,obs,act,emb = None):
		x = torch.cat((obs,act,emb),dim = -1)
		mean = self.mean(self.net1(x))
		return mean
	def forward(self,obs,act,emb):
		mean = self._default_forward(obs,act,emb)
		return mean 
	def sample(self,obs,act,emb,deterministic):
		mean = self._default_forward(obs,act,emb)
		return mean

	def _compute_loss(self,obs,act,reward,emb = None):
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = reward.cpu().numpy() 
			assert y_pred.shape == y_true.shape
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		info = {}
		mean = self._default_forward(obs,act,emb)
		loss = F.mse_loss(mean,reward,reduction='mean')
		info['reward_loss'] = np.round(loss.item(),4)
		info['reward_explained_variance'] = np.round(explained_var,4)
		info['reward_prediction_error'] = np.round(prediction_error,4)
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
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)

class UdpTransition(Transition):
	def __init__(self, obs_dim, act_dim, emb_dim,hidden_size, device):
		super().__init__(obs_dim, act_dim, emb_dim, False, device)
		self.net1 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.mean = nn.Linear(hidden_size[-1],obs_dim)
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def _default_forward(self,obs,act,emb = None):
		x = torch.cat((obs,act,emb),dim = -1)
		mean = self.mean(self.net1(x))
		#! here, we predict the delta, instead of the next obs
		mean = obs + mean 
		return mean
	
	def forward(self,obs,act,emb):
		mean = self._default_forward(obs,act,emb)
		obs2 = mean 
		return obs2

	def _compute_loss(self,obs,act,obs2,emb = None):
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = obs2.cpu().numpy() 
			assert y_pred.shape == y_true.shape
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean = self._default_forward(obs,act,emb)
		info = dict()
		loss = F.mse_loss(mean,obs2,reduction='mean')

		info['dyanmic_loss'] = np.round(loss.item(),4)
		info['dynamic_explained_variance'] = np.round(explained_var,4)
		info['dynamic_prediction_error'] = np.round(prediction_error,4)
		return loss,info
	
	def sample(self,obs,act,emb,deterministic):
		mean= self._default_forward(obs,act,emb)
		return mean

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)

class UDP_Decoder(nn.Module):
	def __init__(self,obs_dim,act_dim,with_reward,with_transition,emb_dim,hidden_size,device):
		super().__init__()
		assert with_reward or with_transition
		self.with_reward,self.with_transition = with_reward,with_transition
		self.reward = UdpReward(obs_dim,act_dim,emb_dim,hidden_size,device) 
		self.transition = UdpTransition(obs_dim,act_dim,emb_dim,hidden_size,device) 
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def sample_reward(self,obs,act,emb,deterministic):
		if not self.with_reward: 
			emb = emb.detach()
		return self.reward.sample(obs,act,emb,deterministic)
	def sample_transition(self,obs,act,emb,deterministic):
		if not self.with_transition: 
			emb = emb.detach()
		return self.transition.sample(obs,act,emb,deterministic)
	def _compute_loss(self,obs,act,obs2,rew,emb = None):
		loss,info = 0.0,{}
		if not self.with_transition:
			trans_emb = emb.detach()
		else:
			trans_emb = emb 
		trans_loss,trans_info = self.transition._compute_loss(obs,act,obs2,trans_emb)
		loss += trans_loss
		info.update(trans_info)
		if not self.with_reward:
			rew_emb = emb.detach()
		else:
			rew_emb = emb
		rew_loss,rew_info = self.reward._compute_loss(obs,act,rew,rew_emb) 
		loss += rew_loss
		info.update(rew_info)
		return loss,info 
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			with_reward = parameter.with_reward,
			with_transition = parameter.with_transition,
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

class World_Decoder(nn.Module):
	def __init__(self,obs_dim,act_dim,with_reward,with_transition,emb_dim,hidden_size,device):
		super().__init__()
		assert with_reward or with_transition
		self.with_reward,self.with_transition = with_reward,with_transition
		self.reward = C_Reward(obs_dim,act_dim,emb_dim,hidden_size,device) 
		self.transition = C_Transition(obs_dim,act_dim,emb_dim,hidden_size,device) 
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def sample_reward(self,obs,act,emb,deterministic):
		if not self.with_reward: 
			emb = emb.detach()
		return self.reward.sample(obs,act,emb,deterministic)
	def sample_transition(self,obs,act,emb,deterministic):
		if not self.with_transition: 
			emb = emb.detach()
		return self.transition.sample(obs,act,emb,deterministic)
	def _compute_loss(self,obs,act,obs2,rew,emb = None):
		loss,info = 0.0,{}
		if not self.with_transition:
			trans_emb = emb.detach()
		else:
			trans_emb = emb 
		trans_loss,trans_info = self.transition._compute_loss(obs,act,obs2,trans_emb)
		loss += trans_loss
		info.update(trans_info)
		if not self.with_reward:
			rew_emb = emb.detach()
		else:
			rew_emb = emb
		rew_loss,rew_info = self.reward._compute_loss(obs,act,rew,rew_emb) 
		loss += rew_loss
		info.update(rew_info)
		return loss,info 
	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			with_reward = parameter.with_reward,
			with_transition = parameter.with_transition,
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)
	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)

	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		self.load_state_dict(torch.load(path, map_location=map_location))

class C_Reward(nn.Module):
	def __init__(self,obs_dim, act_dim, emb_dim,hidden_size, device) -> None:
		super().__init__()
		self.net1 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.net2 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.mean = nn.Linear(hidden_size[-1],1)
		self.logstd = nn.Linear(hidden_size[-1],1)
		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	def _default_forward(self,obs,act,emb = None):
		x = torch.cat((obs,act,emb),dim = -1)
		mean,logstd = self.mean(self.net1(x)),self.logstd(self.net2(x))
		logstd = torch.clamp(logstd,min=MIN_LOGSTD,max=MAX_LOGSTD)
		return mean,logstd
	def forward(self,obs,act,emb,reward = None):
		mean,logstd = self._default_forward(obs,act,emb)
		std = logstd.exp()
		dist = pyd.Normal(mean,std)
		if reward is None:
			reward = dist.rsample()
		log_prob = dist.log_prob(reward)
		log_prob = torch.sum(log_prob,dim = -1,keepdim=True)
		return reward,log_prob
	def sample(self,obs,act,emb,deterministic):
		mean,logstd = self._default_forward(obs,act,emb)
		std = logstd.exp()
		if deterministic:
			return mean
		else:
			dist = pyd.Normal(mean,std)
			obs2 = dist.rsample()
			return obs2

	def _compute_loss(self,obs,act,reward,emb = None):
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = reward.cpu().numpy() 
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean,logstd = self._default_forward(obs,act,emb)
		info = dict()
		loss = -1.0 * pyd.Normal(mean,logstd.exp()).log_prob(reward).mean()
		loss += F.mse_loss(mean,reward,reduction='mean')

		info['reward_loss'] = np.round(loss.item(),4)
		info['reward_explained_variance'] = np.round(explained_var,4)
		info['reward_prediction_error'] = np.round(prediction_error,4)
		return loss,info
	
	def _compute_weighted_loss(self,obs,act,reward,emb,confidence):
		assert confidence.ndim == obs.ndim
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = reward.cpu().numpy() 
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean,logstd = self._default_forward(obs,act,emb)
		info = dict()
		loss = -1.0 * pyd.Normal(mean,logstd.exp()).log_prob(reward)
		loss += F.mse_loss(mean,reward,reduction='none')
		loss = (loss * confidence).mean()

		info['reward_loss'] = np.round(loss.item(),4)
		info['reward_explained_variance'] = np.round(explained_var,4)
		info['reward_prediction_error'] = np.round(prediction_error,4)
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
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)

class C_Transition(Transition):
	def __init__(self, obs_dim, act_dim, emb_dim,hidden_size, device):
		super().__init__(obs_dim, act_dim, emb_dim, False, device)
		self.net1 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.net2 = mlp([obs_dim + act_dim + emb_dim,]+hidden_size,nn.Tanh,nn.Tanh)
		self.mean = nn.Linear(hidden_size[-1],obs_dim)
		self.logstd = nn.Linear(hidden_size[-1],obs_dim)

		self.device = torch.device('cpu')
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def _default_forward(self,obs,act,emb = None):
		x = torch.cat((obs,act,emb),dim = -1)
		mean,logstd = self.mean(self.net1(x)),self.logstd(self.net2(x))
		logstd = torch.clamp(logstd,min=MIN_LOGSTD,max=MAX_LOGSTD)
		# mean = obs + mean 
		return mean,logstd
	
	def forward(self,obs,act,emb,obs2 = None):
		mean,logstd = self._default_forward(obs,act,emb)
		std = logstd.exp()
		dist = pyd.Normal(mean,std)
		if obs2 is None:
			obs2 = dist.rsample()
		log_prob = dist.log_prob(obs2)
		log_prob = torch.sum(log_prob,dim = -1,keepdim=True)
		return obs2,log_prob
	
	def _compute_loss(self,obs,act,obs2,emb = None):
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = obs2.cpu().numpy() 
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean,logstd = self._default_forward(obs,act,emb)
		info = dict()
		loss = -1.0 * pyd.Normal(mean,logstd.exp()).log_prob(obs2).mean()
		loss += F.mse_loss(mean,obs2,reduction='mean')

		info['dyanmic_loss'] = np.round(loss.item(),4)
		info['dynamic_explained_variance'] = np.round(explained_var,4)
		info['dynamic_prediction_error'] = np.round(prediction_error,4)
		return loss,info
	
	def _compute_weighted_loss(self,obs,act,obs2,emb,confidence):
		assert confidence.ndim == obs.ndim
		with torch.no_grad():
			y_pred = self.sample(obs,act,emb,deterministic=True).cpu().numpy()
			y_true = obs2.cpu().numpy() 
			var_y = np.var(y_true)
			prediction_error = ((y_pred - y_true)**2).mean()
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
		mean,logstd = self._default_forward(obs,act,emb)
		info = dict()
		loss = -1.0 * pyd.Normal(mean,logstd.exp()).log_prob(obs2)
		loss += F.mse_loss(mean,obs2,reduction='none')
		loss = (loss * confidence).mean()
		info['dyanmic_loss'] = np.round(loss.item(),4)
		info['dynamic_explained_variance'] = np.round(explained_var,4)
		info['dynamic_prediction_error'] = np.round(prediction_error,4)
		return loss,info

	
	def _compute_confidence_loss(self,obs,act,obs2,emb,confidence):
		mean,logstd = self._default_forward(obs,act,emb)
		std = logstd.exp()
		dist = pyd.Normal(mean,std)
		log_prob = dist.log_prob(obs2)
		target_prob = log_prob.exp().detach()
		c_loss = ((confidence - target_prob)**2).mean()
		c_info = {
			'prob_loss':c_loss.item(),
			'average_prob':target_prob.mean().item(),
			'average_confidence':confidence.mean().item()
		}
		return c_loss,c_info
	
	def sample(self,obs,act,emb,deterministic):
		mean,logstd = self._default_forward(obs,act,emb)
		std = logstd.exp()
		if deterministic:
			return mean
		else:
			dist = pyd.Normal(mean,std)
			obs2 = dist.rsample()
			return obs2

	@staticmethod
	def make_config_from_param(parameter):
		return dict(
			emb_dim = parameter.emb_dim,
			hidden_size = parameter.transition_hidden_size,
			device = parameter.device
		)

class C_Encoder(nn.Module):
	def __init__(self,obs_dim,act_dim,emb_dim,hidden_size = (256,256),deterministic = True):
		super().__init__()
		self.obs_dim = obs_dim 
		self.act_dim = act_dim 
		self.embed_dim =emb_dim 
		self.hidden_size = hidden_size 
		self.deterministic = True
		input_dim , output_dim = 2 * obs_dim + act_dim + 1 ,emb_dim
		self.net = mlp([input_dim,]+hidden_size,activation=nn.LeakyReLU,output_activation=nn.LeakyReLU)
		self.confidence_branch = mlp([hidden_size[-1],]+hidden_size + [1,],activation=nn.LeakyReLU,output_activation=nn.Sigmoid)
		self.predict_branch = mlp([hidden_size[-1],]+hidden_size + [output_dim,],activation=nn.LeakyReLU)
		self.device = torch.device('cpu')
	
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)

	def _default_forward(self,obs,act,obs2,rew):
		delta_obs = obs2 - obs
		x = torch.cat((obs,delta_obs,act,rew),dim = -1)
		res = self.net(x)
		prediction,confidence = self.predict_branch(res),self.confidence_branch(res)
		return prediction,confidence

	def forward(self,obs,act,obs2,rew,deterministic = False):
		pred,confidence = self._default_forward(obs,act,obs2,rew)
		return pred,confidence
	
	def _compute_log_constraint(self,obs,act,obs2,rew,confidence = None):
		if confidence is None:
			_,confidence = self.forward(obs,act,obs2,rew)
		c_loss = -torch.log(confidence).mean()
		c_info = {
			'average_confidence':confidence.mean().item(),
			'c_loss':c_loss.item()
		}
		return c_loss,c_info
	
		
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
		)