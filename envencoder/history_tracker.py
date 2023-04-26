from collections import deque
from collections import namedtuple
import sys,os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import copy 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math import from_queue
from envencoder.history_tracker_v2 import CUSUM

MIN_STD = np.exp(-7.0)
MAX_STD = np.exp(2.0)

tuplenames = ('obs', 'act','obs2','rew')
Support = namedtuple('Support', tuplenames)

class EnvTracker(object):
	def __init__(self,obs_dim,act_dim,n_support) -> None:
		self.history_length = n_support 
		self.obs_dim,self.act_dim = obs_dim,act_dim
		self.obs_support = deque(maxlen= self.history_length)
		self.act_support = deque(maxlen= self.history_length)
		self.obs2_support = deque(maxlen= self.history_length)
		self.rew_support = deque(maxlen = self.history_length)
		self.pred_error_support = deque(maxlen= self.history_length)
	def feed_support(self,device):
		support = Support(from_queue(self.obs_support,device),from_queue(self.act_support,device),from_queue(self.obs2_support,device),from_queue(self.rew_support,device))
		return support

	def update_history(self,obs,act,obs2,rew,pred):
		err = ((pred - obs2)**2).mean()
		self.obs_support.append(obs)
		self.act_support.append(act)
		self.obs2_support.append(obs2)
		self.rew_support.append(rew)
		self.pred_error_support.append(err)
	def init(self):
		self.clear()
		self.obs_support.append(np.zeros(shape=(self.obs_dim,)))
		self.act_support.append(np.zeros(shape=(self.act_dim,)))
		self.obs2_support.append(np.zeros(shape=(self.obs_dim,)))
		self.rew_support.append(np.zeros(shape=(1,)))
		self.pred_error_support.append(0)
	def clear(self):
		self.obs_support.clear()
		self.act_support.clear()
		self.obs2_support.clear()
		self.rew_support.clear()
		self.pred_error_support.clear()





def get_rbf_matrix(data, centers, alpha, element_wise_exp=False):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    if element_wise_exp:
        mtx = (-(centers - data).pow(2) * alpha).exp().mean(dim=-1, keepdim=False)
    else:
        mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
    return mtx


def get_loss_dpp(y, kernel='rbf', rbf_radius=3000.0):
    # K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0]) * 1e-3
    if kernel == 'rbf':
        K = get_rbf_matrix(y, y, alpha=rbf_radius, element_wise_exp=False) + torch.eye(y.shape[0], device=y.device) * 1e-3
    elif kernel == 'rbf_element_wise':
        K = get_rbf_matrix(y, y, alpha=rbf_radius, element_wise_exp=True) + torch.eye(y.shape[0], device=y.device) * 1e-3
    elif kernel == 'inner':
        # y = y / y.pow(2).sum(dim=-1, keepdim=True).sqrt()
        K = y.matmul(y.t()).exp()
        # K = torch.softmax(K, dim=0)
        K = K + torch.eye(y.shape[0], device=y.device) * 1e-3
        # print(K)
        # print('k shape: ', K.shape, ', y_mtx shape: ', y_mtx.shape)
    else:
        assert False
    loss = -torch.logdet(K)
    # loss = -(y.pinverse().t().detach() * y).sum()
    return loss


class MOCO(nn.Module):
	"""
	### Description
	maintain the mean vectors for each env; for training Q (since stable)
	maintain the most recent batched env vectors; for being sampled for policy or
	calculte the diversity / consistency of mean vectors
	"""
	def __init__(self,emb_dim,tau) -> None:
		super(MOCO,self).__init__()
		self.tau = tau 
		self.emb_dim = emb_dim
		self.mean_embs = nn.ParameterDict()
		self.std_embs = nn.ParameterDict()
		self.loss_dictionary = dict()
		self.loss_decay = 0.95
		self.unknown_embs = {}
		self.device = torch.device('cpu')
	
	def track_loss(self,id,loss):
		id = str(id)
		if id not in self.loss_dictionary:
			self.loss_dictionary[id] = loss
		else:
			loss = self.loss_decay * self.loss_dictionary[id] + (1-self.loss_decay) * loss
			self.loss_dictionary[id]= loss
	
	def check_loss(self,id = None):
		id = str(id)
		if (id is not None) and (id not in self.loss_dictionary):
			return 0.0 
		if id is None:
			return np.mean(list(self.loss_dictionary.values()))
		else:
			return self.loss_dictionary[id]


	def add_and_update(self,id,batch_embs):
		if batch_embs.ndim == 3:
			batch_embs = batch_embs.reshape((-1,self.emb_dim))
			
		id = str(id)
		mean = torch.mean(batch_embs,dim = 0).detach()
		std = torch.clamp(torch.std(batch_embs,dim = 0,unbiased = False),min = MIN_STD,max=MAX_STD).detach()
		if id not in self.mean_embs:
			self.mean_embs[id] = nn.Parameter(mean,requires_grad=False).float()
			self.std_embs[id] = nn.Parameter(std,requires_grad=False).float()
		else:
			self.mean_embs[id].data = self.mean_embs[id].data * self.tau + mean.data * (1 - self.tau)
			self.std_embs[id].data = self.std_embs[id].data * self.tau + std.data * (1 - self.tau)

	def get_value_emb(self,id,batch_size,device,deterministic = True):
		id = str(id)
		if id not in self.mean_embs:
			emb = torch.zeros((self.emb_dim,)).float()
			if deterministic:
				embs_to_use = emb.expand((batch_size,-1)).to(device)
			else:
				std = torch.ones_like(emb)
				normal = Normal(emb,std)
				embs_to_use = normal.sample((batch_size,)).to(device)
		else:
			if deterministic:
				embs_to_use = self.mean_embs[id]
				embs_to_use = embs_to_use.expand((batch_size,-1)).to(device)
			else:
				mean,std = self.mean_embs[id],self.std_embs[id]
				normal = Normal(mean.data,std.data)
				embs_to_use = normal.sample((batch_size,)).to(device)
		return embs_to_use 
	
	def get_multitask_value_emb(self,task_inds,batch_size,device,deterministic = False):
		list_embs = [self.get_value_emb(id, batch_size,device,deterministic) for id in task_inds]
		embs_to_use = torch.stack(list_embs,dim = 0)
		return embs_to_use

	def sample(self,id,device,batch_size = None):
		id = str(id)
		if id not in self.mean_embs:
			mean,std = torch.zeros((self.emb_dim,)).float(),torch.ones((self.emb_dim,)).float()
		else:
			mean,std = self.mean_embs[id],self.std_embs[id]
		normal = Normal(mean.data,std.data)
		if batch_size is not None:
			sample = normal.sample((batch_size,))
			return sample.to(device)
		else:
			return normal.sample().to(device)
	
	def _compute_consistency_loss(self,id,batch_emb):
		id = str(id)
		if (id is None) or (id not in self.mean_embs):
			mean_emb = batch_emb.mean(0).detach()
		else:
			mean_emb = self.mean_embs[id].data
		var =(batch_emb - mean_emb.reshape(1,-1)).pow(2).mean()
		std = var.sqrt()
		info = {
			'consistency_loss':std.item()
		}
		return std , info
	
	def _compute_diversity_loss(self,mean_embs,kernel='rbf', rbf_radius=80.0):
		## mean_embs.shape = (n_env,dim)
		loss = get_loss_dpp(mean_embs,kernel,rbf_radius)
		info = {
			'diversity_loss':loss.item()
		}
		return loss, info 


	def save(self, path):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)
		
	def load(self, path, **kwargs):
		map_location = None
		if 'map_location' in kwargs:
			map_location = kwargs['map_location']
		data = torch.load(path, map_location=map_location)
		for k,v in data.items():
			n,id = k.split('.')
			if 'mean_embs' in n:
				self.mean_embs[id] = v
			elif 'std_embs' in n:
				self.std_embs[id] = v
			else:
				raise NotImplementedError

	def copy_weight_from(self, src_net):
		"""I am target net, tau ~~ 1
			if tau = 0, self <--- src_net
			if tau = 1, self <--- self
		"""
		with torch.no_grad():
			self.mean_embs = copy.deepcopy(src_net.mean_embs)
			self.std_embs = copy.deepcopy(src_net.std_embs)
			
	def to(self, device):
		if not device == self.device:
			self.device = device
			super().to(device)
	


if __name__ == '__main__':
	mean = torch.zeros((3,))
	stds = torch.ones((3,))
	samples = Normal(mean,stds).sample((50,))
	moco = MOCO(3,0.95,10)
	moco.update_mean(mean,id = 0)
	print(samples.shape)

	var = moco.compute_env_std(samples,0)
	print(var)

	
