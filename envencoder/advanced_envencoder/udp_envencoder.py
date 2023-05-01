import os,torch,itertools
import torch.optim  as optim 
import numpy as np
from utils.math import from_queue,dump_info
from envencoder.advanced_encoder.e_encoder import UDP_Encoder
from envencoder.advanced_encoder.c_encoder import UDP_Decoder
from torch.distributions.categorical import Categorical
from parameter.optm_Params import Parameters
import torch.nn.functional as F


def calc_distance_entropy(correlation):
    logits = correlation
    dist = Categorical(logits=logits)
    ent = dist.entropy()
    return ent

def get_rbf_matrix(data, centers, alpha, element_wise_exp=False):
    out_shape = torch.Size([data.shape[0], centers.shape[0], data.shape[-1]])
    data = data.unsqueeze(1).expand(out_shape)
    centers = centers.unsqueeze(0).expand(out_shape)
    if element_wise_exp:
        mtx = (-(centers - data).pow(2) * alpha).exp().mean(dim=-1, keepdim=False)
    else:
        mtx = (-(centers - data).pow(2) * alpha).sum(dim=-1, keepdim=False).exp()
    return mtx

def get_loss_dpp(y,center = None, kernel='rbf', rbf_radius=3000.0):
    # K = (y.matmul(y.t()) - 1).exp() + torch.eye(y.shape[0]) * 1e-3
    center = y if center is None else center
    if kernel == 'rbf':
        K = get_rbf_matrix(y, center, alpha=rbf_radius, element_wise_exp=False) + torch.eye(y.shape[0], device=y.device) * 1e-3
    elif kernel == 'rbf_element_wise':
        K = get_rbf_matrix(y, center, alpha=rbf_radius, element_wise_exp=True) + torch.eye(y.shape[0], device=y.device) * 1e-3
    elif kernel == 'inner':
        # y = y / y.pow(2).sum(dim=-1, keepdim=True).sqrt()
        K = y.matmul(center.t()).exp()
        # K = torch.softmax(K, dim=0)
        K = K + torch.eye(y.shape[0], device=y.device) * 1e-3
    else:
        assert False
    loss = -torch.logdet(K)
    # loss = -(y.pinverse().t().detach() * y).sum()
    return loss


class UdpEnvencoder:
    def __init__(self,obs_dim,act_dim,emb_dim,parameter:Parameters) -> None:
        self.emb_dim = emb_dim 
        self.parameter = parameter
        self.encoder = UDP_Encoder(obs_dim,act_dim,max_env_num=50,**UDP_Encoder.make_config_from_param(parameter))
        self.world_decoder = UDP_Decoder(obs_dim,act_dim,**UDP_Decoder.make_config_from_param(parameter))
        ## TODO: if we add the new W, we need reset the parameters
        self.envcoder_parameters = itertools.chain(self.encoder.parameters(),self.world_decoder.parameters())
        self.optm = optim.Adam(self.envcoder_parameters, lr= parameter.encoder_lr, amsgrad=False)
        self.device = parameter.device
    
    def get_multitask_value_emb(self,task_indices:list,bz,device,deterministic):
        res = []
        for id in task_indices:
            res.append(self.encoder.get_emb(id,bz))
        res = torch.stack(res,dim=0)
        return res.to(device)

    def add_and_update(self,id,embedding):
        self.encoder.update_emb(embedding.detach(),id)

    def sample(self,obs,act,obs2,rew,deterministic = False,id = None):
        """
        obs: n_history,dim
        可以是 没有 history的obs，需要 expand dim
        但是 return 的时候只能输出 (1,dim) ，对应的是 history 中最好的那个
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            act = act.unsqueeze(0)
            obs2 = obs2.unsqueeze(0)
            rew = rew.unsqueeze(0)
        with torch.no_grad():
            chosen_embedding,chosen_dist,idx,chosen_mean,distance = self.encoder.inference(obs,act,obs2,rew,id = id )
        
        if id is not None:
            chosen_embedding = chosen_embedding.mean(dim=0,keepdim=True)
            chosen_mean = chosen_mean.mean(dim=0,keepdim=True)
        else:
            mode = torch.mode(idx,dim = 0)
            chosen_embedding = chosen_embedding[mode.indices]
            chosen_mean = chosen_mean[mode.indices]
        if deterministic:
            chosen_embedding = chosen_mean
        #! 注意这里是
        return chosen_embedding
    
    def compute_loss(self,
                     support_obs,support_act,support_obs2,support_rew,
                     obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,
                     env_id, 
                     distance_entropy = True,
                     dpp = True,
                     cross_entropy = True,):
        """
        Support: 
            1. bz,dim, 
            2. currently we consider single transition inferecne
        to Predict:
            1. bz,M_to_predict,dim
            2. M_to_predict is the number of transition to predict
        """
        bz = support_obs.shape[0]
        chosen_embedding,chosen_cor,idx,chosen_mean,correlation,all_emb = self.encoder.inference(support_obs,support_act,support_obs2,support_rew,env_id,
                                                                                               with_all_emb=True)
        
        if obs_to_predict.ndim == 3:
            emb_to_use = chosen_embedding.unsqueeze(1).expand(-1,obs_to_predict.shape[1],-1)
        else:
            emb_to_use = chosen_embedding
        
        loss,info = self.world_decoder._compute_loss(obs_to_predict,
                                                     act_to_predict,
                                                     obs2_to_predict,
                                                     rew_to_predict,
                                                     emb_to_use)
        dist = correlation
        ent_loss,ent_info = self._distance_entropy(dist)
        info.update(ent_info)
        if distance_entropy:    
            # dist = self.encoder.calc_distance(support_obs,support_act,support_obs2,support_rew,no_grad=True)
            loss = loss + ent_loss * 1.0
        if dpp:
            dpp_loss,dpp_info = self._emb_ddp(all_emb)
            loss = loss + dpp_loss * 1.0
            info.update(dpp_info)
        if cross_entropy:
            cross_loss,cross_info = self._cross_entropy(correlation,env_id)
            loss = loss + cross_loss * 1.0
            info.update(cross_info)
        
        return loss,info,chosen_embedding
    
    def to(self,device = torch.device('cpu')):
        self.encoder.to(device)
        self.world_decoder.to(device)
    def save(self,path = None):
        if path is None:
            path = self.parameter.model_path
        self.encoder.save(os.path.join(path, 'encoder.pt'))
        self.world_decoder.save(os.path.join(path, 'world_decoder.pt')) 
        torch.save(self.optm.state_dict(), os.path.join(path, 'encoder_optim.pt'))
    def load(self,path):
        if path is None:
            path = self.parameter.model_path
        self.encoder.load(os.path.join(path, 'encoder.pt'))
        self.world_decoder.load(os.path.join(path, 'world_decoder.pt')) 
        self.optm.load_state_dict(torch.load(os.path.join(path, 'encoder_optim.pt')))
        self.envcoder_parameters = itertools.chain(self.encoder.parameters(),self.world_decoder.parameters())
    
    def _distance_entropy(self,dist):
        #! dist.shape = (bz,n_env)
        loss = calc_distance_entropy(dist).mean()
        info = {'dist_entropy':loss.item()}
        return loss,info

    def _emb_ddp(self,embedding,rbf_radius = 1.0):
        #! embedding.shape = (bz,n_env,emb_dim)
        averaged_embedding = embedding.mean(0)
        center_embedding = self.encoder._get_embedding()
        ddp_loss = get_loss_dpp(averaged_embedding,center_embedding,
                                kernel='rbf_element_wise',
                                rbf_radius=rbf_radius)
        info = {
            'ddp_loss':ddp_loss.item(),
            'rbf_radius':rbf_radius
        }

        return ddp_loss,info

    def _cross_entropy(self,dist,id):
        #! dist.shape = (bz,n_env)
        #! id = (bz,)
        id = torch.tensor(id,dtype=torch.int64).to(dist.device).expand(dist.shape[0],)
        loss = F.cross_entropy(dist,id)
        info = {
            'cross_entropy':loss.item()
        }
        return loss,info
