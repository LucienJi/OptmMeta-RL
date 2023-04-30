import os,torch,itertools
import torch.optim  as optim 
import numpy as np
from utils.math import from_queue,dump_info
from envencoder.advanced_encoder.e_encoder import UDP_Encoder
from envencoder.advanced_encoder.c_encoder import World_Decoder
from torch.distributions.categorical import Categorical
from parameter.optm_Params import Parameters

def distance2logit(distance):
    return -distance

def calc_distance_entropy(distance):
    logits = distance2logit(distance)
    # proba = torch.softmax(logits,dim=-1)
    dist = Categorical(logits=logits)
    ent = dist.entropy()
    return ent



class UdpEnvencoder:
    def __init__(self,obs_dim,act_dim,emb_dim,parameter:Parameters) -> None:
        self.emb_dim = emb_dim 
        self.parameter = parameter
        self.encoder = UDP_Encoder(obs_dim,act_dim,max_env_num=50,**UDP_Encoder.make_config_from_param(parameter))
        self.world_decoder = World_Decoder(obs_dim,act_dim,**World_Decoder.make_config_from_param(parameter))
        ## TODO: if we add the new W, we need reset the parameters
        self.envcoder_parameters = itertools.chain(self.encoder.parameters(),self.world_decoder.parameters())
        self.optm = optim.Adam(self.envcoder_parameters, lr= parameter.encoder_lr, amsgrad=False)
        self.device = parameter.device
    
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
                     env_id, distance_entropy = True):
        """
        Support: 
            1. bz,dim, 
            2. currently we consider single transition inferecne
        to Predict:
            1. bz,M_to_predict,dim
            2. M_to_predict is the number of transition to predict
        """
        bz = support_obs.shape[0]
        chosen_embedding,chosen_dist,idx,chosen_mean,distance = self.encoder.inference(support_obs,support_act,support_obs2,support_rew,env_id)
        
        if obs_to_predict.ndim == 3:
            emb_to_use = chosen_embedding.unsqueeze(1).expand(-1,obs_to_predict.shape[1],-1)
        else:
            emb_to_use = chosen_embedding
        
        loss,info = self.world_decoder._compute_loss(obs_to_predict,
                                                     act_to_predict,
                                                     obs2_to_predict,
                                                     rew_to_predict,
                                                     emb_to_use)
        if distance_entropy:
            # dist = self.encoder.calc_distance(support_obs,support_act,support_obs2,support_rew,no_grad=True)
            dist = distance
            ent_loss = calc_distance_entropy(dist).mean()
            loss = loss + ent_loss * 0.1 
            info['dist_entrop'] = ent_loss.item()
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
    