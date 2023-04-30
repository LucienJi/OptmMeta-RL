import os, sys
import torch 
import itertools
import torch.optim as optim
import numpy as np 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.math import from_queue
from parameter.optm_Params import Parameters
from envencoder.history_tracker import MOCO
from envencoder.advanced_encoder.c_encoder import C_Encoder,C_Transition,C_Reward


def average_embs(embs,confidences):
    ## embs.shape = (bz,n_support,dim) or (n_support,dim)
    ## confidences = (bz,n_support,1) or (n_support,1)
    confidences = torch.flatten(confidences,-2,-1)
    confidences = torch.softmax(confidences,dim = -1)
    embs = torch.sum(embs * confidences.unsqueeze(-1),dim = -2)
    return embs # shape (bz,dim,) or (dim,)


def max_embs(embs,confidence):
    ## embs.shape = (bz,n_support,dim) or (n_support,dim)
    ## confidences = (bz,n_support,1) or (n_support,1)
    confidences = torch.flatten(confidence,-2,-1)
    max_id = torch.argmax(confidences,dim = -1,keepdim= False)
    max_id= torch.nn.functional.one_hot(max_id,num_classes=confidences.shape[-1])
    emb = (embs * max_id.unsqueeze(-1)).sum(-2)
    return emb 


class C_Envcoder:
    def __init__(self,obs_dim,act_dim,emb_dim,parameter:Parameters) -> None:
        self.emb_dim = emb_dim
        self.parameter = parameter
        encoder_params = C_Encoder.make_config_from_param(parameter)
        transition_params = C_Transition.make_config_from_param(parameter)
        rew_params = C_Reward.make_config_from_param(parameter)
        ### Encoder
        self.encoder = C_Encoder(obs_dim,act_dim,**encoder_params)
        ### Transition
        self.transition = C_Transition(obs_dim,act_dim,**transition_params)
        self.reward = C_Reward(obs_dim,act_dim,**rew_params)

        # print("Type Check: ",type(self.transition))
        ### MOCO
        self.moco = MOCO(emb_dim,self.parameter.emb_tau)
        self.encoder_task_parameters = itertools.chain(self.encoder.parameters(),self.transition.parameters(),self.reward.parameters())
        self.optm = optim.Adam(self.encoder_task_parameters, lr= parameter.encoder_lr, amsgrad=False)
        self.device = parameter.device
    
    def sample(self,obs_support, act_support, obs2_support,rew_support, method = 'mean'):
        embs,confidences = self.encoder.forward(obs_support,act_support,obs2_support,rew_support)
        if method == 'argmax':
            emb = max_embs(embs,confidences)
        elif method == 'mean':
            emb = average_embs(embs,confidences)
        return emb 
    
    def compute_encoder_loss(self,obs,act,obs2,rew,
                            obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,
                            env_id):
        ## obs.shape = (bz,1,dim) 
        ## obs2_to_predict = (bz,M,dim)
        embs,confidences = self.encoder.forward(obs,act,obs2,rew)
        bz = embs.shape[0]
        target_embs = self.moco.get_value_emb(env_id,bz,device=embs.device,deterministic=True)
        assert embs.shape[1] == 1 and confidences.shape[1] == 1 
        embs,confidences = embs.reshape((bz,-1)),confidences.reshape((bz,1))
        # print(embs.shape,confidences.shape,target_embs.shape)
        embs_to_use = confidences * embs + (1 - confidences) * target_embs.detach()
        ## Transition Loss
        M = obs_to_predict.shape[1]
        extended_embs_to_use = embs_to_use.unsqueeze(-2).expand(size=(bz,M,-1))
        transition_loss,transition_info = self.transition._compute_loss(obs_to_predict,act_to_predict,obs2_to_predict,extended_embs_to_use)
        rew_loss,rew_info = self.reward._compute_loss(obs_to_predict,act_to_predict,rew_to_predict,extended_embs_to_use)
        transition_loss += rew_loss
        confidence_loss,confidence_info = self.encoder._compute_log_constraint(obs,act,obs2,rew,confidences)
        transition_loss += self.parameter.log_confidence_coef * confidence_loss

        transition_info.update(confidence_info)
        transition_info.update(rew_info)
        ## use embs or embs_to_use to update
        return transition_loss,transition_info,embs.detach()



    def to(self,device = torch.device('cpu')):
        self.encoder.to(device)
        self.transition.to(device)
        self.reward.to(device)
        self.moco.to(device)

    def save(self,path):
        self.encoder.save(os.path.join(path, 'encoder.pt'))
        self.transition.save(os.path.join(path, 'transition.pt')) 
        self.reward.save(os.path.join(path, 'reward.pt'))
        self.moco.save(os.path.join(path, 'moco.pt'))

        torch.save(self.optm.state_dict(), os.path.join(path, 'encoder_optim.pt'))
    
    def load(self,path,**kwargs):
        self.encoder.load(os.path.join(path, 'encoder.pt'),**kwargs)
        self.transition.load(os.path.join(path, 'transition.pt'),**kwargs)
        self.reward.load(os.path.join(path, 'reward.pt'),**kwargs)
        self.moco.load(os.path.join(path, 'moco.pt'),**kwargs)
        self.optm.load_state_dict(torch.load(os.path.join(path, 'encoder_optim.pt'),
                                                         map_location=self.device))
                                                
    def load_state_dict(self,other_envcoder):
        self.encoder.load_state_dict(other_envcoder.encoder.state_dict())
        self.transition.load_state_dict(other_envcoder.transition.state_dict())
        self.reward.load(other_envcoder.reward.state_dict())
    



