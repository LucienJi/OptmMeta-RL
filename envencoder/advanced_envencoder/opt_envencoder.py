import os,torch 
import itertools
import torch.optim as optim
import numpy as np 

from utils.math import from_queue,dump_info
from parameter.optm_Params import Parameters
from envencoder.optimization.maml import MAML
from envencoder.history_tracker import MOCO
from envencoder.advanced_encoder.c_encoder import C_Encoder,C_Reward,C_Transition,World_Decoder

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

def get_per_step_loss_importance_vector(
        number_of_training_steps_per_iter,
        current_epoch,
        multi_step_loss_num_epochs,device
        ):
 
        loss_weights = np.ones(shape=(number_of_training_steps_per_iter)) * (
                1.0 / number_of_training_steps_per_iter)
        decay_rate = 1.0 / number_of_training_steps_per_iter / multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / number_of_training_steps_per_iter
        
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device)
        return loss_weights


class OptEnvencoder:
    def __init__(self,obs_dim,act_dim,emb_dim,parameter:Parameters) -> None:
        self.emb_dim = emb_dim 
        self.parameter = parameter
        encoder_params = C_Encoder.make_config_from_param(parameter)
        
        self.world_decoder = World_Decoder(obs_dim,act_dim,**World_Decoder.make_config_from_param(parameter))
        self.encoder = C_Encoder(obs_dim,act_dim,**encoder_params)
        self.maml = MAML(model = self.encoder,lr = parameter.meta_lr) ## ada_lr = 5-3 ,too small ?
        
        self.moco = MOCO(emb_dim,self.parameter.emb_tau)

        self.envcoder_parameters = itertools.chain(self.maml.parameters(),self.world_decoder.parameters())
        self.optm = optim.Adam(self.envcoder_parameters, lr= parameter.encoder_lr, amsgrad=False)
        self.device = parameter.device
    def sample(self,obs_support, act_support, obs2_support, res_support, method = 'argmax'):
        with torch.no_grad():
            embs,confidences = self.maml.forward(obs_support,act_support,obs2_support,res_support)
        if method == 'argmax':
            emb = max_embs(embs,confidences)
        elif method == 'mean':
            emb = average_embs(embs,confidences)
        return emb 

    def _compute_world_loss(self,obs,act,obs2,rew,emb):
        loss,info = self.world_decoder._compute_loss(obs,act,obs2,rew,emb)
        return loss,info

    def _compute_loss(self,obs,act,obs2,rew,
                        obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,
                        env_id,use_target_embs = True):
        ## Normal Version
        ## obs.shape = (bz,1,dim)
        embs,confidences = self.maml.forward(obs,act,obs2,rew)
        bz = embs.shape[0]
        M = obs_to_predict.shape[1]
        assert embs.shape[1] == 1, print("Trianing Error: N_support Error")
        embs,confidences = embs.reshape((bz,-1)),confidences.reshape((bz,1))
        confidence_loss,confidence_info = self.encoder._compute_log_constraint(obs,act,obs2,rew,confidences)
        if use_target_embs:
            target_embs = self.moco.get_value_emb(env_id,bz,device=embs.device,deterministic=True)
            embs_to_use = confidences * embs + (1 - confidences) * target_embs.detach()
        else:
            embs_to_use = embs 
            confidence_loss = confidence_loss.detach()
        extended_embs_to_use = embs_to_use.unsqueeze(-2).expand(size=(bz,M,-1))
        loss,info = self._compute_world_loss(obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,extended_embs_to_use)
        info.update(confidence_info)
        loss = loss+ self.parameter.log_confidence_coef * confidence_loss
        return loss,info,embs.detach()

    def compute_encoder_loss(self,obs,act,obs2,rew,
                            obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,
                            env_id,with_maml,use_target_embs,lam1 = 1.0,lam2 = 0.1):
        ## obs.shape = (bz,1,dim)
        ## obs_to_predict = (bz,M,dim)
        assert obs.shape[1] == 1, print("Trianing Error: N_support Error")
        bz = obs.shape[0]
        M = obs_to_predict.shape[1]
        if not with_maml:
            loss,trans_info ,embs= self._compute_loss(obs,act,obs2,rew,obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,env_id,use_target_embs)
            trans_info['sub_task_loss1'] = 0.0
            trans_info['sub_task_loss2'] = 0.0
            return loss,trans_info,embs
        else:
            adapt_step = 5
            # weights = get_per_step_loss_importance_vector(adapt_step,cur_epoch,total_epoch,device=obs.device)
            meta_loss,info = 0.0,{}
            learner = self.maml.clone(first_order=True,allow_nograd=True,allow_unused=True)
            embs,confidences = self.maml.forward(obs,act,obs2,rew)
            embs,confidences = embs.reshape((bz,-1)),confidences.reshape((bz,1))
            for _ in range(adapt_step):
                learner = self.adapt_1_step(learner,obs,act,obs2,rew,first_order=True)
            new_embs,new_confidence = learner.forward(obs,act,obs2,rew) 
            new_embs,new_confidence = new_embs.reshape((bz,-1)),new_confidence.reshape((bz,1))
            ## Loss 1:
            target_embs = self.moco.get_value_emb(env_id,bz,device=embs.device,deterministic=True)
            embs1 = confidences * embs + (1-confidences)*new_embs.detach()
            embs1 = embs1.unsqueeze(-2).expand(size=(bz,M,-1))
            embs2 = new_confidence *  new_embs + ( 1 - new_confidence) * target_embs.detach()
            embs2 = embs2.unsqueeze(-2).expand(size=(bz,M,-1))
            loss1,info1 = self._compute_world_loss(obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,embs1)
            loss2,info2 = self._compute_world_loss(obs_to_predict,act_to_predict,obs2_to_predict,rew_to_predict,embs2)            
            confidence_loss,confidence_info = self.encoder._compute_log_constraint(obs,act,obs2,rew,new_confidence)
            meta_loss = loss1 * lam1 + loss2 * lam2 + confidence_loss * self.parameter.log_confidence_coef
            info = dump_info(info,info1)
            info = dump_info(info,info2)
            info = dump_info(info,confidence_info)
            info['sub_task_loss1'],info['sub_task_loss2'] = loss1.item(),loss2.item()
            return meta_loss,info,embs.detach()


            


    def adapt_1_step(self,learner:MAML,obs,act,obs2,rew,first_order = True):
        emb,confidence = learner.forward(obs,act,obs2,rew)
        loss,info = self._compute_world_loss(obs,act,obs2,rew,emb)
        learner.adapt(loss,first_order=first_order,allow_unused=True,allow_nograd=True)
        return learner
    

    def to(self,device = torch.device('cpu')):
        self.encoder.to(device)
        self.maml.to(device)
        self.world_decoder.to(device)
        self.moco.to(device)

    def save(self,path):
        self.encoder.save(os.path.join(path, 'encoder.pt'))
        self.world_decoder.save(os.path.join(path, 'world_decoder.pt')) 
        self.moco.save(os.path.join(path, 'moco.pt'))
        torch.save(self.optm.state_dict(), os.path.join(path, 'encoder_optim.pt'))
    
    def load(self,path,**kwargs):
        self.encoder.load(os.path.join(path, 'encoder.pt'),**kwargs)
        self.world_decoder.load(os.path.join(path, 'world_decoder.pt'),**kwargs)
        self.moco.load(os.path.join(path, 'moco.pt'),**kwargs)
        self.optm.load_state_dict(torch.load(os.path.join(path, 'encoder_optim.pt'),
                                                         map_location=self.device))
        
        self.maml = MAML(self.encoder,
        lr = self.maml.lr,first_order = self.maml.first_order,
        allow_unused = self.maml.allow_unused,
        allow_nograd=self.maml.allow_nograd)
    
    def load_state_dict(self,other_envcoder):
        self.encoder.load_state_dict(other_envcoder.encoder.state_dict())
        self.maml.load_state_dict(other_envcoder.maml.state_dict())
        self.world_decoder.load_state_dict(other_envcoder.world_decoder.state_dict())
        
