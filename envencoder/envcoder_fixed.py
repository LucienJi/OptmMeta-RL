import os, sys
import torch 
import itertools
import torch.optim as optim
import numpy as np 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envencoder.encoder import Fixed_Feature
from envencoder.transition import Transition, SimpleTransition,LinearTransition,LearnedLinearTransition
from utils.math import from_queue
from parameter.optm_Params import Parameters
from envencoder.optimization.maml import MAML
from envencoder.history_tracker import MOCO
import pandas as pd

class FixedEnvcoder:
    def __init__(self,obs_dim,act_dim,emb_dim,parameter:Parameters) -> None:
        self.emb_dim = emb_dim
        self.parameter = parameter
        encoder_params = Fixed_Feature.make_config_from_param(parameter)
        self.transition_type = Transition
        if parameter.transition_type == 'Linear':
            self.transition_type = LinearTransition
        elif parameter.transition_type == 'Simple':
            self.transition_type = SimpleTransition
        elif parameter.transition_type == 'Learned':
            self.transition_type = LearnedLinearTransition
        else:
            raise NotImplementedError
        
        transition_params = self.transition_type.make_config_from_param(parameter)

        ### Encoder
        self.encoder = Fixed_Feature(obs_dim,act_dim,**encoder_params)
        self.maml = MAML(self.encoder, lr = 1e-2)
        self.encoder_to_sample = self.maml
        ### Transition
        self.transition = self.transition_type(obs_dim,act_dim,**transition_params)
        self.transition_target = self.transition_type(obs_dim,act_dim,**transition_params)
        self.transition_target.copy_weight_from(self.transition,0.0)

        # print("Type Check: ",type(self.transition))
        ### MOCO
        self.moco = MOCO(emb_dim,self.parameter.emb_tau,max_sample=100)
        
        self.encoder_task_parameters = itertools.chain(self.maml.parameters(),self.transition.parameters())

        self.optm = optim.Adam(self.encoder_task_parameters, lr= parameter.encoder_lr, amsgrad=False)
        n_update = parameter.max_iter_num * (parameter.min_batch_size / parameter.update_interval) * parameter.inner_iter_num
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optm, T_max=n_update,
                                                              eta_min=parameter.encoder_lr)
        self.epoch = 0
        self.trans_optm = optim.Adam(self.transition.parameters(),lr = parameter.encoder_lr)
        self.device = parameter.device
    def encoder_adapt(self,learner:MAML,model:Transition,
                      obs_support_to_use,act_support_to_use,obs2_support_to_use,
                      obs_support_to_predict,act_support_to_predict,obs2_support_to_predict,
                      adapt_step = 5,first_order = None,allow_nograd = None,support_weight = None)->MAML:
        reduction = True if support_weight is None else False
        for _ in range(adapt_step):
            ## support_to_use.shape (K,dim),support_to_predict (L,dim)
            emb = learner(obs_support_to_use,act_support_to_use,obs2_support_to_use,deterministic=False)
            emb = emb.mean(0)
            emb_to_use = emb.expand((obs_support_to_predict.shape[0],-1))
            support_loss,info = model._compute_loss(obs_support_to_predict,act_support_to_predict,obs2_support_to_predict,emb_to_use,reduction)
            if not reduction:
                support_weight = support_weight.reshape((-1,))
                assert support_weight.shape == support_loss.shape 
                support_loss = (support_weight * support_loss).sum()
            learner.adapt(support_loss,first_order=first_order,allow_nograd=allow_nograd) 
        return learner
    def sample(self,obs_support, act_support, obs2_support, deterministic,support_weight = None, update_flag = False,
                bkg_obs = None,bkg_act  = None ,bkg_obs2 = None):
        if update_flag and bkg_obs is not None:
            learner = self.maml.clone()
            learner = self.encoder_adapt(learner,self.transition,
                                    bkg_obs, bkg_act, bkg_obs2, 
                                    bkg_obs, bkg_act, bkg_obs2,
                                    adapt_step=self.parameter.adapt_step,first_order=True,support_weight=support_weight)
            self.encoder_to_sample = learner
            emb = self.encoder_to_sample.forward(obs_support, act_support, obs2_support,deterministic)
            emb = emb.mean(0).detach()
            # print("Sanity Check3: ",obs_support.shape,emb.shape)
            return emb
        if update_flag or self.encoder_to_sample is None:
            learner = self.maml.clone()
        ## interacton with env, no additional support to predict
            learner = self.encoder_adapt(learner,self.transition,
                                    obs_support, act_support, obs2_support, 
                                    obs_support, act_support, obs2_support,
                                    adapt_step=self.parameter.adapt_step,first_order=True,support_weight=support_weight)
            self.encoder_to_sample = learner
        emb = self.encoder_to_sample.forward(obs_support, act_support, obs2_support,deterministic)
        emb = emb.mean(0).detach()
        return emb
    
    def compute_encoder_loss(self,obs_support_to_use,act_support_to_use,obs2_support_to_use,
                                  obs_support_to_predict,act_support_to_predict,obs2_support_to_predict,
                                  obs_query_to_use,act_query_to_use,obs2_query_to_use,
                                  obs_query_to_predict,act_query_to_predict,obs2_query_to_predict,
                                  with_emb = False,first_order = True,id = None ):
        learner = self.maml.clone() 
        # print("Sanity Check 1: ",obs_support_to_use.shape,obs_support_to_predict.shape)
        learner = self.encoder_adapt(learner,self.transition_target,
                                    obs_support_to_use,act_support_to_use,obs2_support_to_use,
                                    obs_support_to_predict,act_support_to_predict,obs2_support_to_predict,
                                    adapt_step=self.parameter.adapt_step,support_weight=None,first_order=first_order)
        # TODO rethink the adapted maml's embeddings
        emb = learner(obs_query_to_use,act_query_to_use,obs2_query_to_use,deterministic=False) # obs.dim (bz,K,dim)
        assert emb.ndim == 3 and obs_query_to_predict.ndim ==3
        L = obs_query_to_predict.shape[1]
        emb_to_query = torch.mean(emb,dim = 1).unsqueeze(1).expand((-1,L,-1))
        
        # print("Sanity Check 2: ",obs_query_to_predict.shape,emb_to_query.shape)

        query_loss,info = self.transition._compute_loss(obs_query_to_predict,act_query_to_predict,obs2_query_to_predict,emb_to_query)
        if self.parameter.consistency_loss:
            consistency_loss, consistency_info = self.moco._compute_consistency_loss(id,emb.reshape(-1,self.emb_dim)) # emb.shape = (bz * n_shot,dim), tried mean(1), but results in zero loss
            query_loss += consistency_loss + self.parameter.consistency_coef
            info.update(consistency_info)

        if not self.parameter.encoder_deterministic: 
            kl_loss,kl_info = learner._compute_kl_divergence(obs_query_to_use,act_query_to_use,obs2_query_to_use)
            info.update(kl_info)
            query_loss += self.parameter.beta * kl_loss

        if with_emb:
            return query_loss,info,emb.reshape((-1,self.emb_dim)) ## (bz * n_shot,dim)
        else:
            return query_loss,info
    
    def compute_transition_loss(self,obs,act,emb,obs2):
        transition_loss,transition_info = self.transition._compute_recons_loss(obs,act,emb,obs2)
        return transition_loss,transition_info
    
    def update_transition(self,tau = 0.95):
        self.transition_target.copy_weight_from(self.transition,tau)

    def to(self,device = torch.device('cpu')):
        self.encoder.to(device)
        self.maml.to(device)
        self.transition.to(device)
        self.transition_target.to(device)
        self.moco.to(device)
    
    def save(self,path):
        self.encoder.save(os.path.join(path, 'fixed_embedding.pt'))
        self.transition.save(os.path.join(path, 'transition.pt')) 
        self.transition_target.save(os.path.join(path, 'transition_target.pt'))
        self.moco.save(os.path.join(path, 'moco.pt'))

        torch.save(self.optm.state_dict(), os.path.join(path, 'encoder_optim.pt'))
    
    def load(self,path,**kwargs):
        self.encoder.load(os.path.join(path, 'fixed_embedding.pt'),**kwargs)
        self.transition.load(os.path.join(path, 'transition.pt'),**kwargs)
        self.transition_target.load(os.path.join(path, 'transition_target.pt'),**kwargs)
        self.moco.load(os.path.join(path, 'moco.pt'),**kwargs)
        self.optm.load_state_dict(torch.load(os.path.join(path, 'encoder_optim.pt'),
                                                         map_location=self.device))