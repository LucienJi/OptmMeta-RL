from collections import deque
from envencoder.advanced_encoder.c_encoder import C_Encoder,C_Transition
from envencoder.actor import Actor
import numpy as np
import torch
from scipy.special import softmax

def average_embs(list_embs,list_confidences):
    #[(dim,)],[(1,)]
    embs = torch.stack(list_embs,dim = 0) # N,dim
    confidences = torch.stack(list_confidences,dim = 0) # N, 1 
    confidences = torch.softmax(confidences,dim=0)
    embs = torch.sum((embs * confidences),dim = 0)
    return embs

def max_embs(list_embs,list_confidences):
    #[(dim,)],[(1,)]
    confidences = torch.stack(list_confidences,dim = 0) # N, 1 
    confidences = torch.flatten(confidences,-2,-1)
    max_id = torch.argmax(confidences,dim = 0)
    emb = list_embs[max_id]
    return emb


# class NS_Tracker_v1():
#     def __init__(self,obs_dim,act_dim,emb_dim,max_history_length,
#             threshold,prob_constant) -> None:
#         self.max_history_length = max_history_length
#         self.obs_dim,self.act_dim = obs_dim,act_dim
#         self.emb_dim = emb_dim
#         self.threshold = threshold
#         self.C = prob_constant

#         self.emb_support = list() # list of tensor, no grad required 
#         self.c_support = list() # list of tensor
        
#         self.list_s = list() # list of float
#         self.list_S = list() # list of float 
#         self.max_ratio_G = 0.0 
#         self.old_emb = torch.zeros(size=(self.emb_dim,))
    
#     def update(self,obs,act,obs2,
#                 encoder:C_Encoder,transition:C_Transition,policy:Actor):
#         emb,c = encoder.forward(obs,act,obs2)
#         self.emb_support.append(emb)
#         self.c_support.append(c)
#         ## Emb under H0
#         emb_under_h0 = emb* c + self.old_emb * (1-c) 
#         ## Joint Prob if No Env Change
#         _,log_prob_transition = transition.forward(obs,act,emb_under_h0,obs2)
#         log_prob_action = policy.get_prob(obs,act,emb_under_h0)

#         ## log likeliratio of New Env(C) / 
#         s = np.log(self.C) - (log_prob_transition + log_prob_action)
#         self.list_s.append(s)
#         S = self.list_S[-1] + s 
#         self.list_S.append(S)
#         self.max_ratio_G = max(self.max_ratio_G + s, 0.0) 
        
    
#     def _check_ns(self):
#         if self.max_ratio_G > self.threshold:
#             # H1 hold
#             estimated_nc = np.argmax(self.list_S)
#             cur_t = len(self.list_S)
#             estimated_delay = cur_t - estimated_nc 
#             return estimated_nc
#         else:
#             return -1 
    
#     def _update(self,nc,method = 'average'):
#         list_embs = self.emb_support[nc:]
#         list_confidences = self.c_support[nc:] 
#         if method == 'method':
#             embs = average_embs(list_embs,list_confidences)
#         elif method == 'max':
#             embs = max_embs(list_embs,list_confidences)
#         return embs 
    
#     def reset(self):
#         self.emb_support = list() # list of tensor, no grad required 
#         self.c_support = list() # list of tensor
#         self.list_s = list() # list of float
#         self.list_S = list() # list of float 
#         self.max_ratio_G = 0.0 


class NS_Tracker_v2():
    def __init__(self,obs_dim,act_dim,emb_dim,threshold) -> None:
        self.obs_dim,self.act_dim = obs_dim,act_dim
        self.emb_dim = emb_dim 
        self.threshold = threshold 
        self.reset()

        ## for debug 
        self.__last_emb = None 
        self.__last_s = None 
        self.__last_S = None 
        self.__last_G = None 
        self.__last_c = None 
        

    def feed(self,device):
        # use self.emb_0 or average all listed emb
        
        ## V1 
        emb = self.emb_h0
        ## V2 
        # emb = average_embs(self.emb_support,self.c_support)

        emb = emb.to(device)
        return emb 
    
    def __debug(self,emb,s,S,G,c):
        self.__last_emb = emb 
        self.__last_G = G 
        self.__last_S = S 
        self.__last_s = s 
        self.__last_c = c
    
    def debug(self):
        return self.__last_emb,self.__last_s,self.__last_S,self.__last_G,self.__last_c


    def update(self,obs,act,obs2,
                encoder:C_Encoder,transition:C_Transition,policy:Actor):
        with torch.no_grad():
            emb,c = encoder.forward(obs,act,obs2)
            c = c * 0.5

        self._update_emb_h1(emb,c)
        self._update_emb_h0(emb,c)

        s = self.calc_ratio(obs,act,obs2,transition,policy)
        S = self.list_S[-1] + s 
        self.G = max(0.0, self.G + s)

        self.__debug(emb.numpy(),s,S,self.G,c.numpy()) 
        self.list_S.append(S)
        self.list_s.append(s)

        if self.G > self.threshold:
            delay = self.post_change()
            return True,delay
        else:
            return False,-1 

    
    def _update_emb_h1(self,emb,c):

        ## update emb under H1 
        self.emb_h1 = emb
        # last_c,last_emb =  self.c_support[-1], self.emb_support[-1]
        # self.emb_h0 =  last_c * last_emb + (1 - last_c) * self.emb_h0

        self.emb_support.append(emb)
        self.c_support.append(c)

    def _update_emb_h0(self,emb,c):

        ## update emb under H0
        self.emb_h0 = self.emb_h0 * (1 - c) + emb * c
        


    def reset(self,emb_h0 = None,emb_h1 = None ):
        self.list_s = list() # single logporb ratio
        self.list_S = list() # cumulative ratio
        # self.list_s = deque(maxlen=10)
        # self.list_S = deque(maxlen=10)

        self.G = 0.0
        self.list_S.append(0.0)
        self.list_s.append(0.0)

        self.emb_support = list() # torch.Tensor
        self.c_support = list() # torch.Tensor
        # self.emb_support = deque(maxlen=10)
        # self.c_support = deque(maxlen=10)
        
        self.emb_h0 = emb_h0 if emb_h0 is not None else torch.zeros(size = (self.emb_dim,))
        self.emb_h1 = emb_h1 if emb_h1 is not None else torch.zeros(size = (self.emb_dim,))
        self.emb_support.append(self.emb_h0)
        self.c_support.append(torch.ones(size = (1,)))

    def calc_ratio(self,obs,act,obs2,transition:C_Transition,policy:Actor):
        with torch.no_grad():
            ## Calc Prob under H0 with self.emb_h0
            _,log_prob_transition_0 = transition.forward(obs,act, self.emb_h0,obs2)
            log_prob_action_0 = policy.get_prob(obs,act, self.emb_h0)

            ## Calc Prob under H1 with self,emb_h1
            _,log_prob_transition_1 = transition.forward(obs,act, self.emb_h1,obs2)
            log_prob_action_1 = policy.get_prob(obs,act, self.emb_h1) 
        s = (log_prob_transition_1 + log_prob_action_1) - (log_prob_transition_0 + log_prob_action_0) 
        # s = log_prob_transition_1 - log_prob_transition_0
        # s = log_prob_action_1 - log_prob_action_0
        return s.numpy()

    def post_change(self):
        assert self.G > self.threshold
        # estimated_nc = np.argmax(self.list_S)
        assert len(self.emb_support) == len(self.list_s)

        estimated_nc = np.argmax(self.list_s)
        delay = len(self.list_S) - estimated_nc 

        ## the result of burn in
        valid_emb = self.emb_support[estimated_nc:]
        # print("Change Detection: ", len(valid_emb), valid_emb)
        valid_confidence = self.c_support[estimated_nc:]
        new_proposed_emb = average_embs(valid_emb,valid_confidence)
        # print("New Proposed Embs: ",new_proposed_emb)


        ## we should reset the buffer 
        self.reset(new_proposed_emb,new_proposed_emb)
        return delay
        









        
