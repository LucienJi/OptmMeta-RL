import torch 
import torch.nn as nn 
from metaGait.storage.replay_buffer import StackedReplayBuffer
from metaGait.modules.encoder import StackEncoder
from metaGait.modules.decoder import WorldDecoder
import itertools 
class TaskEncoder:
    encoder:StackEncoder
    decoder:WorldDecoder
    def __init__(self,
                 encoder,decoder,
                 batch_size = 4096,num_mini_batch = 1,num_learning_epochs=1,
                 capacity = 10000,max_history_len = 10,
                 learning_rate = 1e-3,max_grad_norm = 1.0,device = 'cpu'):

        self.device = device
        self.encoder = encoder
        self.encoder.to(self.device)
        self.decoder = decoder 
        self.decoder.to(self.device)

        self.storage = None
        self.stacked_transition = None
        self.transition = None 

        self.parameters = itertools.chain(self.encoder.parameters(),
                                          self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.parameters,lr = learning_rate)
        

        # Model Parameters 
        self.num_mini_batch = num_mini_batch
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_learning_epochs = num_learning_epochs
        self.capacity = capacity
        self.max_history_len = max_history_len
        self.learning_rate = learning_rate 
    
    def init_storage(self,num_envs,obs_shape,act_shape):
        self.storage = StackedReplayBuffer(self.capacity,num_envs,obs_shape,act_shape,self.max_history_len,self.device)
        self.stacked_transition = StackedReplayBuffer.StackedTransition(num_envs,self.max_history_len,obs_shape,act_shape,self.device)
        self.transition = StackedReplayBuffer.Transition() 

    def test_mode(self):
        self.encoder.eval()
        self.decoder.eval()
    def train_mode(self):
        self.encoder.train()
        self.decoder.train()
    
    def act(self):
        emb = self.encoder.forward(self.stacked_transition.observations,self.stacked_transition.actions,
                                   self.stacked_transition.rewards,self.stacked_transition.next_observations,).detach()
        return emb 
    
    def prcess_env_step(self,obs,act,rew,obs2,done):
        self.transition.observations = obs.detach()
        self.transition.actions = act.detach()
        self.transition.rewards = rew.detach()
        self.transition.next_observations = obs2.detach()
        #! 这个顺序很重要, 使用 old stacked transition 来预测新的 transition
        self.storage.add_transitions(self.stacked_transition,self.transition)

        self.stacked_transition.add_transition(obs,act,rew,obs2)
        self.stacked_transition.reset_idx(done)

    def update(self):
        #! 这里计算 目标函数
        mean_obs_prediction_loss = 0
        mean_rew_prediction_loss = 0

        generator = self.storage.mini_batch_generator(self.batch_size,self.num_mini_batch,self.num_learning_epochs)
        for stacked_obs,stacked_act,stacked_rew,stacked_obs2, obs,act,rew,obs2 in generator:
            #! 这里计算 目标函数
            emb = self.encoder.forward(stacked_obs,stacked_act,stacked_rew,stacked_obs2)
            obs2_hat,rew_hat = self.decoder.forward(obs,act,emb)
            obs_prediction_error = (obs2_hat - obs2).pow(2).mean()
            rew_prediction_error = (rew_hat - rew).pow(2).mean()

            prediction_loss = obs_prediction_error + rew_prediction_error
            self.optimizer.zero_grad()
            prediction_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters,self.max_grad_norm)
            self.optimizer.step()
            
            mean_obs_prediction_loss += obs_prediction_error.item()
            mean_rew_prediction_loss += rew_prediction_error.item()
        num_updates = self.num_learning_epochs * self.num_mini_batch
        mean_obs_prediction_loss /= num_updates
        mean_rew_prediction_loss /= num_updates
        return mean_obs_prediction_loss,mean_rew_prediction_loss
