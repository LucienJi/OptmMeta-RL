import torch 
import numpy as np 

class StackedReplayBuffer:
    class StackedTransition:
        def __init__(self,num_envs,max_history_len,obs_shape,act_shape,device = 'cpu') -> None:
            self.observations = torch.zeros(num_envs,max_history_len,*obs_shape,device = device)
            self.actions = torch.zeros(num_envs,max_history_len,*act_shape,device = device)
            self.rewards = torch.zeros(num_envs,max_history_len,1,device = device)
            # self.dones = torch.zeros(max_history_len,num_envs,1,device = device).byte()
            self.next_observations = torch.zeros(num_envs,max_history_len,*obs_shape,device = device)
        def clear(self):
            self.__init__()
        
        def add_transition(self,obs,act,reward,obs2):
            #! obs.shape = (num_envs,*obs_shape)
            self.observations = torch.roll(self.observations,shifts=-1,dims=1)
            self.observations[:,-1] = obs

            self.actions = torch.roll(self.actions,shifts=-1,dims=1)
            self.actions[:,-1] = act

            self.rewards = torch.roll(self.rewards,shifts=-1,dims=1)
            self.rewards[:,-1] = reward.view(-1,1)

            self.next_observations = torch.roll(self.next_observations,shifts=-1,dims=1)
            self.next_observations[:,-1] = obs2
        
        def reset_idx(self,dones):
            env_ids = dones.nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                return
            self.observations[env_ids] = 0
            self.actions[env_ids] = 0
            self.rewards[env_ids] = 0
            self.next_observations[env_ids] = 0

    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
        
        def clear(self):
            self.__init__()
    
    def __init__(self,capacity,num_envs,obs_shape,act_shape,max_history_len,device = 'cpu') -> None:
        self.device = device
        self.obs_shape = obs_shape
        self.actions_shape = act_shape
        self.max_history_len = max_history_len

        # Core Stacked
        self.stacked_observations = torch.zeros(capacity,num_envs, max_history_len, *obs_shape, device=self.device)
        self.stacked_next_observations = torch.zeros(capacity,num_envs, max_history_len, *obs_shape, device=self.device)
        self.stacked_rewards = torch.zeros(capacity, num_envs,max_history_len, 1, device=self.device)
        self.stacked_actions = torch.zeros(capacity,num_envs, max_history_len, *act_shape, device=self.device)
        self.stacked_dones = torch.zeros(capacity,num_envs, max_history_len, 1, device=self.device).byte()
        # Core Transition
        self.observations = torch.zeros(capacity,num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(capacity,num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(capacity,num_envs, 1, device=self.device)
        self.actions = torch.zeros(capacity,num_envs, *act_shape, device=self.device)
        self.dones = torch.zeros(capacity,num_envs, 1, device=self.device).byte()


        self.capacity = capacity
        self.num_envs = num_envs

        self.step = 0 
        self.size = 0
    
    def add_transitions(self,stacked_transition:StackedTransition,
                        transition:Transition):
        self.stacked_observations[self.step].copy_(stacked_transition.observations)
        self.stacked_next_observations[self.step].copy_(stacked_transition.next_observations)
        self.stacked_rewards[self.step].copy_(stacked_transition.rewards)
        self.stacked_actions[self.step].copy_(stacked_transition.actions)

        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.actions[self.step].copy_(transition.actions)

        self.step = (self.step + 1) % self.capacity
        self.size = min(self.size + 1,self.capacity)
    
    def mini_batch_generator(self,batch_size,num_mini_batches,num_epochs):
        batch_size = min(batch_size,self.size * self.num_envs)
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(self.size, requires_grad=False, device=self.device)[0:mini_batch_size * num_mini_batches]


        #! Stacked History
        stacked_observations = self.stacked_observations[0:self.size].view(-1,self.max_history_len,*self.obs_shape)
        stacked_next_observations = self.stacked_next_observations[0:self.size].view(-1,self.max_history_len,*self.obs_shape)
        stacked_rewards = self.stacked_rewards[0:self.size].view(-1,self.max_history_len,1)
        stacked_actions = self.stacked_actions[0:self.size].view(-1,self.max_history_len,*self.actions_shape)

        #! Transition
        observations = self.observations[0:self.size].view(-1,*self.obs_shape)
        next_observations = self.next_observations[0:self.size].view(-1,*self.obs_shape)
        rewards = self.rewards[0:self.size].view(-1,1)
        actions = self.actions[0:self.size].view(-1,*self.actions_shape)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield stacked_observations[batch_idx],stacked_actions[batch_idx],stacked_rewards[batch_idx],stacked_next_observations[batch_idx],\
                    observations[batch_idx],actions[batch_idx],rewards[batch_idx],next_observations[batch_idx]


