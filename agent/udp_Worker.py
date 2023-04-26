
import sys
import os
import gym
import ray
import torch
import numpy as np
import random
import math
import pandas as pd
from envencoder.actor import Actor
import envencoder
from parameter.optm_Params import Parameters
from agent.base_Worker import Base_Worker,Base_RemoteWorkers
from parameter.private_config import  NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL
from parameter.private_config import ENV_DEFAULT_CHANGE
from envencoder.advanced_envencoder.udp_envencoder import UdpEnvencoder
from envs.nonstationary_env import NonstationaryEnv

class Udp_Worker(Base_Worker):
    def __init__(self, parameter: Parameters, env_name='Hopper-v2', seed=0, policy_type=Actor, encoder_type=UdpEnvencoder,env_decoration = NonstationaryEnv, env_tasks=None, non_stationary=False, fix_env_setting=False) -> None:
        super().__init__(parameter, env_name, seed, policy_type, encoder_type, env_decoration, env_tasks, non_stationary, fix_env_setting)
    
    def set_action(self, action, pred_next_state, env_ind, render, need_info):
        next_state, reward, done, info = self.env.step(self.env.denormalization(action))
        self.tracker.update_history(self.state,action,next_state,reward,pred_next_state)
        ## update history tracker
        reward = np.reshape(reward,newshape=(1,))
        if render:
            self.env.render()
        if self.non_stationary:
            self.env_param_vector = self.env.env_parameter_vector
        current_env_step = self.env._elapsed_steps
        self.state = next_state
        self.ep_len += 1
        self.ep_cumrew += reward
        cur_task_ind = self.task_ind
        cur_env_param = self.env_param_vector
        if done:
            self.ep_len_list.append(self.ep_len)
            self.ep_cumrew_list.append(self.ep_cumrew)
            self.ep_rew_list.append(self.ep_cumrew / self.ep_len)
            self.ep_task_id.append(self.task_ind)
            state = self.reset(env_ind)
            self.state = state
        reward,done,cur_task_ind = np.array(reward).reshape(1,),np.array(done).reshape(1,),np.array(cur_task_ind).reshape(1,)
        if need_info:
            return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step, info
        return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step

    def get_action(self, state, policy:Actor, encoder:UdpEnvencoder, deterministic, random, device):
        # state.shape = (dim,)
        with torch.no_grad():
            support = self.tracker.feed_support(device)
            emb = encoder.sample(support.obs, support.act, support.obs2, support.rew)
            if random:
                action = self.env.normalization(self.action_space.sample())
            else:
                action = policy.act(x = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                        emb = emb,deterministic=deterministic).to(torch.device('cpu')).detach().numpy()
            pred_next_state = encoder.world_decoder.sample_transition(obs = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                                            act = torch.from_numpy(action).to(device = device,dtype=torch.float32),
                                                            emb = emb,deterministic=True).detach().to(torch.device('cpu')).numpy()
            pred_reward = encoder.world_decoder.sample_reward(obs = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                                            act = torch.from_numpy(action).to(device = device,dtype=torch.float32),
                                                            emb = emb,deterministic=True).detach().to(torch.device('cpu')).numpy()
        return action,pred_next_state,pred_reward,emb.detach().cpu().numpy()
            
    
    def set_weight(self,policy_state_dict,encoder_state_dict,workd_decoder_state_dict):
        self.policy.load_state_dict(policy_state_dict)
        self.envcoder.encoder.load_state_dict(encoder_state_dict)
        self.envcoder.world_decoder.load_state_dict(workd_decoder_state_dict)


class Udp_Workers(Base_RemoteWorkers):
    def __init__(self, parameter, 
                 env_name, worker_num=2,
                   seed=None, deterministic=False, 
                   use_remote=False, policy_type=Actor, encoder_type=UdpEnvencoder, env_decoration=NonstationaryEnv, env_tasks=None, non_stationary=False):
        super().__init__(parameter, env_name, worker_num, seed, deterministic, use_remote, Udp_Worker, policy_type, encoder_type, env_decoration, env_tasks, non_stationary)
    def eval(self,test_envs:list,num_path,policy = None,envcoder = None):
        cur_policy = policy if policy is not None else self.policy
        cur_envcoder = envcoder if envcoder is not None else  self.envcoder
        task_assignments = [[] for _ in range(len(self.workers))]
        for i in range(len(test_envs)):
            task_assignments[i%len(self.workers)].append(i)

        if self.use_remote:
            ray.get([worker.set_weight.remote(cur_policy.state_dict(),
                cur_envcoder.encoder.state_dict(),
                cur_envcoder.world_decoder.state_dict()) for worker in self.workers])
            tasks = [worker.eval.remote(task_assignments[i],num_path,deterministic=True,device = torch.device('cpu')) for i, worker in enumerate(self.workers)]
            list_res = ray.get(tasks) # list of tuples : log,embs,ids
        else:
            [worker.set_weight(cur_policy.state_dict(),
            cur_envcoder.encoder.state_dict(),
            cur_envcoder.world_decoder.state_dict()) for worker in self.workers]
            list_res = [worker.eval(task_assignments[i],num_path,deterministic=True,device = torch.device('cpu')) for i, worker in enumerate(self.workers)]
        embs,ids = [],[]
        logs = {key: [] for key in list_res[0][0].keys()}
        for res in list_res:
            log,emb,id = res 
            # print(type(res),res)
            for key in logs:
                logs[key] += log[key]
            embs.append(emb),ids.append(id)

        embs = np.concatenate(embs,axis =0)
        ids = np.concatenate(ids,axis =0)
        df = logs
        return df,embs,ids