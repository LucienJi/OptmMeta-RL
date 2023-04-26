
import sys
import os
import gym
import ray
import torch
import numpy as np
import random
import math
import pandas as pd
from abc import ABC, abstractmethod
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from envencoder.actor import Actor
from envencoder.buffer import Memory
from parameter.optm_Params import Parameters
from log_util.logger import Logger

from parameter.private_config import  NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL
from parameter.private_config import ENV_DEFAULT_CHANGE
from envs.nonstationary_env import NonstationaryEnv
from envencoder.history_tracker import EnvTracker

class Base_Worker(object):
    def __init__(self,parameter: Parameters, env_name='Hopper-v2', seed=0, policy_type =Actor,encoder_type = None,
                 env_decoration=NonstationaryEnv, env_tasks=None, non_stationary=False,fix_env_setting=False) -> None:
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.fix_env_setting = fix_env_setting
        self.set_global_seed(seed)
        self.non_stationary = non_stationary
        
        if env_decoration is not None:
            default_change_range = ENV_DEFAULT_CHANGE if not hasattr(parameter, 'env_default_change_range') \
                else parameter.env_default_change_range
            if not hasattr(parameter, 'env_default_change_range'):
                print('[WARN]: env_default_change_range does not appears in parameter!')
            self.env = env_decoration(self.env, log_scale_limit=default_change_range,
                                    rand_params=parameter.varying_params)
        self.observation_space = self.env.observation_space
        self.env_tasks = None
        self.task_ind = -1
        self.env.reset()
        if env_tasks is not None and isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env_tasks = env_tasks
            self.task_ind = random.randint(0, len(self.env_tasks) - 1)
            self.env.set_task(self.env_tasks[self.task_ind])

        self.obs_dim=self.observation_space.shape[0]
        self.act_dim=self.action_space.shape[0]
        self.emb_dim=parameter.emb_dim

        ### Set Model ### 
        policy_config = policy_type.make_config_from_param(parameter)
        self.policy = policy_type(self.obs_dim,self.act_dim,**policy_config)
        self.envcoder = encoder_type(self.obs_dim,self.act_dim,self.emb_dim,parameter)
        self.tracker = EnvTracker(self.obs_dim,self.act_dim,parameter.n_support)
        ### Init ###
        self.ep_len = 0
        self.ep_cumrew = 0
        self.skip_max_len_done = parameter.skip_max_len_done
        self.ep_len_list = []
        self.ep_cumrew_list = []
        self.ep_rew_list = []
        self.ep_task_id = []
        self.state = self.reset()

    def set_global_seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def reset(self, env_ind=None,ns_env_list = None,ns_period = None):
        self.change_env_param(env_ind,ns_env_list,ns_period) ### after calling change env param, the self.task_ind will be new task idx 
        state = self.env.reset()
        self.tracker.init()
        self.env_param_vector = self.env.env_parameter_vector
        self.ep_len = 0
        self.ep_cumrew = 0
        return state

    def change_env_param(self, set_env_ind=None,ns_env_list=None,ns_period = None ):
        if self.fix_env_setting:
            self.env.set_task(self.env_tasks[self.task_ind])
            return
        if ns_env_list is not None:
            self.env.set_task(ns_env_list[0])
            self.task_ind = -1 ### may have error
            ns_period = NON_STATIONARY_PERIOD if ns_period is None else ns_period
            self.env.set_nonstationary_para(ns_env_list,ns_period, NON_STATIONARY_INTERVAL)
            return 

        if self.env_tasks is not None and len(self.env_tasks) > 0:
            self.task_ind = random.randint(0, len(self.env_tasks) - 1) if set_env_ind is None or set_env_ind >= \
                                                                          len(self.env_tasks) else set_env_ind
            self.env.set_task(self.env_tasks[self.task_ind])
            if self.non_stationary:
                another_task = random.randint(0, len(self.env_tasks) - 1)
                env_param_list = [self.env_tasks[self.task_ind]] + [self.env_tasks[random.randint(0, len(self.env_tasks)-1)] for _ in range(15)]
                self.env.set_nonstationary_para(env_param_list,
                                                NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL)
    
    @abstractmethod
    def get_action(self,state,policy,encoder,deterministic,random,with_moco,device):
        pass 
    @abstractmethod
    def set_action(self,action,pred_next_state,env_ind,render,need_info):
        pass 

    @abstractmethod
    def set_weight(self):
        pass 

    def get_current_state(self):
        return self.state
    def eval(self,env_inds:list,num_path,deterministic=False,device = torch.device('cpu')):
        log = {'EpRet': [],
               'EpMeanRew': [],
               'EpLen': [],
               'EpMeanError':[],
               'Env id':[]}
        embs = []
        ids = []
        for env_id in env_inds:
            for iter in range(num_path):
                state = self.reset(env_id)
                err = 0.0
                while True:
                    action ,pred_next_state,pred_reward,emb = self.get_action(
                            state,self.policy,self.envcoder,deterministic=deterministic,
                            random = False,with_moco = False,device = device)
                    next_state, reward, done, _ = self.env.step(self.env.denormalization(action))
                    reward = np.reshape(reward,newshape=(1,))
                    self.tracker.update_history(state,action,next_state,reward,pred_next_state)
                    if self.non_stationary:
                        self.env_param_vector = self.env.env_parameter_vector
                    err += ((pred_next_state - next_state)**2).mean()
                    self.ep_cumrew += reward
                    self.ep_len += 1
                    embs.append(emb)
                    ids.append(env_id)
                    if done:
                        log['EpMeanRew'].append(self.ep_cumrew / self.ep_len)
                        log['EpLen'].append(self.ep_len)
                        log['EpRet'].append(self.ep_cumrew)
                        log['EpMeanError'].append(err / self.ep_len)
                        assert env_id  == self.task_ind
                        log['Env id'].append(self.task_ind)
                        # print("IS Done: ",done, iter)
                        break
                    ### Important !!! Do not forget it
                    state = next_state 
        embs = np.stack(embs,axis = 0)
        ids = np.array(ids)
        self.collect_result()
        return log,embs,ids
    def step_locally(self,random = False,with_moco = True,
                         deterministic = False,env_ind=None,
                         policy = None, encoder = None,
                         device = torch.device('cpu')):
        cur_policy = policy if policy is not None else self.policy
        cur_encoder = encoder if encoder is not None else self.envcoder
        mem = Memory()
        state = self.get_current_state()
        action,pred_next_state,pred_reward,_ = self.get_action(state,cur_policy,cur_encoder,
                                        deterministic,random,with_moco,device)
        next_state, reward, done, _, cur_task_ind, cur_env_param, current_env_step = self.set_action(
                action,pred_next_state,env_ind,render = False,need_info = False)
        if self.skip_max_len_done and done and current_env_step >= self.env._max_episode_steps:
            done = np.array([0]).reshape(1,)
        mem.push(state,action,next_state,reward,done,cur_task_ind,cur_env_param)
        return mem

    def step_local_1env(self,env_ind,policy,encoder,device = torch.device('cpu')):
        cur_policy = policy
        cur_encoder = encoder 
        mem = Memory()
        state = self.get_current_state()
        action,pred_next_state,pred_reward,emb = self.get_action(state,cur_policy,cur_encoder,deterministic=True,random=False,with_moco=False,device = device)
        next_state, reward, done, _, cur_task_ind, cur_env_param, current_env_step,info = self.set_action(action,pred_next_state,env_ind,False,need_info=True)
        err_s = ((pred_next_state - next_state)**2).mean()
        err_r = ((pred_reward - reward)**2).mean()
        # print("Ind Check: ",cur_task_ind)
        mem.push(state,action,next_state,reward,done,cur_task_ind,cur_env_param)
        return mem,info, dict( emb = emb, err_s = err_s,err_r = err_r)

    def ns_eval(self,policy,encoder,device,num_steps,env_ind):
        embs = []
        real_param = []
        actions = []
        rews = []
        action_discrepancy = []
        keep_at_target = []
        errs = []
        done = False 
        while not done:
            mem,env_info ,info = self.step_local_1env(env_ind,policy,encoder,device)
            done = mem.memory[0].done[0]
        for _ in range(num_steps):
            mem,env_info ,info = self.step_local_1env(env_ind,policy,encoder,device)
            real_param.append(mem.memory[0].env_param)
            embs.append(info['emb'])
            actions.append(mem.memory[0].act)
            rews.append(mem.memory[0].rew)
            errs.append(info['err_s'])
            if isinstance(env_info, dict) and 'action_discrepancy' in env_info and env_info['action_discrepancy'] is not None:
                action_discrepancy.append(np.array([env_info['action_discrepancy'][0],
                                                    env_info['action_discrepancy'][1]]))
                keep_at_target.append(1 if env_info['keep_at_target'] else 0)
        self.collect_result()
        embs = np.array(embs)
        real_param = np.array(real_param)
        rews = np.array(rews)
        errs = np.array(errs).reshape(-1)
        # print(pred_errors.shape)
        change_inds = np.where(np.abs(np.diff(real_param[:, -1])) > 0)[0] + 1
        diff_from_expert = 0
        at_target_ratio = 0
        if len(action_discrepancy) > 0:
            fig = utils.action_discrepancy(embs,real_param,action_discrepancy,change_inds)
            action_discrepancy = np.array(action_discrepancy)
            abs_res = np.abs(action_discrepancy[:, 0]) / 3 + np.abs(action_discrepancy[:, 1]) / 3
            diff_from_expert = np.mean(abs_res)
            at_target_ratio = np.mean(keep_at_target)
        else:
            fig = utils.action_discrepancy(embs,real_param,None,change_inds)

        df = {
            'rew':rews.reshape(-1),'err':errs,'real_para':real_param[:,-1].reshape(-1),
        }
        return fig,df,embs,change_inds,(diff_from_expert, at_target_ratio)


    def collect_result(self):
        ep_len_list = self.ep_len_list
        self.ep_len_list = []
        ep_cumrew_list = self.ep_cumrew_list
        self.ep_cumrew_list = []
        ep_rew_list = self.ep_rew_list
        self.ep_rew_list = []
        ep_task_id = self.ep_task_id
        self.ep_task_id = []
        log = {
        'EpMeanRew': ep_rew_list,
        'EpLen': ep_len_list,
        'EpRet': ep_cumrew_list,
        'task id':ep_task_id
        }
        return log

    

class Base_RemoteWorkers:
    def __init__(self,parameter, env_name, worker_num=2, seed=None,
                 deterministic=False, use_remote=False,worker_type=Base_Worker,
                 policy_type=Actor, encoder_type = None, env_decoration=None,
                 env_tasks=None, non_stationary=False):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.emb_dim = parameter.emb_dim
        self.action_space = self.env.action_space
        self.use_remote = use_remote
        self.non_stationary = non_stationary
        self.env_tasks = env_tasks
        RemoteEnvWorker = ray.remote(worker_type) if use_remote else worker_type
        if use_remote:
            self.workers = [RemoteEnvWorker.remote(parameter, env_name, random.randint(0, 10000), 
                                            policy_type,encoder_type, env_decoration, env_tasks, non_stationary) for _ in range(worker_num)]
        else:
            self.workers = [RemoteEnvWorker(parameter, env_name, random.randint(0, 10000),
                                            policy_type,encoder_type, env_decoration, env_tasks,non_stationary) for _ in range(worker_num)]
        if env_decoration is not None:
            default_change_range = ENV_DEFAULT_CHANGE if not hasattr(parameter, 'env_default_change_range') \
                else parameter.env_default_change_range
            if not hasattr(parameter, 'env_default_change_range'):
                print('[WARN]: env_default_change_range does not appears in parameter!')
            self.env = env_decoration(self.env, log_scale_limit=default_change_range,
                                    rand_params=parameter.varying_params)

        ### Set Model ### 
        policy_config = policy_type.make_config_from_param(parameter)
        self.policy = policy_type(self.obs_dim,self.act_dim,**policy_config)
        self.envcoder = encoder_type(self.obs_dim,self.act_dim,self.emb_dim,parameter)
        self.worker_num = worker_num
        self.env_name = env_name
        self.env.reset()
        if isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env.set_task(random.choice(env_tasks))
        self.env_parameter_len = self.env.env_parameter_length
        self.deterministic = deterministic
        self.total_steps = 0
        self.set_seed(seed)

    def set_seed(self, seed):
        if seed is None:
            return
        if self.use_remote:
            ray.get([worker.set_global_seed.remote(seed+i) for i,worker in enumerate(self.workers)])
        else:
            [worker.set_global_seed(seed+i) for i,worker in enumerate(self.workers)]
    
    ### Locally Interact with Environment 
    def step_local(self,random = False,with_moco = True,
                         deterministic = False,env_ind=None,
                         policy = None, encoder = None,device = torch.device('cpu')):
        assert not self.use_remote
        list_mem = [worker.step_locally(random = random,with_moco = with_moco,
                         deterministic = deterministic,env_ind=env_ind,
                         policy = policy, encoder = encoder,
                         device = device) for worker in self.workers]
    
        return list_mem
    
    
    


    def collect_results(self,total_step):
        assert not self.use_remote
        list_logs = [worker.collect_result() for worker in self.workers]
        logs = {key:[] for key in list_logs[0].keys()}
        for log in list_logs:
            for k in logs.keys():
                logs[k]+= log[k]
        k = list(logs.keys())[0]
        l = len(logs[k])
        logs['global step'] = [total_step for _ in range(l)]
        return logs