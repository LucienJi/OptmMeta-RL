import sys
import os
import gym
import ray
import torch
import numpy as np
import random
import math
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envencoder.actor import Actor
from envencoder.buffer import Memory
from parameter.optm_Params import Parameters
from log_util.logger import Logger

from parameter.private_config import SKIP_MAX_LEN_DONE, NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL
from parameter.private_config import ENV_DEFAULT_CHANGE
from envs.grid_world_general import RandomGridWorldPlat
import envencoder
from envencoder.envcoder import C_Envcoder
from gym.envs.registration import register
from envs.nonstationary_env import NonstationaryEnv
register(
id='GridWorldPlat-v2', entry_point=RandomGridWorldPlat
)

class Worker:
    def __init__(self,parameter: Parameters, env_name='Hopper-v2', seed=0, policy_type=Actor,encoder_type = C_Envcoder ,
                 env_decoration=NonstationaryEnv, env_tasks=None, non_stationary=False) -> None:
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.fix_env_setting = False
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
        self.tracker = envencoder.EnvTracker(self.obs_dim,self.act_dim,parameter.n_support)
        ### Init ###
        self.ep_len = 0
        self.ep_cumrew = 0
        self.skip_max_len_done = parameter.skip_max_len_done
        self.ep_len_list = []
        self.ep_cumrew_list = []
        self.ep_rew_list = []
        self.ep_task_id = []
        self.state = self.reset()
    
    ### Env Hander: set env parameter, reset, step 
    def set_global_seed(self, seed):
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def reset(self, env_ind=None,ns_env_list = None,ns_period = None):
        ### Attention !!!! history should be reset too!!!!
        self.change_env_param(env_ind,ns_env_list,ns_period) ### after calling change env param, the self.task_ind will be new task idx 
        state = self.env.reset()
        # print("Check: ",self.env.setted_env_changing_period)
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
                print("Set NS")
                another_task = random.randint(0, len(self.env_tasks) - 1)
                env_param_list = [self.env_tasks[self.task_ind]] + [self.env_tasks[random.randint(0, len(self.env_tasks)-1)] for _ in range(15)]
                self.env.set_nonstationary_para(env_param_list,
                                                NON_STATIONARY_PERIOD, NON_STATIONARY_INTERVAL)
    
    def set_action(self, action:np.ndarray,pred_next_state = None,env_ind=None, render=False, need_info=False):
        next_state, reward, done, info = self.env.step(self.env.denormalization(action))
        ## update history tracker
        reward = np.reshape(reward,newshape=(1,))
        self.tracker.update_history(self.state,action,next_state,reward,pred_next_state)
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

    def get_action(self,state,cur_policy:Actor,cur_encoder:envencoder.C_Envcoder,deterministic,random,with_moco,device =torch.device('cpu')):
        if with_moco:
            emb = cur_encoder.moco.sample(self.task_ind,device,None)
        else:
            support = self.tracker.feed_support(device)
            emb = cur_encoder.sample(support.obs,support.act,support.obs2,support.rew,method='mean')
        if random:
            action = self.env.normalization(self.action_space.sample())
        else:
            action = cur_policy.act(x = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                    emb = emb,deterministic=deterministic).to(torch.device('cpu')).detach().numpy()
        
        pred_next_state = cur_encoder.transition.sample(obs = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                                        act = torch.from_numpy(action).to(device = device,dtype=torch.float32),
                                                        emb = emb,deterministic=True).detach().to(torch.device('cpu')).numpy()
        pred_reward = cur_encoder.reward.sample(obs = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                                        act = torch.from_numpy(action).to(device = device,dtype=torch.float32),
                                                        emb = emb,deterministic=True).detach().to(torch.device('cpu')).numpy()
        return action,pred_next_state,pred_reward,emb.detach().cpu().numpy()
    
    def raw_emb_collection(self,num_steps,device = 'cpu',log_scale = 3.0,n_tasks = 10,linear = True):
        self.env.reset_nonstationary()
        tasks,ns_tasks = self.env.resample_tasks(log_scale,n_tasks,linear = linear,ns_sequence = None)
        self.env_tasks = tasks
        list_embs,list_c,list_paras = [],[],[]
        l_s,l_a,l_s2,l_r = [],[],[],[]

        for i in range(n_tasks):
            embs,cs,para = [],[],[]
            s,a,s2,r = [],[],[],[]
            self.ns_tracker.reset()
            emb = self.ns_tracker.feed(device)
            state = self.reset(i)
            for _ in range(num_steps):
                with torch.no_grad():
                    action = self.policy.act(x = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                        emb = emb,deterministic=True).to(torch.device('cpu')).detach().numpy()
                    next_state,reward,done,_ = self.env.step(self.env.denormalization(action))
                    s.append(state),a.append(action),s2.append(next_state),r.append(reward)
                    emb,c = self.envcoder.encoder.forward(torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                                        torch.from_numpy(action).to(device= device,dtype=torch.float32),
                                                        torch.from_numpy(next_state).to(device = device,dtype=torch.float32))
                    para.append(self.env.env_parameter_vector)
                embs.append(emb.numpy())
                cs.append(c.numpy())
                state = next_state
                if done:
                    state = self.reset(i)
                self.state  = state
            list_embs.append(np.array(embs))
            list_c.append(np.array(cs))
            list_paras.append(np.array(para))
            l_s.append(np.array(s)),l_a.append(np.array(a)),l_s2.append(np.array(s2)),l_r.append(np.array(r))
        trans = {
            's':np.array(l_s),'a':np.array(l_a),'s2':np.array(l_s2),'r':np.array(l_r)
        }
        return np.array(list_embs),np.array(list_c),np.array(list_paras),trans


    def ns_test(self,num_steps,device = 'cpu',log_scale = 3.0,n_tasks = 10 ,ns_sequence = None ,ns_period = 100):
        tasks,ns_tasks = self.env.resample_tasks(log_scale,n_tasks,linear = True,ns_sequence = ns_sequence)
        
        real_param = []
        embs,raw_embs,cs,Gs = [],[],[],[]
        ss,Ss = [],[]
        update_flag,delays = [],[]
        state = self.reset(None,ns_tasks,ns_period=ns_period)
        self.ns_tracker.reset()

        for _ in range(num_steps):
            emb = self.ns_tracker.feed(device)
            embs.append(emb.numpy())
            with torch.no_grad():
                action = self.policy.act(x = torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                        emb = emb,deterministic=True).to(torch.device('cpu')).detach().numpy()
            next_state,reward,done,_ = self.env.step(self.env.denormalization(action))

            detect_flag,delay = self.ns_tracker.update(obs=torch.from_numpy(state).to(device = device,dtype=torch.float32),
                                    act = torch.from_numpy(action).to(device = device,dtype=torch.float32),
                                   obs2=torch.from_numpy(next_state).to(device = device,dtype=torch.float32),
                                    encoder=self.envcoder.encoder,transition=self.envcoder.transition,policy=self.policy)

            raw_emb,s,S,G,c = self.ns_tracker.debug()
            raw_embs.append(raw_emb)
            ss.append(s)
            Ss.append(S)
            Gs.append(G) 
            cs.append(c)
            update_flag.append(detect_flag)
            delays.append(delay)
            

            cur_env_param = self.env.env_parameter_vector
            real_param.append(cur_env_param)
            self.state = state
            if done:
                state = self.reset()
                self.state = state 
            state = next_state
        embs = np.array(embs)
        real_param = np.array(real_param)
        change_inds = np.where(np.abs(np.diff(real_param[:, -1])) > 0)[0] + 1

        log = {
            'raw_embs':np.array(raw_embs),
            'Confidence':np.array(cs),
            's':np.array(ss),
            'S':np.array(Ss),
            'G':np.array(Gs),
            'detect_flag':np.array(update_flag),
            'delays':np.array(delays)
        }
        
        
        return embs,change_inds,real_param,log
    
    
    ### EnvEncoder And Policy Interact with Env
    
    def sample(self, min_batch, deterministic=False, env_ind=None,device = torch.device("cpu")):
        ## Use Worker's policy and encoder to interact with env
        step_count = 0
        list_mem = []
        log = {'EpRet': [],
               'EpMeanRew': [],
               'EpLen': [],
               'EpMeanError':[],
               'Env id':[]}
        while step_count < min_batch:
            mem = Memory()
            state = self.reset(env_ind)
            # print("Env Task Id: ",self.task_ind)
            err = 0.0
            while True:
                action ,pred_next_state,pred_reward,emb = self.get_action(state,self.policy,self.envcoder,deterministic=deterministic,
                            random = False,with_moco = False,device = device)
                next_state, reward, done, _ = self.env.step(self.env.denormalization(action))
                self.tracker.update_history(state,action,next_state,pred_next_state)
                if self.non_stationary:
                    self.env_param_vector = self.env.env_parameter_vector
                mem.push(state,action,next_state,np.array(reward).reshape(1,),np.array(done).reshape(1,),np.array(self.task_ind).reshape(1,),self.env_param_vector)
                err += ((pred_next_state - next_state)**2).mean()
                self.ep_cumrew += reward
                self.ep_len += 1
                step_count += 1
                if done:
                    list_mem.append(mem)
                    log['EpMeanRew'].append(self.ep_cumrew / self.ep_len)
                    log['EpLen'].append(self.ep_len)
                    log['EpRet'].append(self.ep_cumrew)
                    log['EpMeanError'].append(err / self.ep_len)
                    log['Env id'].append(self.task_ind)
                    break
                state = next_state
        return list_mem,log
    
    def eval(self,env_inds:list,num_path,deterministic=False,device = torch.device('cpu')):
        log = {'EpRet': [],
               'EpMeanRew': [],
               'EpLen': [],
               'EpMeanError':[],
               'Env id':[]}
        embs = []
        ids = []
        for env_id in env_inds:
            # print("Env IDs: ",env_id)
            for iter in range(num_path):
                # print("Iter::",iter)
                state = self.reset(env_id)
                err = 0.0
                while True:
                    action ,pred_next_state,pred_reward,emb = self.get_action(state,self.policy,self.envcoder,deterministic=deterministic,
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
        action,pred_next_state,pred_reward,_ = self.get_action(state,cur_policy,cur_encoder,deterministic,random,with_moco,device)
        next_state, reward, done, _, cur_task_ind, cur_env_param, current_env_step = self.set_action(action,pred_next_state,env_ind,False,False)
        if self.skip_max_len_done and done and current_env_step >= self.env._max_episode_steps:
            done = np.array([0]).reshape(1,)
        mem.push(state,action,next_state,reward,done,cur_task_ind,cur_env_param)
        return mem

    def get_current_state(self):
        return self.state
    
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

    def set_weight(self,policy_state_dict,encoder_state_dict,transition_state_dict,reward_state_dict):
        self.policy.load_state_dict(policy_state_dict)
        self.envcoder.encoder.load_state_dict(encoder_state_dict)
        self.envcoder.transition.load_state_dict(transition_state_dict)
        self.envcoder.reward.load_state_dict(reward_state_dict)


class EnvRemoteWorkers:
    def __init__(self,parameter, env_name, worker_num=2, seed=None,
                 deterministic=False, use_remote=True,
                 policy_type=Actor, encoder_type = None, env_decoration=None,
                 env_tasks=None, non_stationary=False):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.emb_dim = parameter.emb_dim
        self.action_space = self.env.action_space
        self.set_seed(seed)
        self.non_stationary = non_stationary
        self.env_tasks = env_tasks
        RemoteEnvWorker = ray.remote(Worker) if use_remote else Worker
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
        self.use_remote = use_remote
        self.total_steps = 0
    def set_seed(self, seed):
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.env.seed(seed)
    
    ### For test 
    def submit_task(self,min_batch,policy = None,envcoder = None,env_ind = None,device=torch.device('cpu')):
        ## use trained policy and envcoder to interact with test env
        cur_policy = policy if policy is not None else self.policy
        cur_envcoder = envcoder if envcoder is not None else  self.envcoder
        min_batch_per_worker = min_batch // self.worker_num + 1 
        print("Time to Set Weight:  ", cur_policy.device,cur_envcoder.encoder.device)
        if self.use_remote:
            # ray.get([worker.set_weight.remote(cur_policy,cur_envcoder) for worker in self.workers])
            ray.get([worker.set_weight.remote(cur_policy.state_dict(),cur_envcoder.encoder.state_dict(),cur_envcoder.transition.state_dict(),cur_envcoder.reward.state_dict()) for worker in self.workers])
            tasks = [worker.sample.remote(min_batch_per_worker,self.deterministic,env_ind,device) for worker in self.workers]
            return tasks
        else:
            # [worker.set_weight(cur_policy,cur_envcoder) for worker in self.workers]
            [worker.set_weight(cur_policy.state_dict(),cur_envcoder.encoder.state_dict(),cur_envcoder.transition.state_dict(),cur_envcoder.reward.state_dict()) for worker in self.workers]
            tasks = [worker.sample(min_batch_per_worker,self.deterministic,env_ind,device) for worker in self.workers]
            return tasks
    def query_task(self,tasks,need_memory):
        if self.use_remote:
            res = ray.get(tasks)
            
        else:
            res = tasks
        list_mem = []
        [list_mem.append(mem) for mem,_ in res ]
        logs = {key:[] for key in res[0][1]}
        for key in logs:
            for _ , item in res:
                logs[key] += item[key]
        total_steps = 0 
        for mem in list_mem:
            total_steps += len(mem)
        logs['Total Steps'] = total_steps
        if need_memory:
            return logs,list_mem
        else:
            return logs
    
    ### RemoteAgent sample 
    def eval(self,test_envs:list,num_path,policy = None,envcoder = None):
        cur_policy = policy if policy is not None else self.policy
        cur_envcoder = envcoder if envcoder is not None else  self.envcoder
        task_assignments = [[] for _ in range(len(self.workers))]
        for i in range(len(test_envs)):
            task_assignments[i%len(self.workers)].append(i)
        if self.use_remote:
            ray.get([worker.set_weight.remote(cur_policy.state_dict(),cur_envcoder.encoder.state_dict(),cur_envcoder.transition.state_dict(),cur_envcoder.reward.state_dict()) for worker in self.workers])
            tasks = [worker.eval.remote(task_assignments[i],num_path,deterministic=True,device = torch.device('cpu')) for i, worker in enumerate(self.workers)]
            list_res = ray.get(tasks) # list of tuples : log,embs,ids
        else:
            [worker.set_weight(cur_policy.state_dict(),cur_envcoder.encoder.state_dict(),cur_envcoder.transition.state_dict(),cur_envcoder.reward.state_dict()) for worker in self.workers]
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
    ### For Embedding Analysis
    def step_local_1env(self,env_ind=None,
                         policy = None, encoder = None,device = torch.device('cpu')):
        mem = Memory()
        cur_policy = policy 
        cur_encoder = encoder 
        worker = self.workers[0]     
        assert not self.use_remote  
        state = worker.get_current_state()
        action,pred_next_state,pred_reward,emb = worker.get_action(state,cur_policy,cur_encoder,deterministic=True,random=False,with_moco=False,device = device)
        next_state, reward, done, _, cur_task_ind, cur_env_param, current_env_step,info = worker.set_action(action,pred_next_state,env_ind,False,need_info=True)
        err = ((pred_next_state - next_state)**2).mean()

        # print("Ind Check: ",cur_task_ind)
        mem.push(state,action,next_state,reward,done,cur_task_ind,cur_env_param)
        return mem,info, dict(
            emb = emb,
            err = err
        )
    
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
        

if __name__ == '__main__':
    from envs.nonstationary_env import NonstationaryEnv
    env_name = 'Hopper-v2'

    ray.init()
    logger = Logger(parameter=Parameters())
    parameter = logger.parameter
    env = NonstationaryEnv(gym.make(env_name), rand_params=parameter.varying_params)
    remote_workers = EnvRemoteWorkers(parameter,env_name,2,0,False,use_remote=True,env_decoration=NonstationaryEnv,env_tasks=env.sample_tasks(10),non_stationary=False)
    list_mem,logs = remote_workers.sample(1000)
    
