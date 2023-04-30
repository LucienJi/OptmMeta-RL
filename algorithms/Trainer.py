import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from parameter.private_config import *
from agent.Worker import EnvRemoteWorkers
from envs.nonstationary_env import NonstationaryEnv
from envencoder.buffer import MetaBuffer
import gym
import torch
import numpy as np
import random
import utils
from utils.torch_utils import to_device
from utils.math import dump_info
from envencoder.actor import Actor,QNetwork
from parameter.optm_Params import Parameters
from envencoder.sac import SAC
import envencoder
import torch.nn as nn
import seaborn as sns
import pandas as pd 

class SimpleTrainer:
    def __init__(self,parameter:Parameters = None,log_dir = None ) -> None:
        if parameter is not None:
            self.logger = Logger(parameter=parameter,base_dir=log_dir)
        else:
            self.logger = Logger(parameter=Parameters(),base_dir=log_dir)
        # self.logger.set_tb_x_label('TotalInteraction')
        # self.timer = Timer()
        self.parameter = self.logger.parameter
        self.policy_type = Actor
        self.encoder_type = envencoder.C_Envcoder

        self.policy_config = Actor.make_config_from_param(self.parameter)
        self.value_config = QNetwork.make_config_from_param(self.parameter)
        self.env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)
        self.ood_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_ood_change_range,
                                        rand_params=self.parameter.varying_params)
        self.global_seed(np.random, random, self.env, self.ood_env, seed=self.parameter.seed)
        torch.manual_seed(seed=self.parameter.seed)
        self.env_tasks = self.env.sample_tasks(self.parameter.task_num,linspace=False) ## From small to big
        self.test_tasks = self.env.sample_tasks(self.parameter.test_task_num,linspace=False) ## same range, but different test
        self.ood_tasks = self.ood_env.sample_tasks(self.parameter.test_task_num,linspace=False) ## OOD 

        self.training_agent = EnvRemoteWorkers(parameter=self.parameter, env_name=self.parameter.env_name,
                                             worker_num=1, seed=self.parameter.seed,
                                             deterministic=False, use_remote=False, policy_type=Actor,encoder_type=self.encoder_type,
                                             env_decoration=NonstationaryEnv,
                                             env_tasks=self.env_tasks,
                                             non_stationary=False)

        ## Stationary
        self.s_test_agent = EnvRemoteWorkers(parameter=self.parameter, env_name=self.parameter.env_name,
                                         worker_num=self.parameter.num_threads, seed=self.parameter.seed + 1,
                                         deterministic=True, use_remote=self.parameter.use_remote, policy_type=Actor,encoder_type=self.encoder_type,
                                         env_decoration=NonstationaryEnv, 
                                         env_tasks=self.ood_tasks,  #### Here, in or odd 
                                         non_stationary=False)

        self.ns_test_agent = EnvRemoteWorkers(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=1, seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=False, policy_type=self.policy_type,encoder_type=self.encoder_type,
                                                env_decoration=NonstationaryEnv, env_tasks=self.ood_tasks,non_stationary=True)

        self.obs_dim = self.training_agent.obs_dim
        self.act_dim = self.training_agent.act_dim

        self.sac = SAC(self.obs_dim,self.act_dim,self.parameter)
        self.encoder = self.encoder_type(self.obs_dim,self.act_dim,self.parameter.emb_dim,self.parameter)
        
        ## Buffer 
        self.replay_buffer = MetaBuffer(max_traj_num=100,max_traj_step=1000)
        self.device = self.parameter.device
   
    def update_encoder(self,task_indices):
        bz = self.parameter.encoder_batch_size
        meta_loss = 0
        info ={}
        # list_embs = []
        for id in task_indices:
            support,to_Predict = self.replay_buffer.sample_support(id,bz = bz,n_support=1,M_to_predict=5,with_tensor=True,device=self.device)
            encoder_loss,encoder_info,embs = self.encoder.compute_encoder_loss(support.obs,support.act,support.obs2,support.rew,
                                                                               to_Predict.obs,to_Predict.act,to_Predict.obs2,to_Predict.rew,id )
            meta_loss += encoder_loss
            info = dump_info(info,encoder_info)
            self.encoder.moco.add_and_update(id,embs)
            # list_embs.append(embs)
        # list_embs = torch.stack(list_embs,dim = 0) # (n_task,bz,dim)
        meta_loss = meta_loss / len(task_indices)
        loss = meta_loss 
        self.encoder.optm.zero_grad()
        loss.backward()
        self.encoder.optm.step()
        info = {k:np.mean(v) for k,v in info.items()}
        return info,None
        # return info,list_embs.detach()

    
    def update_sac_multitask(self,task_indices_train,with_embs = None):
         ## update value 
        info = {}
        bz ,n_task_per_batch = self.parameter.sac_mini_batch_size,self.parameter.task_per_batch
        task_indices = self.replay_buffer.sample_task_id(n_task_per_batch) if task_indices_train is None else task_indices_train
        data = self.replay_buffer.sample_multi_task_batch(task_indices,bz,True,self.device) # (n_env,bz,dim)
        stable_emb = self.encoder.moco.get_multitask_value_emb(task_indices,bz,self.device,deterministic=False) ## stable emb
        

        if with_embs is not None:
            if with_embs.ndim == 3:# (n_env,n_support,dim)
                with_embs = with_embs.mean(1).unsqueeze(1).expand((-1,bz,-1))
                emb = with_embs

        q_loss ,q_info = self.sac.compute_q_loss(data.obs,data.act,data.obs2,data.rew,data.done,stable_emb,stable_emb)
        self.sac.value_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.sac.value_parameters,self.parameter.max_grad_norm)
        self.sac.value_optimizer.step()

        self.sac.policy_optimizer.zero_grad()
        actor_loss,actor_info = self.sac.compute_policy_loss(data.obs,stable_emb)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.sac.policy_parameters,self.parameter.max_grad_norm)
        self.sac.policy_optimizer.step()

        self.sac.alpha_optimizer.zero_grad()
        alpha_loss,alpha_info = self.sac.compute_alpha_loss(data.obs,stable_emb)
        alpha_loss.backward()
        self.sac.alpha_optimizer.step()

        info = dump_info(info,q_info)
        info = dump_info(info,actor_info)
        info = dump_info(info,alpha_info)
        self.sac.update_value_function()

        return info,task_indices_train

    def update(self,with_encoder):
        info = {}
        for _ in range(self.parameter.inner_iter_num):
            indices = self.replay_buffer.sample_task_id(self.parameter.task_per_batch)
            if with_encoder:
                encoder_info,_ = self.update_encoder(indices)
                info = dump_info(info,encoder_info)
            sac_info,indices = self.update_sac_multitask(indices,None) 
            info = dump_info(info,sac_info)
        return {k:np.mean(v) for k,v in info.items()}

    @staticmethod
    def global_seed(*args, seed):
        for item in args:
            item.seed(seed)
    def learn(self):
        total_steps = 0
        # tracker = TrainingTracker(self.logger.output_dir,'traning_info')
        if self.replay_buffer.size < self.parameter.start_train_num:
            self.encoder.to(device=torch.device('cpu'))
            self.sac.policy.to(device=torch.device('cpu'))
            self.logger(f"init samples!!!")
            while total_steps <= self.parameter.start_train_num:
                list_mem = self.training_agent.step_local(total_steps < self.parameter.random_num,with_moco = True,
                        deterministic=False,env_ind=None,policy = self.sac.policy,encoder = self.encoder,device = torch.device('cpu'))
                for mem in list_mem:
                    self.replay_buffer.push_mem(mem)
                    total_steps += 1
            self.logger("init done!!!")
        # tracker.writekvlist(self.training_agent.collect_results(total_steps))

        for iter in range(self.parameter.max_iter_num):
            single_step_iterater = self.parameter.min_batch_size
            step = 0
            self.sac.policy.to(self.device)
            self.encoder.to(self.device)
            while step < single_step_iterater:
                list_mem = self.training_agent.step_local(random = False,with_moco=True,deterministic=False,env_ind=None,policy = self.sac.policy,encoder = self.encoder,device = self.device)
                for mem in list_mem:
                    self.replay_buffer.push_mem(mem)
                    total_steps += 1
                    step += 1
                
                if ((step % self.parameter.update_sac_interval) == 0):
                    update_encoder = (step % self.parameter.update_encoder_interval == 0)
                    update_info = self.update(update_encoder)
                    self.logger.add_tabular_data(tb_prefix='training',**update_info)
            training_info = self.training_agent.collect_results(total_steps)
            # tracker.writekvlist(training_info)
            self.logger.add_tabular_data(tb_prefix='training',**training_info)
            self.logger.log_tabular("Global Step",total_steps,tb_prefix = 'timestep',average_only=True)

            if iter % 10 == 0:
                print("Start Testing")
                self.sac.policy.to(torch.device('cpu'))
                self.encoder.to(torch.device('cpu'))
                id_s_log,fig,id_ns_fig,id_ns_fig2,id_ns_fig3 = self.indistribution_ns_test(self.s_test_agent,self.ns_test_agent,
                                                                                num_paths=2,num_ns_steps=self.parameter.ns_test_steps,save_test_results=False)
                self.logger.add_tabular_data(tb_prefix='evaluation',**self.append_key(id_s_log,"Test"))
                self.logger.tb.add_figure('ns_test/embedding_behavior',fig,global_step= total_steps)
                self.logger.tb.add_figure('ns_test/Embedding_Similarity',id_ns_fig,global_step= total_steps)
                self.logger.tb.add_figure('ns_test/Prediction_Error',id_ns_fig2,global_step= total_steps)
                self.logger.tb.add_figure('ns_test/Embedding_Variation',id_ns_fig3,global_step= total_steps)

            if iter % 10 == 0:
                self.save()
            self.logger.dump_tabular(average_only=True)
        
        # tracker.close()
        self.save()
        self.logger.finish()
    
    def indistribution_ns_test(self,id_s_agent:EnvRemoteWorkers,id_ns_agent:EnvRemoteWorkers,num_paths:int, num_ns_steps:int,
                                save_test_results = False):
        print("Start ID-NS test")
        self.encoder.to(device=torch.device('cpu'))
        self.sac.policy.to(device=torch.device('cpu'))
        env_tasks= id_s_agent.env_tasks
        print(f"Num of Tasks: {len(env_tasks)}, num of Path : {num_paths}")
        id_s_log,embs,ids = id_s_agent.eval(env_tasks,num_paths,self.sac.policy,self.encoder)

        N_steps = num_ns_steps
        print(f"Num of Tasks: {len(env_tasks)}, num of Stpes : {N_steps}")
        fig,id_ns_df,ns_embs,change_inds,(diff_from_expert, at_target_ratio)= self.get_figure(id_ns_agent,self.sac.policy,self.encoder,
                                                    torch.device('cpu'),N_steps,env_id=None)
        assert ns_embs.shape[0] == N_steps
        ns_total_rews = np.sum(id_ns_df['rew'])
        ns_mean_rews = np.mean(id_ns_df['rew'])

        min_ids,max_ids = ids.min(),ids.max()
        mean_embs = [] # (n_task,emb)
        for id in range(min_ids,max_ids+1):
            tmp = embs[np.where(ids.reshape(-1) == id)]
            if len(tmp.shape) == 2:
                mean_embs.append(np.mean(tmp,axis = 0))
            elif len(tmp.shape) == 1:
                mean_embs.append(tmp)
        mean_embs = np.stack(mean_embs,axis = 0)
        id_ns_fig = utils.emb_similarity(ns_embs,mean_embs,change_inds)
        id_ns_fig2 = utils.times_series_value(id_ns_df['err'],change_inds)
        id_ns_fig3 = utils.embedding_variation(ns_embs)

        if save_test_results:
            id_s_df = pd.DataFrame(id_s_log)
            id_ns_df= pd.DataFrame(id_ns_df)
            id_s_df.to_csv(os.path.join(self.logger.output_dir, "id-s_test.csv"))
            id_ns_df.to_csv(os.path.join(self.logger.output_dir, "id-ns_test.csv"))
            np.save(os.path.join(self.logger.output_dir, "id-s_embs.npy"),embs)
            np.save(os.path.join(self.logger.output_dir, "id-s_ids.npy"),ids)
            np.save(os.path.join(self.logger.output_dir, "id-ns_embs.npy"),ns_embs)

        id_s_log['NS_EpRet'] = ns_total_rews
        id_s_log['NS_EpMeanRew'] = ns_mean_rews
        return id_s_log,fig,id_ns_fig,id_ns_fig2,id_ns_fig3


    
    def ood_test(self):
        print("Start OOD Test")
        self.ood_agent_single_thread = EnvRemoteWorkers(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num= 4 , seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=True, policy_type=self.policy_type,encoder_type=self.encoder_type,
                                                env_decoration=NonstationaryEnv, env_tasks=self.ood_tasks,
                                                adaptive_maml=self.parameter.adaptive_maml,non_stationary=False)
        
        self.encoder.to(device=torch.device('cpu'))
        self.sac.policy.to(device=torch.device('cpu'))
        env_tasks=self.ood_tasks
        df,embs,ids = self.ood_agent_single_thread.eval(env_tasks,5,self.sac.policy,self.encoder,
                                        adaptive_envtracker=self.parameter.adaptive_envtracker)                                        
        df.to_csv(os.path.join(self.logger.output_dir, "ood_test.csv"))
        np.save(os.path.join(self.logger.output_dir, "embs.npy"),embs)
        np.save(os.path.join(self.logger.output_dir, "ids.npy"),ids)

    def ns_test(self):
        print("Start NS Test")
        self.non_station_agent_single_thread = EnvRemoteWorkers(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=1, seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=False, policy_type=self.policy_type,encoder_type=self.encoder_type,
                                                env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                                adaptive_maml=self.parameter.adaptive_maml,non_stationary=True)
        self.encoder.to(device=torch.device('cpu'))
        self.sac.policy.to(device=torch.device('cpu'))
        num_steps = len(self.test_tasks) * 1000
        repr_fig,df,embs,change_inds,diff_from_expert = self.get_figure(self.non_station_agent_single_thread,self.sac.policy,self.encoder,
                        torch.device('cpu'),num_steps,self.parameter.adaptive_maml,None)
        ### Try Embedding Analysis
        principalComponents1 = df['principalComponents1']
        principalComponents2 = df['principalComponents2']
        ids = df["task_id"]
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        sns.set_theme(style="darkgrid")
        sns.scatterplot(x= principalComponents1,
                    y=principalComponents2,
                    hue=ids.reshape(-1),
                    palette="deep",
                    ax = ax)
        ax.legend(loc = 'upper right',title = 'Task Id')
        ax.set_title(f"Embedding Analysis")
        self.encoder.to(device=self.device)
        self.sac.policy.to(device=self.device)
        self.logger.tb.add_figure('ns_test/embedding_behavior',repr_fig,global_step= -1)
        self.logger.tb.add_figure('ns_test/embedding_analysis',fig,global_step= -1)

        df = pd.DataFrame(df)
        df.to_csv(os.path.join(self.logger.output_dir, "ns_test.csv"))
        print("Complete NS Test")
        
    def save(self,path = None ):
        if path is None:
            self.sac.save(self.logger.model_output_dir)
            self.encoder.save(self.logger.model_output_dir)
        else:
            self.sac.save(path)
            self.encoder.save(path)

    def load(self,path = None):
        if path is None:
            self.sac.load(self.logger.model_output_dir,map_location = self.device)
            self.encoder.load(self.logger.model_output_dir,map_location = self.device)
        else:
            self.sac.load(path,map_location = self.device)
            self.encoder.load(path,map_location = self.device)
    @staticmethod
    def append_key(d, tail):
        res = {}
        for k, v in d.items():
            res[k+tail] = v
        return res

    def get_figure(self,agent:EnvRemoteWorkers,policy:Actor,encoder,device,num_steps,env_id = None):
        assert agent.use_remote == False 
        assert agent.worker_num == 1 

        embs = []
        real_param = []
        actions = []
        rews = []
        action_discrepancy = []
        keep_at_target = []
        errs = []
        done = False 
        while not done:
            mem,env_info ,info = agent.step_local_1env(env_id,policy,encoder,device)
            done = mem.memory[0].done[0]
        for _ in range(num_steps):
            mem,env_info ,info = agent.step_local_1env(env_id,policy,encoder,device)
            real_param.append(mem.memory[0].env_param)
            embs.append(info['emb'])
            actions.append(mem.memory[0].act)
            rews.append(mem.memory[0].rew)
            errs.append(info['err'])
            if isinstance(env_info, dict) and 'action_discrepancy' in env_info and env_info['action_discrepancy'] is not None:
                action_discrepancy.append(np.array([env_info['action_discrepancy'][0],
                                                    env_info['action_discrepancy'][1]]))
                keep_at_target.append(1 if env_info['keep_at_target'] else 0)
        agent.collect_results(-1)
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




    