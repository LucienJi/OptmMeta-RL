import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from parameter.private_config import *
from agent.udp_Worker import Udp_Worker,Udp_Workers
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

class Udp_Trainer:
    def __init__(self,parameter:Parameters = None,log_dir = None ) -> None:
        if parameter is not None:
            self.logger = Logger(parameter=parameter,base_dir=log_dir)
        else:
            self.logger = Logger(parameter=Parameters(),base_dir=log_dir)
    
        self.parameter = self.logger.parameter
        self.policy_type = Actor
        self.encoder_type = envencoder.UdpEnvencoder

        self.policy_config = Actor.make_config_from_param(self.parameter)
        self.value_config = QNetwork.make_config_from_param(self.parameter)
        self.env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)
        self.ood_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_ood_change_range,
                                        rand_params=self.parameter.varying_params)
        self.global_seed(np.random, random, self.env, self.ood_env, seed=self.parameter.seed)
        torch.manual_seed(seed=self.parameter.seed)
        self.env_tasks = self.env.sample_tasks(self.parameter.task_num,linspace=True) ## From small to big
        self.test_tasks = self.env.sample_tasks(self.parameter.test_task_num,linspace=False) ## same range, but different test
        self.ood_tasks = self.ood_env.sample_tasks(self.parameter.test_task_num,linspace=True) ## OOD 

        self.training_agent = Udp_Workers(parameter=self.parameter, env_name=self.parameter.env_name,
                                             worker_num=1, seed=self.parameter.seed,
                                             deterministic=False, use_remote=False, policy_type=Actor,encoder_type=self.encoder_type,
                                             env_decoration=NonstationaryEnv,
                                             env_tasks=self.env_tasks,
                                             non_stationary=False)

        ## Stationary
        self.s_test_agent = Udp_Workers(parameter=self.parameter, env_name=self.parameter.env_name,
                                         worker_num=self.parameter.num_threads, seed=self.parameter.seed + 1,
                                         deterministic=True, use_remote=self.parameter.use_remote, policy_type=Actor,encoder_type=self.encoder_type,
                                         env_decoration=NonstationaryEnv, 
                                         env_tasks=self.ood_tasks,  #### Here, in or odd 
                                         non_stationary=False)

        self.ns_test_agent = Udp_Workers(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=1, seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=False, policy_type=self.policy_type,encoder_type=self.encoder_type,
                                                env_decoration=NonstationaryEnv, env_tasks=self.ood_tasks,non_stationary=True)

        self.obs_dim = self.training_agent.obs_dim
        self.act_dim = self.training_agent.act_dim

        self.sac = SAC(self.obs_dim,self.act_dim,self.parameter)
        self.encoder = envencoder.UdpEnvencoder(self.obs_dim,self.act_dim,self.parameter.emb_dim,self.parameter)
        if self.encoder.encoder.num_envs == 0:
            self.encoder.encoder._set_env(len(self.env_tasks))
        ## Buffer 
        self.replay_buffer = MetaBuffer(max_traj_num=100,max_traj_step=1000)
        self.device = self.parameter.device

    def update_sac_multitask(self,task_indices = None):
         ## update value 
        info = {}
        if task_indices is None:
            task_indices = self.replay_buffer.sample_task_id(self.parameter.task_per_batch)
        bz ,n_task_per_batch = self.parameter.sac_mini_batch_size,self.parameter.task_per_batch
        task_indices = self.replay_buffer.sample_task_id(n_task_per_batch) if task_indices is None else task_indices
        data = self.replay_buffer.sample_multi_task_batch(task_indices,bz,True,self.device) # (n_env,bz,dim)
        stable_emb = self.encoder.get_multitask_value_emb(task_indices,bz,self.device,deterministic=False) ## stable emb
        

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

        return info
    
    def learn(self):
        total_steps = 0
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
                    task_indices_train = self.replay_buffer.sample_task_id(self.parameter.task_per_batch)
                    update_info = self.update(task_indices_train,with_sac=False,stop_gradient=True)
                    sac_info = self.update_sac_multitask(task_indices_train)
                    self.logger.add_tabular_data(tb_prefix='training',**update_info)
                    self.logger.add_tabular_data(tb_prefix='training',**sac_info)

            training_info = self.training_agent.collect_results(total_steps)
            self.logger.add_tabular_data(tb_prefix='training',**training_info)

            self.logger.log_tabular("Global Step",total_steps,tb_prefix = 'timestep',average_only=True)

            # if iter % 10 == 0:
            #     print("Start Testing")
            #     self.sac.policy.to(torch.device('cpu'))
            #     self.encoder.to(torch.device('cpu'))
            #     id_s_log,fig,id_ns_fig,id_ns_fig2,id_ns_fig3 = self.id_ns_test(self.s_test_agent,self.ns_test_agent,
            #                                                                     num_paths=2,num_ns_steps=self.parameter.ns_test_steps,save_test_results=False)
            #     self.logger.add_tabular_data(tb_prefix='evaluation',**self.append_key(id_s_log,"Test"))
            #     self.logger.tb.add_figure('ns_test/embedding_behavior',fig,global_step= total_steps)
            #     self.logger.tb.add_figure('ns_test/Embedding_Similarity',id_ns_fig,global_step= total_steps)
            #     self.logger.tb.add_figure('ns_test/Prediction_Error',id_ns_fig2,global_step= total_steps)
            #     self.logger.tb.add_figure('ns_test/Embedding_Variation',id_ns_fig3,global_step= total_steps)

            if iter % 10 == 0:
                self.save()

            self.logger.dump_tabular(average_only=True)
        
        # tracker.close()
        self.save()
        self.logger.finish()

    


    def update(self,task_indices:list,
               with_sac = True,stop_gradient = False,
               entropy = False,dpp = False,
               cross_entropy = False,cosine = False,certainty = False):
        info = {}
        meta_loss = 0.0
        for id in task_indices:
            support = self.replay_buffer.sample_batch(id,self.parameter.encoder_batch_size,device = self.device)
            query = self.replay_buffer.sample_query(id,
                                                    self.parameter.encoder_batch_size,
                                                    M_to_predict=self.parameter.M_to_predict,
                                                    device = self.device)
            #! 先计算 encoder 的 loss
            encoder_loss,encoder_info,embedding = self.encoder.compute_loss(support.obs,
                    support.act,support.obs2,support.rew,query.obs,query.act,query.obs2,query.rew,
                    env_id = id,
                    distance_entropy=entropy,
                    dpp=dpp,
                    cross_entropy=cross_entropy,
                    cosine=cosine,
                    certainty=certainty)
            info = dump_info(info,encoder_info)
            if stop_gradient:
                embedding = embedding.detach()
            meta_loss += encoder_loss 
            if with_sac:
                q_loss ,q_info = self.sac.compute_q_loss(
                    support.obs,support.act,support.obs2,support.rew,support.done,embedding,embedding)
                actor_loss,actor_info = self.sac.compute_policy_loss(support.obs,embedding)
                alpha_loss,alpha_info = self.sac.compute_alpha_loss(support.obs,embedding)
                info = dump_info(info,q_info)
                info = dump_info(info,actor_info)
                info = dump_info(info,alpha_info)
                meta_loss +=  q_loss + actor_loss + alpha_loss
            #! 更新 moco
            self.encoder.add_and_update(id,embedding)
        meta_loss /= len(task_indices)

        self.encoder.optm.zero_grad()
        self.sac.policy_optimizer.zero_grad()
        self.sac.value_optimizer.zero_grad()
        self.sac.alpha_optimizer.zero_grad()
        meta_loss.backward()
        self.encoder.optm.step()
        self.encoder.update_decoder()

        if with_sac:
            self.sac.policy_optimizer.step()
            self.sac.value_optimizer.step()
            self.sac.alpha_optimizer.step()
            self.sac.update_value_function()

        #! 更新 target
        return {k:np.mean(v) for k,v in info.items()}



    @staticmethod
    def global_seed(*args, seed):
        for item in args:
            item.seed(seed)
    
    def save(self,path = None ):
        if path is None:
            self.sac.save(self.logger.model_output_dir)
            self.encoder.save(self.logger.model_output_dir)
        else:
            self.sac.save(path)
            self.encoder.save(path)

    def load(self,path = None):
        if path is None:
            self.sac.load(self.logger.model_output_dir)
            self.encoder.load(self.logger.model_output_dir)
        else:
            self.sac.load(path)
            self.encoder.load(path)
    
    def data_collection(self,total_steps,random = True):
        self.encoder.to(device=torch.device('cpu'))
        self.sac.policy.to(device=torch.device('cpu'))
        self.logger("Data Collection Start with ",self.replay_buffer.size , " Samples")
        assert random == True, print("Only Support Random Sample")
        list_mem = self.training_agent.collect_random_data(total_steps)
        for mem in list_mem:
            self.replay_buffer.push_mem(mem)
        self.logger("Data Collection Done with ", self.replay_buffer.size, " Samples")
    
    def pretrain(self,iter = 1000,start_aux = 0,
                 entropy = False,
                 dpp = False,
                 cross_entropy = False,
                 cosine = False,certainty = False):
        data_size = self.replay_buffer.size  ## 这个 size 是 n_env 总和
        epochs = max(data_size // self.parameter.encoder_batch_size // self.parameter.task_per_batch,1)
        self.encoder.to(device=self.device)
        self.sac.policy.to(device=self.device)
        self.logger(f"Pretrain: Iter: {iter}, Epochs: {epochs}")
        for i in range(iter):
            for _ in range(epochs):
                task_indices = self.replay_buffer.sample_task_id(self.parameter.task_per_batch)
                pretrain_info = self.update(task_indices,with_sac= False,
                                            stop_gradient=True,
                                            entropy=entropy and i > start_aux,
                                            dpp=dpp and i > start_aux,
                                            cross_entropy=cross_entropy and i > start_aux,
                                            cosine=cosine and i > start_aux ,
                                            certainty=certainty and i > start_aux )
                self.logger.add_tabular_data(tb_prefix='pretrain',**pretrain_info)

            self.logger.log_tabular("iter",i,tb_prefix = 'iteration',average_only=True)
            self.logger.dump_tabular()
        self.logger("Pretrain Done")