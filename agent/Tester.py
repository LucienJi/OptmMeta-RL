import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from parameter.private_config import *
from envs.nonstationary_env import NonstationaryEnv
from envencoder.buffer import MetaBuffer
from agent.udp_Worker import Udp_Worker,Udp_Workers
import gym
import torch
import numpy as np
import random
import utils.visualization as vis
from utils.torch_utils import to_device
from utils.math import dump_info
from envencoder.actor import Actor,QNetwork
from parameter.optm_Params import Parameters
from envencoder.sac import SAC
import envencoder
import torch.nn as nn
import seaborn as sns
import pandas as pd 

class Tester(object):
    def __init__(self,parameter:Parameters) -> None:
        self.parameter = parameter
        assert os.path.exists(self.parameter.model_path),print("Model Not saved")
        assert os.path.exists(self.parameter.save_path),print("Path not exists")
        self.policy_type = Actor
        self.encoder_type = envencoder.UdpEnvencoder
        #! 不同的测绘环境
        self.training_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)        
        self.test_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                        rand_params=self.parameter.varying_params)
        ## TODO 这里需要确定在 min max parameter 的时候，使用归一化的表示方式
        self.train_tasks = self.training_env.sample_tasks(self.parameter.task_num,linspace=True)
        self.test_tasks = self.test_env.sample_tasks(self.parameter.test_task_num,linspace=False) #! 不使用 linspace，保证随机采样
        
        self.training_agent = Udp_Workers(parameter=self.parameter, env_name=self.parameter.env_name,
                                             worker_num=1, seed=self.parameter.seed,
                                             deterministic=False, use_remote=False, policy_type=Actor,encoder_type=self.encoder_type,
                                             env_decoration=NonstationaryEnv,
                                             env_tasks=self.train_tasks,
                                             non_stationary=False)
        self.obs_dim = self.training_env.observation_space.shape[0]
        self.act_dim = self.training_env.action_space.shape[0]

        self.sac = SAC(self.obs_dim,self.act_dim,self.parameter)
        self.encoder = envencoder.UdpEnvencoder(self.obs_dim,self.act_dim,self.parameter.emb_dim,self.parameter)

        ## Load Model 
        self.sac.load(self.parameter.model_path)
        self.encoder.load(self.parameter.model_path)
        self.replay_buffer = MetaBuffer(max_traj_num=100,max_traj_step=1000)
        
    
    # def emb_test(self):


    @staticmethod
    def global_seed(*args, seed):
        for item in args:
            item.seed(seed)
    
    def data_collection(self,total_steps,random = True):
        self.encoder.to(device=torch.device('cpu'))
        self.sac.policy.to(device=torch.device('cpu'))
        print("Data Collection Start with ",self.replay_buffer.size , " Samples")
        assert random == True, print("Only Support Random Sample")
        list_mem = self.training_agent.collect_random_data(total_steps)
        for mem in list_mem:
            self.replay_buffer.push_mem(mem)
        print("Data Collection Done with ", self.replay_buffer.size, " Samples")
    
    @torch.no_grad()
    def emb_test(self,replay_buffer:MetaBuffer = None,bz = 128):
        replay_buffer = self.replay_buffer if replay_buffer is None else replay_buffer
        n_env = replay_buffer.num_envs
        task_indices = [i for i in range(n_env)]
        data_list = [self.replay_buffer.sample_batch(i,bz,with_tensor=True,device=self.parameter.device) for i in task_indices]
        emb_list = []
        feature_list = []
        for i,data in enumerate(data_list):
            feature = self.encoder.encoder.compute_feature(data.obs,data.act,data.obs2,data.rew)
            chosen_embedding,chosen_cor,idx,chosen_mean,correlation,f2e = self.encoder.encoder.inference(data.obs,
                                                 data.act,
                                                 data.obs2,data.rew,task_indices[i],with_all_emb=True)
            feature_list.append(feature.cpu().numpy())
            emb_list.append(chosen_embedding.cpu().numpy())

        mean_emb = self.encoder.encoder._get_embedding().cpu().numpy()
        return feature_list,emb_list,mean_emb

    @torch.no_grad()
    def dist_test(self,replay_buffer:MetaBuffer = None,bz = 128):
        replay_buffer = self.replay_buffer if replay_buffer is None else replay_buffer
        n_env = replay_buffer.num_envs
        dist_matrix = np.zeros(shape=(n_env,n_env))
        pred_err_matrix = np.zeros(shape = (n_env,n_env))
        pred_id_matrix = np.zeros(shape=(n_env,n_env))
        task_indices = [i for i in range(n_env)]
        data_list = [self.replay_buffer.sample_batch(i,bz,with_tensor=True,device=self.parameter.device) for i in task_indices]
        mean_emb = self.encoder.encoder._get_embedding() # (n_env,dim)
        for i,data in enumerate(data_list):
            chosen_embedding,chosen_cor,idx,chosen_mean,correlation,f2e = self.encoder.encoder.inference(data.obs,
                                                 data.act,
                                                 data.obs2,data.rew,task_indices[i],with_all_emb=True)
            predicted_id = torch.argmax(correlation,dim = 1,keepdim = True) ## bz,1
            predicted_id = predicted_id.numpy().reshape(-1)
            counts = np.bincount(predicted_id,minlength=n_env)
            counts = counts / np.sum(counts)
            pred_id_matrix[i,:] = counts
            #! f2e.shape (bz,n_env,dim)
            dist = (f2e - mean_emb.unsqueeze(0))
            assert dist.shape == f2e.shape 
            dist = dist.mean(0).norm(dim = -1).cpu().numpy()
            dist_matrix[i,:] = dist
            
            for j in range(n_env):
                emb_to_use = f2e[:,j,:]
                loss,info = self.encoder.world_decoder._compute_loss(data.obs,data.act,data.obs2,data.rew,emb_to_use)
                pred_err_matrix[i,j] = loss.cpu().numpy()
        return dist_matrix,pred_err_matrix,pred_id_matrix
            


    def save(self,path = None ):
        if path is None:
            self.sac.save(self.parameter.model_path)
            self.encoder.save(self.parameter.model_path)
        else:
            self.sac.save(path)
            self.encoder.save(path)

    def load(self,path = None):
        if path is None:
            self.sac.load(self.parameter.model_path)
            self.encoder.load(self.parameter.model_path)
        else:
            self.sac.load(path)
            self.encoder.load(path)