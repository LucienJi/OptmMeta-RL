import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from parameter.private_config import *
from envs.nonstationary_env import NonstationaryEnv
from envencoder.buffer import MetaBuffer
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

        #! 不同的测绘环境
        self.training_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)        
        self.test_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                        rand_params=self.parameter.varying_params)
        ## TODO 这里需要确定在 min max parameter 的时候，使用归一化的表示方式
        self.train_tasks = self.training_env.sample_tasks(self.parameter.task_num,linspace=True)
        self.test_tasks = self.test_env.sample_tasks(self.parameter.test_task_num,linspace=False) #! 不使用 linspace，保证随机采样
        
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