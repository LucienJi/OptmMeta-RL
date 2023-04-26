import torch 
import torch.nn as nn 
import numpy as np
from envencoder.advanced_encoder.cusum_tracker import NS_Tracker_v2
from algorithms.Trainer import SimpleTrainer
from log_util.logger import Logger
from parameter.optm_Params import Parameters
from envencoder.advanced_encoder.e_encoder import E_Encoder,UDP_Encoder
from envencoder.advanced_envencoder.udp_envencoder import UDP_Envencoder



if __name__ == '__main__':
    para = Parameters(config_path= "configs/toy_config.json",default_config_path="configs/default_config.json")

    envencoder = UDP_Envencoder(obs_dim=5,act_dim=2,emb_dim=2,parameter=para)
    # model =  UDP_Encoder(obs_dim=5,act_dim=2,emb_dim=2,max_env_num=10)
    envencoder.encoder.add_env(5)

    # obs = torch.randn(10,5)
    # act = torch.randn(10,2)
    # obs2 = torch.randn(10,5)
    # rew = torch.randn(10,1)
    obs = torch.randn(5)
    act = torch.randn(2)
    obs2 = torch.randn(5)
    rew = torch.randn(1)

    # loss,info = envencoder.compute_loss(obs,act,obs2,rew,1,True)
    embeddings = envencoder.sample(obs,act,obs2,rew)
    print(embeddings.shape)
        
     

    


    
    
    
    
   

    