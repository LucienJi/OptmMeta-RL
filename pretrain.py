import torch 
import torch.nn as nn 
import numpy as np
from envencoder.advanced_encoder.cusum_tracker import NS_Tracker_v2
from algorithms.udp_Trainer import Udp_Trainer
from log_util.logger import Logger
from parameter.optm_Params import Parameters
# from envencoder.advanced_encoder.e_encoder import E_Encoder,UDP_Encoder
from envencoder.advanced_envencoder.udp_envencoder import UdpEnvencoder
import envs 

if __name__ == '__main__':
    para = Parameters(config_path= "configs/pretrain_config_v2.json",default_config_path="configs/default_config.json")
    trainer = Udp_Trainer(parameter=para)

    trainer.data_collection(total_steps=500000,random=True)
    trainer.pretrain(100,start_aux=-1,
                     entropy=False,
                     dpp=False,
                     cross_entropy=True,
                     cosine=False,
                     certainty=False,)
    trainer.save()
    trainer.load()