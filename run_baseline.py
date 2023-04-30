import ray
from algorithms.Trainer import SimpleTrainer
from log_util.logger import Logger
import gym
from envs.nonstationary_env import NonstationaryEnv
from parameter.optm_Params import Parameters
import os 
import argparse

def run(path):
    parameter = Parameters(path)
    trainer = SimpleTrainer(parameter=parameter,log_dir='data')
    # trainer.load()
    trainer.learn()

if __name__ == '__main__':
    base_dir = "configs/video_config"
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',type=str,default='friction')
    parser.add_argument('--env',type=str,default='halfcheetah')
    args = parser.parse_args()
    filename = '_'.join([args.env,args.test]) + '.json'
    path = os.path.join(base_dir,filename)
    run(path)