import os
from datetime import datetime
from typing import Tuple
import numpy as np
from legged_gym.envs import LeggedRobot,LeggedRobotCfg
from legged_gym.utils.helpers import class_to_dict,update_class_from_dict,parse_sim_params,get_load_path,update_cfg_from_args,get_args
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

import torch 

def launch(args):
    env_cfg = LeggedRobotCfg()

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    env = LeggedRobot(cfg = env_cfg, 
                      sim_params = sim_params,physics_engine=args.physics_engine,
                      sim_device= args.sim_device,
                        headless=True)
                        #    headless=args.headless) 
    
    return env 


if __name__ == '__main__':
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name="littledog")
    # override some parameters for testing
    # env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)
    # env_cfg.num_envs =  2

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(int(10)):
        actions = 0.01*torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, p_obs, rew, done, info = env.step(actions)
        print(obs.shape,p_obs.shape)
    

    



