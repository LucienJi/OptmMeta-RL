import os, sys
import argparse
import json
import socket
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter.private_config import *
from datetime import datetime

class Parameters:
    def __init__(self, config_path=None,default_config_path = None ,save_path ="data"):
        self.save_path = save_path
        self.args_name = []
        self.json_name = 'parameter.json'
        self.txt_name = 'parameter.txt'
        if default_config_path is not None:
            self.info(f"Defaul config path: {default_config_path}")
            with open(default_config_path, 'r') as f:
                default_config = json.load(f)
        else:
            default_config = dict()

        if config_path is not None:
            self.info("Use additional config")
            self.config_path = config_path
            with open(config_path,'r') as f:
                add_config = json.load(f)
            for k,new_vaule in add_config.items():
                default_config[k] = new_vaule
        else:
            self.info('Use command line arguments')
            args = self.parse()
            args = args._get_kwargs()
            for arg in args:
                default_config[arg[0]] = arg[1]
        self.args = default_config
        self.apply_vars(self.args)
        self.save_path =  os.path.join(save_path,self.short_name)
        self.model_path = self.save_path
        # self.save_config(self.save_path)

    def info(self, info):
        print(info)

    @staticmethod
    def important_configs():
        res = ['env_name','algo_name','task_name']
        return res
    @property
    def short_name(self):
        name = f"{self.env_name}-{self.algo_name}-{self.task_name}"
        return name 

    def apply_vars(self, args):
        for k,v in args.items():
            setattr(self, k,v)
            self.args_name.append(k)

    def get_experiment_description(self):
        description = f"实验简称: {self.short_name}\n"
        vars = ''
        important_config = self.important_configs()
        for k,v in self.args.items():
            vars += f'{k}: {getattr(self, k)}\n'
        
        return description + vars

    def __str__(self):
        return self.get_experiment_description()

    def save_config(self,path):
        self.info(f'save json config to {os.path.join(path, self.json_name)}')
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, self.json_name), 'w') as f:
            ser = json.dumps(self.args)
            f.write(ser)
        self.info(f'save readable config to {os.path.join(path, self.txt_name)}')
        with open(os.path.join(path, self.txt_name), 'w') as f:
            print(self, file=f)

    def load_config(self,path):
        self.info(f'load json config from {os.path.join(path, self.json_name)}')
        with open(os.path.join(path, self.json_name), 'r') as f:
            ser = json.load(f)
            self.args = ser

    def parse(self):
        parser = argparse.ArgumentParser(description=EXPERIMENT_TARGET)
        ### ENV ###
        self.use_remote = False
        parser.add_argument('--use_remote',type = bool, default=self.use_remote, metavar='N',
                            help='use remote to test')

        self.use_wandb = False
        parser.add_argument('--use_wandb',type=bool, default=self.use_wandb, metavar='N',
                            help='use wandb to track')

        self.task_name = "CodeTest"
        parser.add_argument('--task_name', default=self.task_name, metavar='G',
                            help='name of the task to test')
        

        self.algo_name = "baseline"
        parser.add_argument('--algo_name', default=self.algo_name, metavar='G',
                            help='name of the algo to test')
        

        # self.env_name = "InvertedPendulum-v2"
        self.env_name = "HalfCheetah-v2"
        parser.add_argument('--env_name', default=self.env_name, metavar='G',
                            help='name of the environment to run')
        

        self.skip_max_len_done = True
        parser.add_argument('--skip_max_len_done',default=self.skip_max_len_done)
        

        self.task_num = 20
        parser.add_argument('--task_num', type=int, default=self.task_num, metavar='N',
                            help="Number of env during training")
        

        self.test_task_num = 10
        parser.add_argument('--test_task_num', type=int, default=self.test_task_num, metavar='N',
                            help="number of tasks for testing")
        

        self.test_sample_num = 4000
        parser.add_argument('--test_sample_num', type=int, default=self.test_sample_num, metavar='N',
                            help='sample num in test phase')
        

        self.ns_test_steps = 1000
        parser.add_argument('--ns_test_steps', type=int, default=self.ns_test_steps, metavar='N',
                            help='sample num in test phase')
        


        self.env_default_change_range = 3.0
        parser.add_argument('--env_default_change_range', type=float, default=self.env_default_change_range, metavar='N',
                            help="environment default change range")
        

        self.env_ood_change_range = 4.0
        parser.add_argument('--env_ood_change_range', type=float, default=self.env_ood_change_range,
                            metavar='N',
                            help="environment OOD change range")


        # self.varying_params = ['gravity', 'body_mass']
        self.varying_params = ['gravity']
        parser.add_argument('--varying_params', nargs='+', type=str, default=self.varying_params)
        


        self.render = False
        parser.add_argument('--render', action='store_true', default=self.render,
                            help='render the environment')
        


        self.gamma = 0.99
        parser.add_argument('--gamma', type=float, default=self.gamma, metavar='G',
                            help='discount factor (default: 0.99)')


        self.num_threads = 4
        parser.add_argument('--num_threads', type=int, default=self.num_threads, metavar='N',
                            help='number of threads for agent (default: 1)')

        self.seed = 1
        parser.add_argument('--seed', type=int, default=self.seed, metavar='N',
                            help='random seed (default: 1)')

        self.random_num = 4000
        # self.random_num = 4
        parser.add_argument('--random_num', type=int, default=self.random_num, metavar='N',
                            help='sample random_num fully random samples,')

        self.start_train_num = 20000
        # self.start_train_num = 200
        parser.add_argument('--start_train_num', type=int, default=self.start_train_num, metavar='N',
                            help='after reach start_train_num, training start')

        self.inner_iter_num = 1
        parser.add_argument('--inner_iter_num', type=int, default=self.inner_iter_num, metavar='N',
                                help='model will be optimized for sinner_iter_num times.')
        

        self.update_sac_interval = 1
        parser.add_argument('--update_sac_interval', type=int, default=self.update_sac_interval, metavar='N',
                                help='Update SAC for every update_interval step.')


        self.update_encoder_interval = 1
        parser.add_argument('--update_encoder_interval', type=int, default=self.update_encoder_interval, metavar='N',
                                help='Update encoder for every update_interval step.')
        

        self.max_iter_num = 2000
        parser.add_argument('--max_iter_num', type=int, default=self.max_iter_num, metavar='N',
                            help='maximal number of main iterations (default: 500)')
        

        self.min_batch_size = 1000
        parser.add_argument('--min_batch_size', type=int, default=self.min_batch_size, metavar='N',
                            help='minimal sample number per iteration')
        

        ### EnvEncoder ###
        ##### Transition #####
        
        self.transition_hidden_size = [128,256,128]
        parser.add_argument('--transition_hidden_size',type = int,default=self.transition_hidden_size)

        self.transition_deterministic = False
        parser.add_argument('--transition_deterministic',type = bool,default=self.transition_deterministic)

        ##### ENOCDER #####
        self.emb_dim = 2
        parser.add_argument('--emb_dim', type=int, default=self.emb_dim, metavar='N',
                            help="dimension of environment features")

        self.encoder_batch_size = 128
        parser.add_argument('--encoder_batch_size', type=int, default=self.encoder_batch_size, metavar='N',
                            help="encoder_batch_size")

        self.n_support = 16
        parser.add_argument('--n_support', type=int, default=self.n_support, metavar='N',
                            help="history length for encoder")
        self.M_to_predict = 5
        parser.add_argument('--M_to_predict',type=int, default=self.n_support, metavar='N',
                            help="history length for encoder")

    
        self.task_per_batch = 10
        parser.add_argument('--task_per_batch', type=int, default=self.task_per_batch, metavar='N',
                            help="n task per batch for meta-learning")
        


        self.encoder_hidden_size = [256,256]
        parser.add_argument('--encoder_hidden_size',nargs='+', type=int, default=self.encoder_hidden_size)


        self.encoder_lr = 3e-4
        parser.add_argument('--encoder_lr',type= float,default=self.encoder_lr)

        self.emb_tau = 0.95
        parser.add_argument('--emb_tau', type=float, default=self.emb_tau, metavar='N',
                            help='ratio of update mean emb for target value net')
        self.length_scale = 0.1
        parser.add_argument('--length_scale', type=float, default=self.emb_tau, metavar='N',
                            help='smaller length scale, more likely to be distant for different env')

        ### SAC PART ###
        self.sac_mini_batch_size = 256
        parser.add_argument('--sac_mini_batch_size', type=int, default=self.sac_mini_batch_size, metavar='N',
                                help='update time after sampling a batch data')

        self.max_grad_norm  = 10.0 
        parser.add_argument('--max_grad_norm',type = float,default = self.max_grad_norm)

        ##### POLICY #####
        self.policy_hidden_size = [256,256] # [256, 128]
        parser.add_argument('--policy_hidden_size', nargs='+', type=int, default=self.policy_hidden_size,
                            help="architecture of the hidden layers of Environment Probing Net")

        self.policy_learning_rate = 3e-4
        parser.add_argument('--policy_learning_rate', type=float, default=self.policy_learning_rate, metavar='G',
                            help='learning rate (default: 3e-4)')


        ##### VALUE #####
        self.sac_alpha = 1.0
        parser.add_argument('--sac_alpha', type=float, default=self.sac_alpha, metavar='N',
                            help='sac temperature coefficient')

        self.sac_tau = 0.995
        parser.add_argument('--sac_tau', type=float, default=self.sac_tau, metavar='N',
                            help='ratio of coping value net to target value net')

        self.value_learning_rate = 1e-3
        parser.add_argument('--value_learning_rate', type=float, default=self.value_learning_rate, metavar='G',
                            help='learning rate (default: 1e-3)')

        self.value_hidden_size = [256,256]
        parser.add_argument('--value_hidden_size', nargs='+', type=int, default=self.value_hidden_size,
                            help="architecture of the hidden layers of value")



        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        return args


