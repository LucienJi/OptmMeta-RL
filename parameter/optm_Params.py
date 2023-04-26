import os, sys
import argparse
import json
import socket
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter.private_config import *
from datetime import datetime

class Parameters:
    def __init__(self, config_path=None,default_config_path = None , debug=False, information=None):
        self.base_path = self.get_base_path()
        self.debug = debug
        self.experiment_target = EXPERIMENT_TARGET
        self.DEFAULT_CONFIGS = global_configs()
        self.arg_names = []
        self.host_name = 'localhost'
        self.ip = '127.0.0.1'
        self.exec_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.commit_id = self.get_commit_id()
        self.log_func = None
        self.json_name = 'parameter.json'
        self.txt_name = 'parameter.txt'
        self.information = information
        if default_config_path is not None:
            self.info(f"Defaul config path: {default_config_path}")
            with open(default_config_path, 'r') as f:
                default_config = json.load(f)
        else:
            default_config = dict()
        if config_path:
            self.info("Use additional config")
            self.config_path = config_path
            with open(config_path,'r') as f:
                add_config = json.load(f)
            for k,new_vaule in add_config.items():
                default_config[k] = new_vaule
            for k,v in default_config.items():
                self.register_param(k)
                setattr(self,k,v)
        else:
            self.info('Use command line arguments')
            self.config_path = osp.join(get_base_path(), 'parameter')
            self.args = self.parse()
            self.apply_vars(self.args)
        

    def set_log_func(self, log_func):
        self.log_func = log_func

    def info(self, info):
        if self.log_func is not None:
            self.log_func(info)
        else:
            print(info)
    @staticmethod
    def get_base_path():
        return get_base_path()

    def set_config_path(self, config_path):
        self.config_path = config_path

    @staticmethod
    def important_configs():
        res = ['env_name','algo_name','task_name']
        return res

    def apply_vars(self, args):
        for name in self.arg_names:
            setattr(self, name, getattr(args, name))
    def make_dict(self):
        res = {}
        for name in self.arg_names:
            res[name] = getattr(self, name)
        res['exec_time'] = self.exec_time
        return res
    
    def register_param(self, name):
        self.arg_names.append(name)

    def get_experiment_description(self):
        description = f"本机{self.host_name}, ip为{self.ip}\n"
        description += f"实验简称: {self.short_name}\n"
        vars = ''
        important_config = self.important_configs()
        for name in self.arg_names:
            if name in important_config:
                vars += f'**{name}**: {getattr(self, name)}\n'
            else:
                vars += f'{name}: {getattr(self, name)}\n'
        for name in self.DEFAULT_CONFIGS:
            vars += f'{name}: {self.DEFAULT_CONFIGS[name]}\n'
        return description + vars

    def __str__(self):
        return self.get_experiment_description()

    def clear_local_file(self):
        cmd = f'rm -f {os.path.join(self.config_path, self.json_name)} {os.path.join(self.config_path, self.txt_name)}'
        system(cmd)

    def save_config(self):
        self.info(f'save json config to {os.path.join(self.config_path, self.json_name)}')
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
        with open(os.path.join(self.config_path, self.json_name), 'w') as f:
            things = self.make_dict()
            ser = json.dumps(things)
            f.write(ser)
        self.info(f'save readable config to {os.path.join(self.config_path, self.txt_name)}')
        with open(os.path.join(self.config_path, self.txt_name), 'w') as f:
            print(self, file=f)

    def load_config(self):
        self.info(f'load json config from {os.path.join(self.config_path, self.json_name)}')
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        for k, v in ser.items():
            if not k == 'description':
                setattr(self, k, v)
                self.register_param(k)
        self.experiment_target = ser['description']

    @property
    def differences(self):
        if not os.path.exists(os.path.join(self.config_path, self.json_name)):
            return None
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        differences = []
        for k, v in ser.items():
            if not hasattr(self, k):
                differences.append(k)
            else:
                v2 = getattr(self, k)
                if not v2 == v:
                    differences.append(k)
        return differences

    def check_identity(self, need_decription=False, need_exec_time=False):
        if not os.path.exists(os.path.join(self.config_path, self.json_name)):
            self.info(f'{os.path.join(self.config_path, self.json_name)} not exists')
            return False
        with open(os.path.join(self.config_path, self.json_name), 'r') as f:
            ser = json.load(f)
        flag = True
        for k, v in ser.items():
            if not k == 'description' and not k == 'exec_time':
                if not hasattr(self, k):
                    flag = False
                    return flag
                v2 = getattr(self, k)
                if not v2 == v:
                    flag = False
                    return flag
        if need_decription:
            if not self.experiment_target == ser['description']:
                flag = False
                return flag
        if need_exec_time:
            if not self.exec_time == ser['exec_time']:
                flag = False
                return flag
        return flag

    @property
    def short_name(self):
        name = ''
        for item in self.important_configs():
            value = getattr(self, item)
            if value:
                if item == 'env_name':
                    name += value
                elif item == 'task_name':
                    name += f'-{value}'
                elif item =='algo_name':
                    name += f'-{value}'
        return name

    def get_commit_id(self):
        base_path = get_base_path()
        cmd = f'cd {base_path} && git log'
        commit_id = None
        try:
            with os.popen(cmd) as f:
                line = f.readline()
                words = line.split(' ')
                commit_id = words[-1][:-1]
        except Exception as e:
            self.info(f'Error occurs while fetching commit id!!! {e}')
        return commit_id
    def parse(self):
        parser = argparse.ArgumentParser(description=EXPERIMENT_TARGET)
        ### ENV ###
        self.use_remote = False
        parser.add_argument('--use_remote',type = bool, default=self.use_remote, metavar='N',
                            help='use remote to test')
        self.register_param('use_remote')

        self.use_wandb = False
        parser.add_argument('--use_wandb',type=bool, default=self.use_wandb, metavar='N',
                            help='use wandb to track')
        self.register_param('use_wandb')

        self.task_name = "CodeTest"
        parser.add_argument('--task_name', default=self.task_name, metavar='G',
                            help='name of the task to test')
        self.register_param('task_name')

        self.algo_name = "baseline"
        parser.add_argument('--algo_name', default=self.algo_name, metavar='G',
                            help='name of the algo to test')
        self.register_param('algo_name')

        # self.env_name = "InvertedPendulum-v2"
        self.env_name = "HalfCheetah-v2"
        parser.add_argument('--env_name', default=self.env_name, metavar='G',
                            help='name of the environment to run')
        self.register_param('env_name')

        self.skip_max_len_done = True
        parser.add_argument('--skip_max_len_done',default=self.skip_max_len_done)
        self.register_param('skip_max_len_done')

        self.task_num = 20
        parser.add_argument('--task_num', type=int, default=self.task_num, metavar='N',
                            help="Number of env during training")
        self.register_param('task_num')

        self.test_task_num = 10
        parser.add_argument('--test_task_num', type=int, default=self.test_task_num, metavar='N',
                            help="number of tasks for testing")
        self.register_param('test_task_num')

        self.test_sample_num = 4000
        parser.add_argument('--test_sample_num', type=int, default=self.test_sample_num, metavar='N',
                            help='sample num in test phase')
        self.register_param('test_sample_num')

        self.ns_test_steps = 1000
        parser.add_argument('--ns_test_steps', type=int, default=self.ns_test_steps, metavar='N',
                            help='sample num in test phase')
        self.register_param('ns_test_steps')


        self.env_default_change_range = 3.0
        parser.add_argument('--env_default_change_range', type=float, default=self.env_default_change_range, metavar='N',
                            help="environment default change range")
        self.register_param('env_default_change_range')

        self.env_ood_change_range = 4.0
        parser.add_argument('--env_ood_change_range', type=float, default=self.env_ood_change_range,
                            metavar='N',
                            help="environment OOD change range")
        self.register_param('env_ood_change_range')

        # self.varying_params = ['gravity', 'body_mass']
        self.varying_params = ['gravity']
        parser.add_argument('--varying_params', nargs='+', type=str, default=self.varying_params)
        self.register_param('varying_params')


        self.model_path = ""
        parser.add_argument('--model_path', metavar='G',
                            help='path of pre-trained model')
        self.register_param("model_path")

        self.render = False
        parser.add_argument('--render', action='store_true', default=self.render,
                            help='render the environment')
        self.register_param("render")


        self.gamma = 0.99
        parser.add_argument('--gamma', type=float, default=self.gamma, metavar='G',
                            help='discount factor (default: 0.99)')
        self.register_param('gamma')


        self.num_threads = 4
        parser.add_argument('--num_threads', type=int, default=self.num_threads, metavar='N',
                            help='number of threads for agent (default: 1)')
        self.register_param('num_threads')

        self.seed = 1
        parser.add_argument('--seed', type=int, default=self.seed, metavar='N',
                            help='random seed (default: 1)')
        self.register_param('seed')

        self.random_num = 4000
        # self.random_num = 4
        parser.add_argument('--random_num', type=int, default=self.random_num, metavar='N',
                            help='sample random_num fully random samples,')
        self.register_param('random_num')

        self.start_train_num = 20000
        # self.start_train_num = 200
        parser.add_argument('--start_train_num', type=int, default=self.start_train_num, metavar='N',
                            help='after reach start_train_num, training start')
        self.register_param('start_train_num')

        self.inner_iter_num = 1
        parser.add_argument('--inner_iter_num', type=int, default=self.inner_iter_num, metavar='N',
                                help='model will be optimized for sinner_iter_num times.')
        self.register_param('inner_iter_num')

        self.update_sac_interval = 1
        parser.add_argument('--update_sac_interval', type=int, default=self.update_sac_interval, metavar='N',
                                help='Update SAC for every update_interval step.')
        self.register_param('update_sac_interval')

        self.update_encoder_interval = 1
        parser.add_argument('--update_encoder_interval', type=int, default=self.update_encoder_interval, metavar='N',
                                help='Update encoder for every update_interval step.')
        self.register_param('update_encoder_interval')

        self.max_iter_num = 2000
        parser.add_argument('--max_iter_num', type=int, default=self.max_iter_num, metavar='N',
                            help='maximal number of main iterations (default: 500)')
        self.register_param('max_iter_num')

        self.min_batch_size = 1000
        parser.add_argument('--min_batch_size', type=int, default=self.min_batch_size, metavar='N',
                            help='minimal sample number per iteration')
        self.register_param('min_batch_size')

        ### EnvEncoder ###
        ##### Transition #####
        
        self.transition_hidden_size = [128,256,128]
        parser.add_argument('--transition_hidden_size',type = int,default=self.transition_hidden_size)
        self.register_param('transition_hidden_size')

        self.transition_deterministic = False
        parser.add_argument('--transition_deterministic',type = bool,default=self.transition_deterministic)
        self.register_param('transition_deterministic')

        ##### ENOCDER #####
        self.emb_dim = 2
        parser.add_argument('--emb_dim', type=int, default=self.emb_dim, metavar='N',
                            help="dimension of environment features")
        self.register_param('emb_dim')

        self.encoder_batch_size = 128
        parser.add_argument('--encoder_batch_size', type=int, default=self.encoder_batch_size, metavar='N',
                            help="encoder_batch_size")
        self.register_param('encoder_batch_size')

        self.n_support = 16
        parser.add_argument('--n_support', type=int, default=self.n_support, metavar='N',
                            help="history length for encoder")
        self.register_param('n_support')


        self.meta_lr = 0.01
        parser.add_argument('--meta_lr', type=float, default=self.meta_lr, metavar='N',
                            help="meta lr for meta-learning")
        self.register_param('meta_lr')

        self.task_per_batch = 10
        parser.add_argument('--task_per_batch', type=int, default=self.task_per_batch, metavar='N',
                            help="n task per batch for meta-learning")
        self.register_param('task_per_batch')


        self.encoder_hidden_size = [256,256]
        parser.add_argument('--encoder_hidden_size',nargs='+', type=int, default=self.encoder_hidden_size)
        self.register_param('encoder_hidden_size')

        self.log_confidence_loss = True
        parser.add_argument('--log_confidence_loss',nargs='+', type=bool, default=self.log_confidence_loss)
        self.register_param('log_confidence_loss')

        self.log_confidence_coef = 0.1
        parser.add_argument('--log_confidence_coef',nargs='+', type=float, default=self.log_confidence_coef)
        self.register_param('log_confidence_coef')

        


        self.encoder_lr = 3e-4
        parser.add_argument('--encoder_lr',type= float,default=self.encoder_lr)
        self.register_param('encoder_lr')

        self.emb_tau = 0.995
        parser.add_argument('--emb_tau', type=float, default=self.emb_tau, metavar='N',
                            help='ratio of update mean emb for target value net')
        self.register_param('emb_tau')

        ### SAC PART ###
        self.sac_mini_batch_size = 256
        parser.add_argument('--sac_mini_batch_size', type=int, default=self.sac_mini_batch_size, metavar='N',
                                help='update time after sampling a batch data')
        self.register_param('sac_mini_batch_size')

        self.max_grad_norm  = 10.0 
        parser.add_argument('--max_grad_norm',type = float,default = self.max_grad_norm)
        self.register_param('max_grad_norm')

        ##### POLICY #####
        self.policy_hidden_size = [256,256] # [256, 128]
        parser.add_argument('--policy_hidden_size', nargs='+', type=int, default=self.policy_hidden_size,
                            help="architecture of the hidden layers of Environment Probing Net")
        self.register_param('policy_hidden_size')

        self.policy_learning_rate = 3e-4
        parser.add_argument('--policy_learning_rate', type=float, default=self.policy_learning_rate, metavar='G',
                            help='learning rate (default: 3e-4)')
        self.register_param('policy_learning_rate')


        ##### VALUE #####
        self.sac_alpha = 1.0
        parser.add_argument('--sac_alpha', type=float, default=self.sac_alpha, metavar='N',
                            help='sac temperature coefficient')
        self.register_param('sac_alpha')

        self.sac_tau = 0.995
        parser.add_argument('--sac_tau', type=float, default=self.sac_tau, metavar='N',
                            help='ratio of coping value net to target value net')
        self.register_param('sac_tau')

        self.value_learning_rate = 1e-3
        parser.add_argument('--value_learning_rate', type=float, default=self.value_learning_rate, metavar='G',
                            help='learning rate (default: 1e-3)')
        self.register_param('value_learning_rate')

        self.value_hidden_size = [256,256]
        parser.add_argument('--value_hidden_size', nargs='+', type=int, default=self.value_hidden_size,
                            help="architecture of the hidden layers of value")
        self.register_param('value_hidden_size')



        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"
        self.register_param('device')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        return args


