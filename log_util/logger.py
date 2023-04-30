# from parameter.Parameter import Parameter

import os
import numpy as np
import time
import copy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from parameter.optm_Params import Parameters
from log_util.logger_base import LoggerBase
from parameter.private_config import *


class Logger(LoggerBase):
    def __init__(self, log_to_file=True, parameter=None, force_backup=False,base_dir = None):
        if parameter:
            self.parameter = parameter
        else:
            self.parameter = Parameters()
        self.base_dir = "data" if base_dir is None else base_dir
        self.output_dir = os.path.join( self.base_dir, self.parameter.short_name,"logs")
        if log_to_file:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if os.path.exists(os.path.join(self.output_dir, 'log.txt')):
                system(f'mv {os.path.join(self.output_dir, "log.txt")} {os.path.join(self.output_dir, "log_back.txt")}')
            self.log_file = open(os.path.join(self.output_dir, 'log.txt'), 'w')
        else:
            self.log_file = None
        super(Logger, self).__init__(self.output_dir, log_file=self.log_file)
        
        self.current_data = {}
        self.logged_data = set()

        self.model_output_dir = self.parameter.model_path
        self.log(f"my output path is {self.output_dir}")

        self.parameter.save_config(self.output_dir)
        if self.parameter.use_wandb:
            import wandb
            self.run = wandb.init(
                project="my-test-project", 
                entity="jingtianji",
                config= {key:getattr(self.parameter,key) for key in self.parameter.args_name},
                tags=[self.parameter.env_name],
                group=self.parameter.task_name,
                sync_tensorboard=True,
                name=self.parameter.short_name,
                monitor_gym=False,
                save_code=False,
          )
        self.init_tb()
        
        # self.backup_code()
        self.tb_header_dict = {}
    
    def finish(self):
        self.tb.close()


    def log(self, *args, color=None, bold=True):
        super(Logger, self).log(*args, color=color, bold=bold)

    def log_dict(self, color=None, bold=False, **kwargs):
        for k, v in kwargs.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def log_dict_single(self, data, color=None, bold=False):
        for k, v in data.items():
            super(Logger, self).log('{}: {}'.format(k, v), color=color, bold=bold)

    def __call__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    def save_config(self):
        self.parameter.save_config()

    def log_tabular(self, key, val=None, tb_prefix=None, with_min_and_max=False, average_only=False, no_tb=False):
        if val is not None:
            super(Logger, self).log_tabular(key, val, tb_prefix, no_tb=no_tb)
        else:
            if key in self.current_data:
                self.logged_data.add(key)
                super(Logger, self).log_tabular(key if average_only else "Average_"+key, np.mean(self.current_data[key]), tb_prefix, no_tb=no_tb)
                if not average_only:
                    super(Logger, self).log_tabular("Std_" + key,
                                                    np.std(self.current_data[key]), tb_prefix, no_tb=no_tb)
                    if with_min_and_max:
                        super(Logger, self).log_tabular("Min" + key, np.min(self.current_data[key]), tb_prefix, no_tb=no_tb)
                        super(Logger, self).log_tabular('Max' + key, np.max(self.current_data[key]), tb_prefix, no_tb=no_tb)

    def add_tabular_data(self, tb_prefix=None, **kwargs):
        for k, v in kwargs.items():
            if tb_prefix is not None and k not in self.tb_header_dict:
                self.tb_header_dict[k] = tb_prefix
            if k not in self.current_data:
                self.current_data[k] = []
            if not isinstance(v, list):
                self.current_data[k].append(v)
            else:
                self.current_data[k] += v

    def update_tb_header_dict(self, tb_header_dict):
        self.tb_header_dict.update(tb_header_dict)

    def dump_tabular(self,average_only = True):
        for k in self.current_data:
            if k not in self.logged_data:
                if k in self.tb_header_dict:
                    self.log_tabular(k, tb_prefix=self.tb_header_dict[k], average_only=average_only)
                else:
                    self.log_tabular(k, average_only=average_only)
        self.logged_data.clear()
        self.current_data.clear()
        super(Logger, self).dump_tabular()


if __name__ == '__main__':
    logger = Logger()
    





