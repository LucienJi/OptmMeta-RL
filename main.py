import ray
from algorithms.Trainer import SimpleTrainer
from algorithms.opt_Trainer import Opt_Trainer
from log_util.logger import Logger
from parameter.optm_Params import Parameters



if __name__ == '__main__':
    # ray.init()
    parameter = Parameters(config_path="configs/toy_config.json",default_config_path="configs/default_config.json")
    # trainer = SimpleTrainer(parameter=parameter,log_dir='data')
    trainer = Opt_Trainer(parameter=parameter,log_dir='data')
    trainer.learn()
    
    

    
        

