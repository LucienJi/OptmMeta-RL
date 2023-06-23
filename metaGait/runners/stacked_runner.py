from .utils import VecEnv
from metaGait.modules.meta_actor_critic import ActorCritic
from metaGait.algorithms.ppo import PPO
from metaGait.algorithms.task_representation import TaskEncoder
from metaGait.modules.encoder import StackEncoder
from metaGait.modules.decoder import WorldDecoder
from torch.utils.tensorboard import SummaryWriter
import torch 
import time 
from collections import deque
import statistics
import os 

class StackedRunner:
    def __init__(self,
                 env:VecEnv,
                 train_cfg,
                 log_dir = None,
                 device = 'cpu'
                 ) -> None:
        self.cfg = train_cfg['runner']
        self.ppo_cfg = train_cfg['ppo']
        self.task_repre_cfg = train_cfg['task_representation']
        self.policy_cfg = train_cfg['policy']
        self.encoder_cfg = train_cfg['encoder']
        self.decoder_cfg = train_cfg['decoder']

        self.device = device

        self.env = env
        self.emb_dim = self.policy_cfg['emb_dim']
        ## Model
        self.policy = ActorCritic(self.env.num_obs,self.env.num_actions,**self.policy_cfg).to(self.device)
        self.encoder = StackEncoder(self.env.num_obs,self.env.num_actions,1,self.emb_dim,
                                    **self.encoder_cfg).to(self.device)
        self.decoder = WorldDecoder(self.env.num_obs,self.env.num_actions,1,self.emb_dim,
                                    **self.decoder_cfg).to(self.device)

        self.ppo_alg = PPO(self.policy,
                           device=self.device,**self.ppo_cfg)
        self.task_encoder_alg = TaskEncoder(self.encoder,
                                            self.decoder,device = self.device,
                                            **self.task_repre_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.ppo_alg.init_storage(self.env.num_envs
                                  ,self.num_steps_per_env,
                                  [self.env.num_obs],[self.emb_dim,],
                                  [self.env.num_actions])
        self.task_encoder_alg.init_storage(self.env.num_envs,
                                           [self.env.num_obs],
                                           [self.env.num_actions])
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Reset 
        _,_ = self.env.reset()
    
    def learn(self,num_learning_iterations):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        obs = self.env.get_observations() # (num_envs,num_obs)
        obs = obs.to(self.device)
        self.ppo_alg.train_mode()
        self.task_encoder_alg.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        #! 可以多次 learn
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode(): #! More Performance Gain 
                for i in range(self.num_steps_per_env):
                    emb = self.task_encoder_alg.act() # (num_envs,emb_dim)
                    actions = self.ppo_alg.act(obs,emb) # (num_envs,num_actions)
                    next_obs,privileged_obs, rewards, dones, infos = self.env.step(actions)
                    next_obs, rewards, dones = next_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    #! Store Data 
                    self.ppo_alg.process_env_step(rewards,dones,infos) 
                    self.task_encoder_alg.prcess_env_step(obs,actions,rewards,next_obs,dones)
                    obs = next_obs

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                start = stop 
                ## start update 
                self.ppo_alg.compute_returns(obs,self.task_encoder_alg.act().detach())

            #! Start Training Policy and Encoder 

            mean_ppo_value_loss,mean_ppo_surrogate_loss = self.ppo_alg.update()
            mean_obs_prediction_loss,mean_rew_prediction_loss = self.task_encoder_alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                #! locals 是使用 locals() 函数返回当前位置的全部局部变量的一个拷贝
                self.log(locals())
            
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir,'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def save(self,path,infos = None):
        to_save = {
            'policy':self.ppo_alg.actor_critic.state_dict(),
            'encoder':self.task_encoder_alg.encoder.state_dict(),
            'decoder':self.task_encoder_alg.decoder.state_dict(),
            'ppo_optimizer_state_dict':self.ppo_alg.optimizer.state_dict(),
            'task_encoder_optimizer_state_dict':self.task_encoder_alg.optimizer.state_dict(),
            'iter':self.current_learning_iteration,
            'infos':infos
        }
        torch.save(to_save,path)
    
    def load(self,path,load_optimizer = True):
        loaded_dict = torch.load(path)
        self.ppo_alg.actor_critic.load_state_dict(loaded_dict['policy'])
        self.task_encoder_alg.encoder.load_state_dict(loaded_dict['encoder'])
        self.task_encoder_alg.decoder.load_state_dict(loaded_dict['decoder'])
        self.current_learning_iteration = loaded_dict['iter']
        if load_optimizer:
            self.ppo_alg.optimizer.load_state_dict(loaded_dict['ppo_optimizer_state_dict'])
            self.task_encoder_alg.optimizer.load_state_dict(loaded_dict['task_encoder_optimizer_state_dict'])
        return loaded_dict['infos']


    def log(self, locs, width=80, pad=35):
        """
        locs 是使用 locals() 函数返回当前位置的全部局部变量的一个拷贝
        不同 runner 的 log 函数不同, 需要根据 learn 的内容进行改变 
        应该是每次 iteration 都会 log 一次
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        mean_std = self.ppo_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_ppo_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_ppo_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/obs_pred_loss', locs['mean_obs_prediction_loss'], locs['it'])
        self.writer.add_scalar('Loss/rew_pred_loss', locs['mean_rew_prediction_loss'], locs['it'])
        self.writer.add_scalar('Loss/ppo_learning_rate', self.ppo_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/task_repre_learning_rate', self.task_encoder_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_ppo_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_ppo_surrogate_loss']:.4f}\n"""
                          f"""{'Obs_pred loss:':>{pad}} {locs['mean_obs_prediction_loss']:.4f}\n"""
                          f"""{'Rew_pred_loss:':>{pad}} {locs['mean_rew_prediction_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_ppo_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_ppo_surrogate_loss']:.4f}\n"""
                          f"""{'Obs_pred loss:':>{pad}} {locs['mean_obs_prediction_loss']:.4f}\n"""
                          f"""{'Rew_pred_loss:':>{pad}} {locs['mean_rew_prediction_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)