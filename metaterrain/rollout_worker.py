import numpy as np
import torch
import ray
import os
import gym 
from collections import OrderedDict
import copy
from envs.wrappers import NormalizedBoxEnv
from envs.nonstationary_env import NonstationaryEnv
import rlkit.torch.pytorch_util as ptu
import random
from metaterrain.replay_buffer import Instance

class RolloutCoordinator:
    def __init__(self,
                 env_name,
                 env_args,
                 agent,
                 replay_buffer,
                 time_steps,
                 max_path_length,
                 permute_samples,
                 use_remote,
                 use_data_normalization,
                 num_workers,
                 model_path
                 ):
        
        environment = gym.make(env_name)
        if env_args['use_normalized_env']:
            environment = NormalizedBoxEnv(environment)
        if env_args['non_stationary']:
            environment = NonstationaryEnv(environment)
        self.env = environment
        self.env_tasks = self.env.sample_tasks(10)

        self.replay_buffer = replay_buffer
        self.time_steps = time_steps
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        

        self.use_remote = use_remote
        self.use_data_normalization = use_data_normalization
        self.num_workers = num_workers
        

        self.num_env_steps = 0

        if self.use_remote:
            ray.init(
                num_cpus=100
                # memory=1000 * 1024 * 1024,
                # object_store_memory=2500 * 1024 * 1024,
                # driver_object_store_memory=1000 * 1024 * 1024
            )
            self.workers = [
                RemoteRolloutWorker.remote(
                              env_name=env_name,
                              env_args=env_args,
                              agent=agent,
                              time_steps=time_steps,
                              max_path_length=max_path_length,
                              permute_samples=permute_samples,
                              use_data_normalization=use_data_normalization,
                              replay_buffer_stats_dict=replay_buffer.stats_dict,
                              env_tasks=self.env_tasks,
                              model_path=model_path
                )
            ]
        else:
            self.workers = [
                RolloutWorker(env_name=env_name,
                              env_args=env_args,
                              agent=agent,
                              time_steps=time_steps,
                              max_path_length=max_path_length,
                              permute_samples=permute_samples,
                              use_data_normalization=use_data_normalization,
                              replay_buffer_stats_dict=replay_buffer.stats_dict,
                              env_tasks=self.env_tasks,
                              model_path=model_path
                              )
            ]

    def sample_one_traj(self,deterministic,path_length= None):
        memory = []
        path_length = self.max_path_length if path_length is None else path_length
        if self.use_remote:
            res = ray.get([worker.rollout.remote(deterministic,path_length) for worker in self.workers])
        else:
            res = [[worker.rollout(deterministic,path_length) for worker in self.workers]]
        for m in res:
            memory.extend(m)
        return memory


    def collect_replay_data(self,max_samples):
        num_samples = 0 
        while num_samples < max_samples:
            memory = self.sample_one_traj(deterministic=False)
            num_samples += len(memory)
            for m in memory:
                self.replay_buffer.push(m)
        return num_samples

    def update_model(self):
        if self.use_remote:
            ray.get([worker.fetch_model.remote() for worker in self.workers])
        else:
            [worker.fetch_model() for worker in self.workers]

    def evaluate(self,deterministic=True,):

        if self.use_remote:
            ray.get([worker.reset.remote() for worker in self.workers])
            eval_res = ray.get([worker.rollout.remote(deterministic=deterministic) for worker in self.workers])
        else:
            [worker.reset() for worker in self.workers]
            eval_res = [worker.rollout(deterministic=deterministic) for worker in self.workers]
        res = {k : [] for k in eval_res[0].keys()}
        for eval in eval_res:
            for k in eval.keys():
                res[k].extend(eval[k])

        eval_statistics = OrderedDict()

        deterministic_string = '_deterministic' if deterministic else '_non_deterministic'
        per_path_rewards = np.array(res['return'])
        per_path_lens = np.array(res['len'])
        
        eval_average_reward = per_path_rewards.mean()
        eval_std_reward = per_path_rewards.std()
        eval_max_reward = per_path_rewards.max()
        eval_min_reward = per_path_rewards.min()

        eval_average_len = per_path_lens.mean()
        eval_std_len = per_path_lens.std()
        eval_max_len = per_path_lens.max()
        eval_min_len = per_path_lens.min()

        eval_statistics['eval_avg_reward' + deterministic_string] = eval_average_reward
        eval_statistics['eval_std_reward' + deterministic_string] = eval_std_reward
        eval_statistics['eval_max_reward' + deterministic_string] = eval_max_reward
        eval_statistics['eval_min_reward' + deterministic_string] = eval_min_reward

        eval_statistics['eval_avg_length' + deterministic_string] = eval_average_len
        eval_statistics['eval_std_length' + deterministic_string] = eval_std_len
        eval_statistics['eval_max_length' + deterministic_string] = eval_max_len
        eval_statistics['eval_min_length' + deterministic_string] = eval_min_len

            
        return eval_statistics


class RolloutWorker:
    def __init__(self,
                 env_name,
                 env_args,
                 agent,
                 time_steps,
                 max_path_length,
                 permute_samples,
                 use_data_normalization,
                 replay_buffer_stats_dict,
                 env_tasks,
                 model_path,
                 ):
        
        environment = gym.make(env_name)
        if env_args['use_normalized_env']:
            environment = NormalizedBoxEnv(environment)
        if env_args['non_stationary']:
            environment = NonstationaryEnv(environment)
        self.env = environment
        #! Agent Info
        self.agent = agent
        self.time_steps = time_steps #! n_history
        self.max_path_length = max_path_length
        self.permute_samples = permute_samples
        self.use_data_normalization = use_data_normalization
        self.model_path = model_path

        self.replay_buffer_stats_dict = replay_buffer_stats_dict

        #! env tasks
        self.env_tasks = env_tasks
        self.action_space = self.env.action_space.low.size
        self.obs_space = self.env.observation_space.low.size
        self.context = None
        #! Init Env
        if env_args['non_stationary']:
            self.task_ind = random.randint(0, len(self.env_tasks) - 1)
            self.env.set_task(self.env_tasks[self.task_ind])
        else:
            self.task_ind = -1 
        self.env_args = env_args

        #! Statistics
        self.ep_len_list = []
        self.ep_cumrew_list = []
        self.ep_task_id = []
        self.ep_len,self.ep_cumrew = 0,0.0

    def reset(self,task_ind = None):
        if self.env_args['non_stationary']:
            self.task_ind = random.randint(0, len(self.env_tasks) - 1) if task_ind is None else task_ind 
            self.env.set_task(self.env_tasks[self.task_ind])
        #! reset task and context
        self.state, self.info = self.env.reset()
        self.context = torch.zeros((self.time_steps, self.obs_space + self.action_space + 1 + self.obs_space))

    def step(self,action):
        o2,r,d,info = self.env.step(action)
        #! update env 
        self.update_context(self.state,action,r,o2)
        self.state = o2 
        self.ep_cumrew += r 
        self.ep_len += 1 

        if d:
            self.ep_len_list.append(self.ep_len)
            self.ep_cumrew_list.append(self.ep_cumrew)
            self.ep_task_id.append(self.task_ind)
            self.ep_len,self.ep_cumrew = 0,0.0
            self.reset()
        
        return o2,r,d,info 

    def rollout(self, deterministic=False, max_path_length=np.inf):
        memeory = []
        path_length = 0
        while path_length < max_path_length:
            encoder_input = self.get_encoder_input(self.context,permute=True)
            o = copy.deepcopy(self.state)
            agent_input = self.agent.get_input(o)
            action , task_emb = self.agent.get_model_output(agent_input, encoder_input, deterministic=deterministic)
            action = action.numpy()
            next_o, r, d, env_info = self.env.step(action)

            memeory.append(
                Instance(
                    obs = o,
                    act = action,
                    r = r,
                    obs2 = next_o,
                    done=d,
                    task_emb=task_emb.numpy(),
                    true_task=self.task_ind
                )
            )
        
            if d and max_path_length == np.inf: 
                break
            path_length += 1
        return memeory

    def update_context(self, o, a, r, next_o):
        if self.use_data_normalization and self.replay_buffer_stats_dict is not None:
            stats_dict = self.replay_buffer_stats_dict
            o = torch.from_numpy((o - stats_dict["observations"]["mean"]) / (stats_dict["observations"]["std"] + 1e-9))
            a = torch.from_numpy((a - stats_dict["actions"]["mean"]) / (stats_dict["actions"]["std"] + 1e-9))
            r = torch.from_numpy((r - stats_dict["rewards"]["mean"]) / (stats_dict["rewards"]["std"] + 1e-9))
            next_o = torch.from_numpy((next_o - stats_dict["next_observations"]["mean"]) / (stats_dict["next_observations"]["std"] + 1e-9))
        else:
            o = torch.from_numpy(o)
            a = torch.from_numpy(a)
            r = torch.from_numpy(r)
            next_o = torch.from_numpy(next_o)
        data = torch.cat([o, a, r, next_o]).view(1, -1)
        context = torch.cat([self.context, data], dim=0)
        context = context[-self.time_steps:]
        self.context = context

    def get_encoder_input(self,context, permute = False):
        encoder_input = context.detach().clone()
        if permute:
            perm = torch.LongTensor(torch.randperm(encoder_input.shape[0]))
            encoder_input = encoder_input[perm]
        #! 这里可以再增加 reshape 等操作
        return encoder_input

    def fetch_model(self,path = None):
        path = self.model_path if path is None else path
        self.agent.fetch_model(path)
    
    def get_statistics(self):
        ep_return = copy.deepcopy(self.ep_cumrew_list)
        ep_len = copy.deepcopy(self.ep_len_list)
        ep_task_ids = copy.deepcopy(self.ep_task_id)
        self.ep_cumrew_list = []
        self.ep_len_list = []
        self.ep_task_id = []

        return {
            'return':ep_return,
            'len':ep_len,
            'task_ids':ep_task_ids
        }



@ray.remote(num_cpus=1)
class RemoteRolloutWorker(RolloutWorker):
    def __init__(self, env_name, env_args, agent, time_steps, max_path_length, permute_samples, use_data_normalization, replay_buffer_stats_dict, env_tasks, model_path):
        super().__init__(env_name, env_args, agent, time_steps, max_path_length, permute_samples, use_data_normalization, replay_buffer_stats_dict, env_tasks, model_path)
