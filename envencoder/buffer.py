import copy
from collections import namedtuple
import random
import numpy as np
import pickle
import time
import sys,os 
import torch 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tuplenames = ('obs', 'act','obs2','rew','done', 'task_id', 'env_param')
Transition = namedtuple('Transition', tuplenames)

class Memory(object):
    """
    ### Description

        Store transitions from one env
        no need to have complete trajectory
    """
    def __init__(self ) -> None:
        self.memory = []
    def clear(self):
        self.memory = []
    def push(self,*args):
        self.memory.append(Transition(*args))
    def append(self,new_memory):
        assert self.task_id == new_memory.task_id
        self.memory += new_memory.memory
        
    def __len__(self):
        return len(self.memory)
    @property
    def size(self):
        return len(self)
    @property
    def task_id(self):
        assert len(self.memory) > 0
        return self.memory[0].task_id[0]

class Buffer(object):
    """
    ### Description 
        Middle-Level data management, 
        store trajectories in 1 env
        ready to be sampled 
    """
    def __init__(self,max_traj_num = 1000,max_traj_step=1050) -> None:
        self.tmp_memory = [] 
        self.max_buffer_size = max_traj_num * max_traj_step 
        self.max_traj_num = max_traj_num
        self.max_traj_step = max_traj_step
        self.memory_buffer = None 
        self.ind_range = None 
        self.task_id = -1 
        self.clear()
    def __len__(self):
        return self._size
    def clear_tmp(self):
        self.tmp_memory = [] 
    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._episode_ends = []
        self._cur_episode_start = 0
    
    def build_buffer(self,memory:list):
        start_dim = 0
        self.ind_range = []
        end_dim = 0
        for item in memory[0]:
            if item is None:
                dim = 0
            else:
                # print(item,item.shape)
                dim = item.shape[-1]
            end_dim = start_dim + dim 
            self.ind_range.append(list(range(start_dim,end_dim)))
            start_dim = end_dim

        self.task_id = memory[0].task_id
        self.memory_buffer = np.zeros((self.max_buffer_size,end_dim))
        self.clear()

    def flush_memory(self,memory:list):
        ## put a new trajectory in the memory buffer 
        for ind , transition in enumerate(memory):
            self.memory_buffer[self._top] = self.transition_to_array(transition)
            self._top =  (self._top + 1) % self.max_buffer_size

        self._size = min(self._size + len(memory),self.max_buffer_size)
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top
        self._episode_ends.append(self._cur_episode_start)

    def push_mem(self,memory:Memory):
        ## could be single step or whole trajectories
        for item in memory.memory:
            self.memory_buffer[self._top] = self.transition_to_array(item)
            self._top = (self._top + 1)% self.max_buffer_size
            self._size = min(self._size + 1,self.max_buffer_size)
            if item.done:
                self._episode_starts.append(self._cur_episode_start)
                self._cur_episode_start = self._top
                self._episode_ends.append(self._cur_episode_start)
    def push_transition(self,*args):
        trans = Transition(*args)
        self.tmp_memory += trans
        if trans.done:
            self.flush_memory(self.tmp_memory)
            self.tmp_memory = []
    def sample_batch(self,batch_size = None,with_tensor = True,device = 'cpu'):
        if batch_size is not None:
            indices = np.random.randint(0,self._size,batch_size)
        else:
            indices = list(range(self._size)) 
        data = self.memory_buffer[indices]
        if with_tensor:
            data = torch.from_numpy(data).to(device).float()
        return self.array_to_transition(data)
    
    def sample_query(self,batch_size,M_to_predict,with_tensor = True,device = 'cpu'):
        all_points = self.all_points(M_to_predict)
        indices = np.random.choice(all_points, batch_size, replace=True if batch_size > all_points.shape[0] else False)
        all_indices = []
        for ind in indices:
            all_indices += list(range(ind-M_to_predict,ind)) 
        data = self.memory_buffer[all_indices]
        data = data.reshape(batch_size,M_to_predict,-1)
        if with_tensor:
            data = torch.from_numpy(data).to(device).float()
        query = self.array_to_transition(data)
        return query

    def _sample_batch_helper(self,batch_size = None):
        if batch_size is not None:
            indices = np.random.randint(0,self._size,batch_size)
        else:
            indices = list(range(self._size)) 
        return self.memory_buffer[indices]
    

    def sample_support(self,bz,n_support,M_to_predict,with_tensor = True,device = 'cpu'):
        all_points = self.all_points(n_support)
        indices = np.random.choice(all_points, bz, replace=True if bz > all_points.shape[0] else False)
        all_indices = []
        for ind in indices:
            all_indices += list(range(ind-n_support,ind+M_to_predict)) 
        data = self.memory_buffer[all_indices]
        if with_tensor:
            data = torch.from_numpy(data).to(device).float()
        data = data.reshape(bz,n_support + M_to_predict,-1) # else data.shape(n_shot+1,dim)
        support_data = data[:,0:n_support,:]
        query_data = data[:,n_support:,:]
        
        support_data = self.array_to_transition(support_data) # (bz,n_support,dim)
        query_data = self.array_to_transition(query_data) # (bz,M_to_predcit,dim)
        return support_data,query_data

    def sample_support_query(self,bz,n_shot,with_tensor = True,device = 'cpu'):
        all_points = self.all_points(n_shot)
        indices = np.random.choice(all_points, bz, replace=True if bz > all_points.shape[0] else False)
        all_indices = []
        for ind in indices:
            all_indices += list(range(ind-n_shot,ind+1)) 
        data = self.memory_buffer[all_indices]
        if with_tensor:
            data = torch.from_numpy(data).to(device).float()

        if bz > 1:
            data = data.reshape(bz,n_shot+1,-1) # else data.shape(n_shot+1,dim)
            support_data = data[:,0:n_shot,:]
            query_data = data[:,n_shot:,:]
        else:
            support_data = data[0:n_shot,:] #(n_shot,dim)
            query_data = data[n_shot:,:] #(1,dim)
        ## n_shot for emb, 1 for query

        support_data = self.array_to_transition(support_data)
        query_data = self.array_to_transition(query_data)
        return support_data,query_data
    

    def transition_to_array(self,transition):
        res = []
        for item in transition:
            res.append(item.reshape(1,-1))
        res = np.hstack(res)
        assert res.shape[-1] == self.memory_buffer.shape[-1]
        return res 
    
    def array_to_transition(self,data):
        ## data.shape = (batch_size,n_history,dim)
        data_list = []
        for item in self.ind_range:
            if len(item) > 0:
                start = item[0]
                end = item[-1] + 1 
                data_list.append(data[...,start:end]) 
            else:
                data_list.append(None)
        return Transition(*data_list)
    def all_points(self,minimum_samples):
        res = list(range(minimum_samples + 1,self._size))
        res = np.array(res)
        return res
    @property
    def num_traj(self):
        return len(self._episode_starts)
    @property
    def num_trans(self):
        return self._size

    
class MetaBuffer(object):
    """
    ### Description

    High-level data management
    Collect buffers from different envs 
    Provide interface to trainer 
    The least unit is Memory
    """
    def __init__(self,max_traj_num = 1000,max_traj_step=1050) -> None:
        self.max_traj_num = max_traj_num
        self.max_traj_step = max_traj_step
        self.task_buffers = dict()
        self.task_id_list = []
    
    def __len__(self):
        l = [len(buf) for buf in self.task_buffers.values()]
        return min(l)
    @property
    def size(self):
        res = 0
        for id in self.task_id_list:
            res += self.task_buffers[id]._size
        return res

    @property
    def num_envs(self):
        return len(self.task_buffers.keys())
    
    def push_mem(self,mem:Memory):
        ind = mem.task_id
        if ind not in self.task_buffers.keys():
            self.task_buffers[ind] = Buffer(self.max_traj_num,self.max_traj_step)
            self.task_buffers[ind].build_buffer(mem.memory)
            self.ind_range = self.task_buffers[ind].ind_range
            self.task_id_list.append(ind)
        self.task_buffers[ind].push_mem(mem)
    def sample_multi_task_batch(self,task_inds,batch_size,with_tensor = True,device = 'cpu'):
        data_list = [self.task_buffers[i]._sample_batch_helper(batch_size) for i in task_inds ]
        data = np.stack(data_list,axis = 0)
        if with_tensor:
            data = torch.from_numpy(data).to(device).float()
        return self.array_to_transition(data) ### shape (n_env,batch_size,dim)

    def sample_batch(self,id,batch_size,with_tensor = True,device = 'cpu'):
        assert id in self.task_id_list
        return self.task_buffers[id].sample_batch(batch_size,with_tensor = with_tensor,device = device)
    def sample_query(self,id,batch_size,M_to_predict,with_tensor = True,device = 'cpu'):
        assert id in self.task_id_list
        return self.task_buffers[id].sample_query(batch_size,M_to_predict,with_tensor = with_tensor,device = device)
    def sample_support(self,id,bz,n_support,M_to_predict,with_tensor ,device):
        assert id in self.task_id_list
        return self.task_buffers[id].sample_support(bz,n_support,M_to_predict,with_tensor ,device )
    def sample_support_query(self,id,bz,n_shot,with_tensor ,device):
        assert id in self.task_id_list
        return self.task_buffers[id].sample_support_query(bz,n_shot,with_tensor ,device )

    def sample_task_id(self,num_task):
        valid_list = []
        for id in self.task_id_list:
            if len(self.task_buffers[id]) > 50:
                valid_list.append(id)
        return np.random.choice(valid_list,num_task,True)

    def num_steps_can_sample(self,id):
        assert id in self.task_id_list
        return self.task_buffers[id].num_trans
    def num_traj_can_sample(self,id):
        assert id in self.task_id_list
        return self.task_buffers[id].num_traj

    def transition_to_array(self,transition):
        res = []
        for item in transition:
            res.append(item.reshape(1,-1))
        res = np.hstack(res)
        assert res.shape[-1] == self.memory_buffer.shape[-1]
        return res 
    
    def array_to_transition(self,data):
        ## data.shape = (batch_size,n_history,dim)
        data_list = []
        for item in self.ind_range:
            if len(item) > 0:
                start = item[0]
                end = item[-1] + 1 
                data_list.append(data[...,start:end]) 
            else:
                data_list.append(None)
        return Transition(*data_list)


def data_gen(id,d = False):
    s,a,s2 = np.random.rand(4),np.random.rand(1),np.random.rand(4)
    r = np.array(1.0).reshape(1,)
    d = np.array(d ).reshape(1,)
    task_id =np.array( id ).reshape(1,)
    env_para = np.random.rand(2)
    return s,a,s2,r,d,task_id,env_para

def mem_gen(id):
    mem = Memory()
    for _ in range(10):
        s,a,s2,r,d,task_id,env_para = data_gen(id)
        mem.push(s,a,s2,r,d,task_id,env_para)
    s,a,s2,r,d,task_id,env_para = data_gen(id,True)
    mem.push(s,a,s2,r,d,task_id,env_para)
    return mem
