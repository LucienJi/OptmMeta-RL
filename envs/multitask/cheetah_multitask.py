from .half_cheetah import HalfCheetahEnv
import numpy as np

class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self, vel = 1.0, max_len = 1000):
        self._vel = vel
        self._min_vel = np.array([0.0])
        self._max_vel = np.array([5.0])

        self._max_len = max_len
        self.step_ct = 0
        super(HalfCheetahVelEnv, self).__init__()

    def set_vel(self,vel):
        self._vel = vel

    def step(self, action):
        self.step_ct += 1
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = self.step_ct > self._max_len
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost)
        return observation, reward, done, infos
    def reset(self):
        self.step_ct = 0
        return super().reset()
        
    @property
    def vel(self):
        return self._vel
    @property
    def min_vel(self):
        return self._min_vel
    @ property
    def max_vel(self):
        return self._max_vel
    

class HalfCheetahDirEnv(HalfCheetahEnv):
    def __init__(self, dir= 1, max_len = 1000):
        self._dir = dir 
        self._min_dir = np.array([-1])
        self._max_dir = np.array([1])

        self._max_len = max_len
        self.step_ct = 0 
        super(HalfCheetahDirEnv, self).__init__()
    def set_dir(self,dir):
        assert dir in [-1,1]
        self._dir = dir 

    def step(self, action):
        self.step_ct += 1 
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = self.step_ct > self._max_len
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost)
        return observation, reward, done, infos

    def reset(self):
        self.step_ct = 0
        super().reset()
    @property
    def dir(self):
        return self._dir
    
    @ property
    def min_dir(self):
        return self._min_dir
    @ property
    def max_dir(self):
        return self._max_dir