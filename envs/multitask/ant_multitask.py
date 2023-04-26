import numpy as np 
from .ant import AntEnv

class AntGoalEnv(AntEnv):
    def __init__(self,vel = -1.0,goal = -1.0,max_step = 1000):
        self._vel = vel 
        self._goal = goal 

        self._min_vel = np.array([0.0])
        self._max_vel = np.array([5.0])
        self._min_goal = np.array([0.0])
        self._max_goal = np.array([np.pi])
        self._max_step = max_step 
        self.step_ct = 0 
        super().__init__()
    
    def set_vel(self,vel):
        self._vel = vel
    def set_goal(self,goal):
        self._goal = goal
    def reset(self):
        self.step_ct = 0
        return super().reset()
        
    def step(self,action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_velocity = torso_velocity[:2]/self.dt

        if self._goal > 0.0:
            forward_reward = np.dot(forward_velocity, direct)
        else:
            forward_reward = 0.0 
        
        if self._vel > 0.0:
            velocity_reward = -1.0 * abs(forward_velocity - self._vel)
        else:
            velocity_reward = 0.0 

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = velocity_reward + forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )
    
    @property
    def vel(self):
        return self._vel
    @property
    def min_vel(self):
        return self._min_vel
    @ property
    def max_vel(self):
        return self._max_vel

    @property
    def goal(self):
        return self._goal
    @property
    def min_goal(self):
        return self._min_goal
    @ property
    def max_goal(self):
        return self._max_goal