import numpy as np
from gym.envs.mujoco import AntEnv as AntEnv_

class AntEnv(AntEnv_):
    def _get_obs(self):
        return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])