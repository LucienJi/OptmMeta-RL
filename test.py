from legged_gym.envs import * 
from legged_gym.utils import get_args, task_registry


if __name__ == '__main__':
    env,env_cfg = task_registry.make_env(name="littledog_terrain", args=None)
    print("env type: ",type(env))
    print("Observation space is", env.observation_space)
    print("Action space is", env.action_space)	