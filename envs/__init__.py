from gym.envs.registration import register
from envs.grid_world_general import RandomGridWorldPlat
register(
id='GridWorldPlat-v2', entry_point=RandomGridWorldPlat
)