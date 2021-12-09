from logging import info
#from CacheSimulator.src.cache_guessing_game_env_impl import CacheGuessingGameEnv
import sys
sys.path.append("..")
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import numpy as np

from typing import Any, Optional, Tuple

from envs.cache_simulator_wrapper import CacheSimulatorWrapper
from rloptim.envs.env import EnvFactory
from rloptim.envs.gym_wrappers import GymWrapper

class CacheSimulatorWrapperFactory(EnvFactory):
    def __init__(self, cfg) -> None:
        super(CacheSimulatorWrapperFactory, self).__init__()
        self.env_config = cfg
        temp_env = CacheGuessingGameEnv(self.env_config)
        temp_env = CacheSimulatorWrapper(temp_env)
        self.action_dim = temp_env.get_act_space_dim() 
        self.obs_dim = temp_env.get_obs_space_dim() 

    def __call__(self, index: int, *args, **kwargs) -> GymWrapper:
        env = CacheGuessingGameEnv(self.env_config)
        self.action_dim = env.get_act_space_dim() 
        self.obs_dim = env.get_obs_space_dim() 
        env = CacheSimulatorWrapper(env)
        env = GymWrapper(env, action_fn=None)
        return env

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_obs_dim(self) -> int:
        return self.obs_dim 
