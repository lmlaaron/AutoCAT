from logging import info
#from CacheSimulator.src.cache_guessing_game_env_impl import CacheGuessingGameEnv
import sys
sys.path.append("..")
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import numpy as np

import gym

from typing import Any, Optional, Tuple

from ray.rllib.models.preprocessors import OneHotPreprocessor
from gym import spaces

class CacheSimulatorWrapper(gym.Env):
    def __init__(self, env: CacheGuessingGameEnv):
        self._env = env
        self.preprocessor = OneHotPreprocessor(self._env.observation_space)
    
    ##def __init__(self, env_config : dict = {}) -> None:#, env: CacheGuessingGameEnv):
    ##    self._env = CacheGuessingGameEnv(env_config)#env
    ##    self.preprocessor = OneHotPreprocessor(self._env.observation_space)#OneHotPreprocessor(env.observation_space)
    ##    self.action_space = self._env.action_space
    ##    obs_len =len( self.preprocessor.transform(self._env.reset()) )
    ##    self.observation_space = spaces.Box(low = np.array([-1] * obs_len), high = np.array([2]* obs_len))
   
    def reset(self):
        self._obs = self.preprocessor.transform(self._env.reset()) #flatten_multisiscrete( self._env.observation_space, self._env.reset()) 
        return self._obs.flatten().astype(np.float32)
    def step(self, action: int):
        obs, reward, done, info = self._env.step( np.array([action]) )
        self._obs = self.preprocessor.transform(obs)
        if reward > 0:                      #when winning positive reward it must be done
            assert(done == True)
        return self._obs.flatten().astype(np.float32), reward, done, info
    def seed(self, seed: Optional[int] = None) -> None:
        return self._env.seed(seed)

    def get_obs_space_dim(self):
        return len( self.preprocessor.transform(self._env.reset()) )

    def get_act_space_dim(self):
        return len(self._env.attacker_address_space) * 2 * 2 * 2 * len(self._env.victim_address_space)
