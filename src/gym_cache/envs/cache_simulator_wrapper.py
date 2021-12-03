from logging import info
#from CacheSimulator.src.cache_guessing_game_env_impl import CacheGuessingGameEnv
import sys
sys.path.append("..")
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import numpy as np

import gym

from typing import Any, Optional, Tuple

#from rloptim.envs.env import EnvFactory
#from rloptim.envs.gym_wrappers import GymWrapper
from ray.rllib.models.preprocessors import OneHotPreprocessor

class CacheSimulatorWrapper(gym.Env):
    def __init__(self, env: CacheGuessingGameEnv):
        self._env = env
        self.preprocessor = OneHotPreprocessor(env.observation_space)

    def reset(self):
        self._obs = self.preprocessor.transform(self._env.reset()) #flatten_multisiscrete( self._env.observation_space, self._env.reset()) 
        return self._obs.flatten().astype(np.float32)

    def step(self, action: int):
        obs, reward, done, info = self._env.step( np.array([action]) )
        self._obs = self.preprocessor.transform(obs)
        return self._obs.flatten().astype(np.float32), reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        return self._env.seed(seed)
