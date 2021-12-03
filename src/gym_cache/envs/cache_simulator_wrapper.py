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

#class CacheSimulatorWrapperFactory(EnvFactory):
#    def __init__(self,
#     length_violation_reward=-10000,
#     double_victim_access_reward=-10,
#     correct_reward=200,
#     wrong_reward=-9999,
#     step_reward=-1) -> None:
#        super(CacheSimulatorWrapperFactory, self).__init__()
#        self.length_violation_reward =length_violation_reward
#        self.double_victim_access_reward = double_victim_access_reward
#        self.correct_reward = correct_reward
#        self.wrong_reward = wrong_reward
#        self.step_reward = step_reward
#
#    def __call__(self, index: int, *args, **kwargs) -> GymWrapper:
#        env = CacheGuessingGameEnvFix(self.length_violation_reward, 
#         self.double_victim_access_reward,
#         self.correct_reward,
#         self.wrong_reward,
#         self.step_reward)
#        env = CacheSimulatorWrapper(env)
#        env = GymWrapper(env, action_fn=None)
#        return env
#