from logging import info
#from CacheSimulator.src.cache_guessing_game_env_impl import CacheGuessingGameEnv
import sys
sys.path.append("..")
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import numpy as np

import gym

from typing import Any, Optional, Tuple


from envs.cache_simulator_wrapper import CacheSimulatorWrapper
from rloptim.envs.env import EnvFactory
from rloptim.envs.gym_wrappers import GymWrapper

class CacheSimulatorWrapperFactory(EnvFactory):
    def __init__(self,
     length_violation_reward=-10000,
     double_victim_access_reward=-10,
     correct_reward=200,
     wrong_reward=-9999,
     step_reward=-1) -> None:
        super(CacheSimulatorWrapperFactory, self).__init__()
        self.length_violation_reward =length_violation_reward
        self.double_victim_access_reward = double_victim_access_reward
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.step_reward = step_reward

    def __call__(self, index: int, *args, **kwargs) -> GymWrapper:
        env = CacheGuessingGameEnv(self.length_violation_reward, 
         self.double_victim_access_reward,
         self.correct_reward,
         self.wrong_reward,
         self.step_reward)
        env = CacheSimulatorWrapper(env)
        env = GymWrapper(env, action_fn=None)
        return env

