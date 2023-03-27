import os
import sys

from typing import Any, Dict

from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrappers import GymWrapper

# for MACTA
# from .cache_guessing_game_env import CacheGuessingGameEnv

# for RLdefense
from .cache_guessing_game_def_env import AttackerCacheGuessingGameEnv

'''
class CacheEnvWrapperFactory(EnvFactory):  # MACTA
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheGuessingGameEnv(self.env_config) 
        env = GymWrapper(env)
        return env
'''


class CacheEnvWrapperFactory(EnvFactory):  # RLdefense
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = AttackerCacheGuessingGameEnv(self.env_config)
        env = GymWrapper(env)
        return env
