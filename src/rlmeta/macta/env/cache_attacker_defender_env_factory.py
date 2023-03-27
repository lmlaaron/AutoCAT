import os
import sys

from typing import Any, Dict
from rlmeta.envs.env import Env, EnvFactory

# from .cache_attacker_detector_env import CacheAttackerDetectorEnv
from .cache_attacker_defender_env import CacheAttackerDefenderEnv
from utils.gym_wrappers import GymWrapper, MAGymWrapper

'''
class CacheAttackerDetectorEnvFactory(EnvFactory):  # MACTA
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheAttackerDetectorEnv(self.env_config)
        env = MAGymWrapper(env)
        return env
        
'''


class CacheAttackerDefenderEnvFactory(EnvFactory):  # RLdefense
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheAttackerDefenderEnv(self.env_config)
        env = MAGymWrapper(env)
        return env
