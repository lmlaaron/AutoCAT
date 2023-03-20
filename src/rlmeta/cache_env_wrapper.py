import os
import sys

from typing import Any, Dict

from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrappers import GymWrapper, MAGymWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from cache_guessing_game_env_impl import CacheGuessingGameEnv
#from cchunter_wrapper import CCHunterWrapper
#from cyclone_wrapper import CycloneWrapper
#from cache_attacker_detector import CacheAttackerDetectorEnv

# new gym env for RLdefense
from cache_guessing_game_env_defense import AttackerCacheGuessingGameEnv
from cache_attacker_defender import CacheAttackerDefenderEnv

class CacheEnvWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        # new gym env for RLdefense
        #env = CacheGuessingGameEnv(self.env_config)
        env = AttackerCacheGuessingGameEnv(self.env_config) # NOTE
        env = GymWrapper(env)
        return env

'''
class CacheEnvCCHunterWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        # env = CacheGuessingGameEnv(self.env_config)
        env = CCHunterWrapper(self.env_config)
        env = GymWrapper(env)
        return env
'''
'''
class CacheEnvCycloneWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheGuessingGameEnv(self.env_config)
        env = CycloneWrapper(self.env_config)
        env = GymWrapper(env)
        return env
'''
'''
class CacheAttackerDetectorEnvFactory(EnvFactory):
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

class CacheAttackerDefenderEnvFactory(EnvFactory):
    # new gym env for RLdefense
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheAttackerDefenderEnv(self.env_config)
        env = MAGymWrapper(env)
        return env