import os
import sys

from typing import Any, Dict

import numpy as np
import torch

import rlmeta.utils.data_utils as data_utils

from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env, EnvFactory
from rlmeta.envs.gym_wrappers import GymWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
from cyclone_wrapper import CycloneWrapper


class CacheEnvWrapper(GymWrapper):
    def __init__(self,
                 env: CacheGuessingGameEnv,
                 training: bool = True) -> None:
        super().__init__(env)

        self.cache_size = env.cache_size
        self.cache_state_size = 3
        self.null_cache_state = torch.full((self.cache_size, ),
                                           -1,
                                           dtype=torch.int64)
        self.training = training

    @property
    def victim_address(self):
        return self.env.victim_address

    def reset(self, *args, **kwargs) -> TimeStep:
        obs = self._env.reset(*args, **kwargs)
        # victim_address = -1 if self.eval else self.victim_address
        victim_address = self.victim_address if self.training else -1
        obs = (torch.from_numpy(obs), torch.tensor([victim_address]),
               self.null_cache_state)
        return TimeStep(obs, done=False)

    def step(self, action: Action) -> TimeStep:
        act = action.action
        if not isinstance(act, int):
            act = act.item()
        obs, reward, done, info = self._env.step(act)

        v_obs = info.get("valuenet_obs", {})
        # victim_address = -1 if self.eval else v_obs.get(
        #     "victim_addr", self.victim_address)
        victim_address = v_obs.get(
            "victim_addr", self.victim_address) if self.training else -1
        # if not self.eval and "cache_state" in v_obs:
        if self.training and "cache_state" in v_obs:
            cache_state = torch.from_numpy(v_obs["cache_state"].astype(
                np.int64).flatten())
        else:
            cache_state = self.null_cache_state

        obs = (torch.from_numpy(obs), torch.tensor([victim_address]),
               cache_state)
        return TimeStep(obs, reward, done, info)

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self) -> None:
        self.training = False


class CacheEnvWrapperFactory(EnvFactory):
    def __init__(self,
                 env_config: Dict[str, Any],
                 training: bool = True) -> None:
        self._env_config = env_config
        self._training = training

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = CacheGuessingGameEnv(self.env_config)
        # env = GymWrapper(env)
        env = CacheEnvWrapper(env, self._training)
        return env


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


class CacheEnvCycloneWrapperFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        # env = CacheGuessingGameEnv(self.env_config)
        env = CycloneWrapper(self.env_config)
        env = GymWrapper(env)
        return env
