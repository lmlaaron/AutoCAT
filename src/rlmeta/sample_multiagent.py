import os
import logging
import sys
from typing import Dict

import hydra
#import torch
#import torch.nn

#import rlmeta.utils.nested_utils as nested_utils

#from rlmeta.agents.ppo.ppo_agent import PPOAgent
#from agents.ppo_agent import PPOAgent
#from agents.spec_agent import SpecAgent
from agents.prime_probe_agent import PrimeProbeAgent
from agents.evict_reload_agent import EvictReloadAgent
from agents.flush_reload_agent import FlushReloadAgent
from agents.random_agent import RandomAgent
#from agents.benign_agent import BenignAgent
#from rlmeta.core.types import Action
#from rlmeta.envs.env import Env
#from rlmeta.utils.stats_dict import StatsDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache_attacker_detector import CacheAttackerDetectorEnv
#from cache_env_wrapper import CacheAttackerDetectorEnvFactory
#from cache_ppo_model import CachePPOModel
#from cache_ppo_transformer_model import CachePPOTransformerModel

from typing import Any, NamedTuple, Optional, Union
class Action(NamedTuple):
    action: Any
    info: Optional[Any] = None



def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    #info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env, agents, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0
    
    env.opponent_weights = [0,1]
    done = {'__all__': False}
    if victim_addr == -1:
        obs = env.reset()
    else:
        obs = env.reset(victim_address=victim_addr)
    print("victim address: ", env.victim_address ) 
    
    while not done["__all__"]:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            #timestep[agent_name].observation.unsqueeze_(0)
            #print("attacker obs")
            #print(timestep["attacker"].observation)
            action = agent.act(obs[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = action[0]
            #if not isinstance(action.action, int):
            #    action = unbatch_action(action)
            if isinstance(action, Action):
                action=action.action
            actions.update({agent_name:action})
        print(actions)
        obs, reward, done, info = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], obs[agent_name])
        
        episode_length += 1
        episode_return += reward['attacker']
        if done["__all__"] and actions['detector']==1:
            detector_count += 1
        detector_accuracy = detector_count

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "detector_accuracy": detector_accuracy,
    }

    return metrics


def run_loops(env,
              agent,
              num_episodes: int,
              seed: int = 0):
    env.seed(seed)
    metrics = StatsDict()
    if env._env.allow_empty_victim_access == False:
        end_address = env._env.victim_address_max + 1
    else:
        end_address = env._env.victim_address_max + 1 + 1
    '''
    for victim_addr in range(env.env._env.victim_address_min, end_address):
        cur_metrics = run_loop(env, agent, victim_addr=victim_addr)
        metrics.extend(cur_metrics)
    '''
    for i in range(num_episodes):
        cur_metrics = run_loop(env, agent, victim_addr=-1)
        metrics.extend(cur_metrics)
    return metrics


@hydra.main(config_path="./config", config_name="sample_multiagent")
def main(cfg):
    # Create env
    cfg.env_config['verbose'] = 1
    env_fac = CacheAttackerDetectorEnv(cfg.env_config)
    env = env_fac

    # Create agent
    attacker_agent = EvictReloadAgent(cfg.env_config)
    #attacker_agent = PrimeProbeAgent(cfg.env_config)
    #attacker_agent = FlushReloadAgent(cfg.env_config)
    
    detector_agent = RandomAgent(1)
    benign_agent = RandomAgent(2)
    agents = {"attacker": attacker_agent, "detector": detector_agent, "benign": benign_agent}
    # Run loops
    metrics = run_loops(env, agents, cfg.num_episodes, cfg.seed)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


import json
import math

from typing import Dict, Optional

from tabulate import tabulate


class StatsItem:

    def __init__(self,
                 key: Optional[str] = None,
                 val: Optional[float] = None) -> None:
        self._key = key
        self.reset()
        if val is not None:
            self.add(val)

    @property
    def key(self) -> str:
        return self._key

    def reset(self):
        self._m0 = 0
        self._m1 = 0.0
        self._m2 = 0.0

        self._min_val = float("inf")
        self._max_val = float("-inf")

    def add(self, v: float) -> None:
        # Welford algorithm.
        self._m0 += 1
        delta = v - self._m1
        self._m1 += delta / self._m0
        self._m2 += delta * (v - self._m1)

        self._min_val = min(self._min_val, v)
        self._max_val = max(self._max_val, v)

    def count(self) -> int:
        return self._m0

    def mean(self) -> float:
        return self._m1

    def var(self, ddof: int = 0) -> float:
        return self._m2 / (self._m0 - ddof)

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.var(ddof))

    def min(self) -> float:
        return self._min_val

    def max(self) -> float:
        return self._max_val

    def dict(self) -> Dict[str, float]:
        ret = {
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "count": self.count(),
        }
        if self.key is not None:
            ret["key"] = self.key
        return ret


class StatsDict:

    def __init__(self) -> None:
        self._dict = {}

    def __getitem__(self, key: str) -> StatsItem:
        return self._dict[key]

    def reset(self):
        self._dict.clear()

    def add(self, k: str, v: float) -> None:
        if k in self._dict:
            self._dict[k].add(v)
        else:
            self._dict[k] = StatsItem(k, v)

    def extend(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            self.add(k, v)

    def update(self, stats) -> None:
        self._dict.update(stats._dict)

    def dict(self) -> Dict[str, float]:
        return {k: v.dict() for k, v in self._dict.items()}

    def json(self, info: Optional[str] = None, **kwargs) -> str:
        data = self.dict()
        if info is not None:
            data["info"] = info
        data.update(kwargs)
        return json.dumps(data)

    def table(self, info: Optional[str] = None, **kwargs) -> str:
        if info is None:
            head = ["key", "mean", "std", "min", "max", "count"]
        else:
            head = ["info", "key", "mean", "std", "min", "max", "count"]

        data = []
        for k, v in self._dict.items():
            if info is None:
                row = [k, v.mean(), v.std(), v.min(), v.max(), v.count()]
            else:
                row = [info, k, v.mean(), v.std(), v.min(), v.max(), v.count()]
            data.append(row)
        for k, v in kwargs.items():
            if info is None:
                row = [k, v, 0.0, v, v, 1]
            else:
                row = [info, k, v, 0.0, v, v, 1]
            data.append(row)

        return tabulate(data,
                        head,
                        numalign="right",
                        stralign="right",
                        floatfmt=".8f")

if __name__ == "__main__":
    main()
