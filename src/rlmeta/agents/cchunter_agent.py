import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gym


class CCHunterAgent:
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True) -> None:
        self.local_step = 0
        self.cc_hunter_history = []
        self.cc_hunter_check_length = 8
        self.threshold = 0.9998
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            assert(self.num_ways == 1) # currently only support direct-map cache
            assert(flush_inst == False) # do not allow flush instruction
            assert(attacker_addr_e - attacker_addr_s == victim_addr_e - victim_addr_s ) # address space must be shared
            #must be no shared address space
            assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)

    # initialize the agent with an observation
    def observe_init(self, timestep):
        self.local_step = 0
        self.cc_hunter_history = []
    
    def autocorr(self, x: np.ndarray, p: int) -> float:
        if p == 0:
            return 1.0
        mean = x.mean()
        var = max(x.var(), 1e-12)
        return ((x[:-p] - mean) * (x[p:] - mean)).mean() / var

    def act(self, timestep):
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            self.local_step = 0
            self.cc_hunter_history = []
        self.local_step += 1
        cur_step_obs = timestep.observation[0][0]
        info = timestep.info
        latency = cur_step_obs[0] #if self.keep_latency else -1
        victim_access = cur_step_obs[1]
        #MUlong Luo
        # change the semantics of cc_hunter_history following the paper
        # only append when there is a conflict miss (i.e., victim_latency is 1(miss))
        # then check the action
        # if the action is attacker access, then it is T->S append 1
        # else if the action is trigger victim, then it is S->T append 0
        if victim_access:
            print(info)
        if "victim_latency" in info.keys() and info["victim_latency"] == 1:
            self.cc_hunter_history.append(0)
        elif latency == 1:
            self.cc_hunter_history.append(1)

        n = min(len(self.cc_hunter_history), self.cache_size * self.cc_hunter_check_length
                )  # Mulong: only calculate 4 * size_cache size lag

        x = np.asarray(self.cc_hunter_history)
        corr = [self.autocorr(x, i) for i in range(n)]
        corr = np.asarray(corr[1:])
        corr = np.nan_to_num(corr)
        mask = corr > self.threshold
        rew = -np.square(corr).mean() if len(corr) > 0 else 0.0

        cnt = mask.sum()
        if cnt >=1 and cur_step_obs[-1]>=64:
            action = 1, info
        else:
            action = 0, info
        # np.set_printoptions(suppress=True)
        # print(f"data = {np.asarray(data)}")
        # print(f"corr_arr = \n{corr}")
        # print(f"corr_std = {corr.std()}")
        # print(f"corr_max = {corr.max()}")
        # print(f"corr_min = {corr.min()}")
        # print(f"threshold = {self.threshold}")
        # print(f"mask = {mask.astype(np.int64)}")
        # print(f"cc_hunter_rew = {rew}")
        # print(f"cc_hunter_cnt = {cnt}")

        return action
    
    def observe(self, action, timestep):
        return
