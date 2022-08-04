# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict

import random

console = Console()


class SpecAgent(Agent):

    def __init__(self,
                 env_config,
                 trace):
        super().__init__()
        
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
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
        '''
        spec_trace_f = open(trace,'r')
        spec_trace = spec_trace_f.read().split('\n')
        y = []
        for line in spec_trace:
            line = line.split()
            y.append(line)
        spec_trace = y
        '''
        self.trace = trace
        self.trace_length = len(self.trace)
        line = self.trace[0]
        self.domain_id_0 = line[0]
        self.domain_id_1 = line[0]
        local_step = 0
        while len(line) > 0:
            local_step+=1
            line = self.trace[local_step]
            self.domain_id_1 = line[0]
            if self.domain_id_1 != self.domain_id_0:
                break
        self.start_idx = random.randint(0, self.trace_length-1)
        self.step = 0

    def act(self, timestep: TimeStep) -> Action:
        line = self.trace[(self.start_idx+self.step) % self.trace_length]
        if self.step >= self.trace_length:
            self.step = 0
        else:
            self.step += 1
        if len(line) == 0:
            action = self.cache_size
            addr = 0#addr % self.cache_size
            info={"file_done" : True}
            return Action(action)
        domain_id = line[0]
        addr = int( int(line[3], 16) / 4 )
        action = addr % self.cache_size
        if domain_id == self.domain_id_0: # attacker access
            action = addr % self.cache_size
            info ={}
        else: # domain_id = self.domain_id_1: # victim access
            action = self.cache_size
            addr = addr % self.cache_size
            info={"reset_victim_addr": True, "victim_addr": addr}
        return Action(action)

    async def async_act(self, timestep: TimeStep) -> Action:
        line = self.trace[(self.start_idx+self.step) % self.trace_length]
        if self.step >= self.trace_length:
            self.step = 0
        else:
            self.step += 1
        if len(line) == 0:
            action = self.cache_size
            addr = 0#addr % self.cache_size
            info={"file_done" : True}
            return Action(action)
        domain_id = line[0]
        addr = int( int(line[3], 16) / 4 )
        action = addr % self.cache_size
        if domain_id == self.domain_id_0: # attacker access
            action = addr % self.cache_size
            info ={}
        else: # domain_id = self.domain_id_1: # victim access
            action = self.cache_size
            addr = addr % self.cache_size
            info={"reset_victim_addr": True, "victim_addr": addr}
        return Action(action)


    async def async_observe_init(self, timestep: TimeStep) -> None:
        pass

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        pass

    def update(self) -> None:
        pass
    
    async def async_update(self) -> None:
        pass



