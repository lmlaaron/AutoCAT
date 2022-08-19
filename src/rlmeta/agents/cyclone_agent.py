import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gym

from sklearn import svm
from sklearn.model_selection import cross_val_score
from cache_guessing_game_env_impl import CacheGuessingGameEnv
import pickle
import os

class CycloneAgent:
    def __init__(self,
                 env_config,
                 svm_model_path="all.txt.svm.txt",
                 keep_latency=True) -> None:

        self.step_count = 0
        self.episode_length = 64
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.svm_model_path = svm_model_path
        self.clf = pickle.load(open(svm_model_path,'rb'))
        
        self.cyclone_window_size = env_config.get("cyclone_window_size", 4)
        self.cyclone_interval_size = env_config.get("cyclone_interval_size", 40)
        self.cyclone_num_buckets = env_config.get("cyclone_num_buckets", 4)
        self.cyclone_bucket_size = self.env_config.cache_configs.cache_1.blocks / self.cyclone_num_buckets
        self.cyclone_collect_data = env_config.get("cyclone_collect_data", False)
        self.cyclone_malicious_trace = env_config.get("cyclone_malicious_trace", False)
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)
    
    def cyclone_attack(self, cyclone_counters):
        for i in range(len(cyclone_counters)):
            self.cyclone_heatmap[i] += cyclone_counters[i]

        if self.cyclone_collect_data == True:
            x = np.array(cyclone_counters).reshape(-1)
            if self.cyclone_malicious_trace == True:
                y = 1
            else:
                y = 0
            self.X.append(x)
            self.Y.append(y)
        x = np.array(cyclone_counters).reshape(-1)
        y = self.clf.predict([x])[0]
        if y >= 0.5:
            return 1
        return 0

    def observe_init(self, timestep):
        self.local_step = 0
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)

    def act(self, timestep):
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            self.local_step = 0
             self.cyclone_counters = []
            for j in range(self.cyclone_num_buckets):
                temp =[]
                for i in range(self.cyclone_window_size):
                    temp.append(0)
                self.cyclone_counters.append(temp)

        self.local_step += 1
        cur_step_obs = timestep.observation[0][0]
        info = timestep.info
        if "cyclic_set_index" in info.keys() and info["cyclic_set_index"] != -1: 
            set = int(info["cyclic_set_index"])
            if self.local_step < self.episode_length:
                self.cyclone_counters[int(set / self.cyclone_bucket_size) ][int(self.step_count / self.cyclone_interval_size) ] += 1    

        if self.local_step >= self.episode_length: 
            action = self.cyclone_attack(self.cyclone_counters)
        else:
            action = 0

        return action, info


    def observe(self):
        return 
