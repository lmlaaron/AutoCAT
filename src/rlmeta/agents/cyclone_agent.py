import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gym

from sklearn import svm
from sklearn.model_selection import cross_val_score
import pickle
import os

class CycloneAgent:
    def __init__(self,
                 env_config,
                 svm_model_path=None,
                 keep_latency=True,
                 mode='data') -> None:

        self.step_count = 0
        self.episode_length = 64
        self.keep_latency = keep_latency
        self.mode = mode # 'data' or 'active'
        self.env_config = env_config
        if svm_model_path is not None:
            print("loading cyclone agent from ", svm_model_path)
            self.clf = pickle.load(open(svm_model_path,'rb'))
        
        self.cyclone_window_size = env_config.get("cyclone_window_size", 4)
        self.cyclone_interval_size = env_config.get("cyclone_interval_size", 16)
        self.cyclone_num_buckets = env_config.get("cyclone_num_buckets", 4)
        self.cyclone_bucket_size = env_config.cache_configs.cache_1.blocks / self.cyclone_num_buckets
        self.cyclone_collect_data = env_config.get("cyclone_collect_data", False)
        self.cyclone_malicious_trace = env_config.get("cyclone_malicious_trace", False)
        self.cyclone_heatmap = [[], [], [], []]
        self.cyclone_counters = []
        for j in range(self.cyclone_num_buckets):
            temp =[]
            for i in range(self.cyclone_window_size):
                temp.append(0)
            self.cyclone_counters.append(temp)
    
    def cyclone_attack(self, cyclone_counters):
        for i in range(len(cyclone_counters)):
            self.cyclone_heatmap[i] += cyclone_counters[i]
        #print("cyclone heatmap",self.cyclone_heatmap)
        if self.cyclone_collect_data == True:
            x = np.array(cyclone_counters).reshape(-1)
            if self.cyclone_malicious_trace == True:
                y = 1
            else:
                y = 0
            self.X.append(x)
            self.Y.append(y)
        x = np.array(cyclone_counters).reshape(-1)
        #print(x)
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

        cur_step_obs = timestep.observation[0][0]
        self.local_step = max(cur_step_obs[-1],0)
        info = timestep.info
        if "cyclic_set_index" in info.keys() and info["cyclic_set_index"] != -1: 
            set = int(info["cyclic_set_index"])
            if self.local_step <= self.episode_length: #"<= or <"?
                #print(info)
                self.cyclone_counters[int(set / self.cyclone_bucket_size) ][int(self.step_count / self.cyclone_interval_size) ] += 1    

        if timestep.observation[0][0][-1] >= self.episode_length-1 and self.mode=='active': 
            action = self.cyclone_attack(self.cyclone_counters)
        else:
            action = 0

        return action, info


    def observe(self, action, timestep):
        return


if __name__ == "__main__":
    CycloneAgent(env_config={})
