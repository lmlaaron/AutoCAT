import os
import copy
import sys
import time

from typing import Any, Dict, Sequence, Tuple
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import hydra
import gym
from gym import spaces
from .cache_guessing_game_env import CacheGuessingGameEnv

from omegaconf.omegaconf import open_dict
sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_simulator import *
import replacement_policy




class CacheAttackerDefenderEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True,
                 ) -> None:
        #env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        # self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        # self.action_space = spaces.Discrete(2**self._env.cache_size)
        #ToDO self.defender_action_space = 
        # self.defender_action_space = spaces.Discrete(2**self._env.cache_size)
        self.max_box_value = max(2**self._env.cache_size, self._env.max_box_value)#max(self.window_size + 2, len(self.attacker_address_space) + 1) 
        self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self._env.window_size, 6))



        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 64
        self.detector_obs = deque([[-1, -1, -1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        self.detector_reward_scale = 1.0 #0.1

        # if "cache_configs" in env_config:
        #     self.configs = env_config["cache_configs"]
        # else:
        #     self.config_file_name = os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1'
        #     self.config_file = open(self.config_file_name)
        #     self.logger.info('Loading config from file ' + self.config_file_name)
        #     self.configs = yaml.load(self.config_file, yaml.CLoader)
        # self.vprint(self.configs)

    def reset(self, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        current = time.time()
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True, seed = int(current))
        validation_opponent_obs = self.validation_env.reset(victim_address=victim_address,
                                       reset_cache_state=True, seed = int(current))
        self.victim_address = self._env.victim_address
        self.detector_obs = deque([[-1, -1, -1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        obs = {}
        obs['defender'] = np.array(list(reversed(self.detector_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_detector_obs(self, opponent_obs, opponent_info):
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        cur_obs = [cur_opponent_obs[0],-1,-1,-1,-1,-1]
        # TODO add the set,way instead of obs[2] 
        if not np.any(cur_opponent_obs==-1):
            if opponent_info.get('invoke_victim'):
                cur_obs[0] = opponent_info['victim_latency']
                cur_obs[1] = self.random_domain #1
            else:
                cur_obs[1] = 1-self.random_domain#0
            cur_obs[3] = self.step_count #0#self.step_count
            binary_string = ''.join([str(bit) for bit in self._env.l1.get_locked_bits()])
            cur_obs[2] = int(binary_string, 2)
            cur_obs[4] = opponent_info['way_index']
            cur_obs[5] = opponent_info['set_index']
            self.detector_obs.append(cur_obs)
            self.detector_obs.popleft()
        return np.array(list(reversed(self.detector_obs)))
        # return self.detector_obs
        # return cur_opponent_obs

    def compute_reward(self, action, reward, opponent_done, obs_for_reward1, obs_for_reward2):#, opponent_attack_success=False):

        bin_rep = bin_rep = format(int(action["defender"]), '04b')  
        detector_action = str(bin_rep)
        action_detector = detector_action
        # action_attacker = action['attacker']
 
        # detector_flag = False
        detector_correct = False
        detector_reward = 0

        #compute negative reward for locking lines
        locking_reward = 0
        if not (self.opponent_agent == 'attacker' and opponent_done):
            for digit in action_detector:
                if int(digit) == 1:
                    locking_reward += 0.00000000000001
        
        
            # for i in range(len(obs_for_reward1)):
                # if not np.any(obs_for_reward1[i]==-1):
            if obs_for_reward1[0][0] == obs_for_reward2[0][0]:
                detector_reward += 1
            else:
                detector_reward += -6

        # if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
        #     # attacker episode terminate, and has attacked *successfully*
        #     detector_reward = -10
        # elif self.opponent_agent == 'attacker' and opponent_done:
        #     # attacker episode terminates, but has done nothing successfully
        #     detector_reward = 20
        
        attacker_reward = reward['attacker']

        rew = {}
        rew['defender'] = (detector_reward + locking_reward) * self.detector_reward_scale
        rew['attacker'] = attacker_reward

        info = {}
        info['guess_correct'] = detector_correct
        return rew, info

    def step(self, action):

        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}
        action_info = action.get('info')

        if action["defender"] > 15 :
            detector_action = '0000'
        else:
            # this line and the next are used to simulate a random loker agent
            # locked_line1 = random.choices(['0','1'], weights=[0.5, 0.5], k=1)[0]
            # locked_line2 = random.choices(['0','1'], weights=[0.5, 0.5], k=1)[0]
            # locked_line3 = random.choices(['0','1'], weights=[0.5, 0.5], k=1)[0]
            # locked_line4 = random.choices(['0','1'], weights=[0.5, 0.5], k=1)[0]
            # detector_action = locked_line1 + locked_line2 + locked_line3 + locked_line4
            bin_rep = format(int(action["defender"]), '04b') #need only this for the correct attack-defender  
            detector_action = str(bin_rep)
        self._env.l1.detector_func(detector_action)
        self.validation_env.l1.detector_func(detector_action)

        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            if self.opponent_agent == 'benign' and benign_reset_victim:
                self._env.set_victim(benign_victim_addr) 
                self.victim_address = self._env.victim_address

        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        va = self._env.get_victim()
        if va == 1:
            self.validation_env.set_victim(victim_address=0)
        else:
            self.validation_env.set_victim(victim_address=1)
        opponent_obs_test, opponent_reward_test, opponent_done_test, opponent_info_test = self.validation_env.step(action[self.opponent_agent])

        obs_for_reward1 = opponent_obs
        obs_for_reward2 = opponent_obs_test

        if opponent_done:
            current = time.time()
            opponent_obs = self._env.reset(reset_cache_state=True, seed = current)
            validation_opponent_obs = self.validation_env.reset(reset_cache_state=True, seed = current)
            self.victim_address = self._env.victim_address
            self.step_count -= 1 # The reset/guess step should not be counted
        if self.step_count >= self.max_step:
            detector_done = True
        else:
            detector_done = False
        # attacker
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = detector_done #Figure out correctness
        info['attacker'] = opponent_info
        
        #benign
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = detector_done #Figure out correctness
        info['benign'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info 
        updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, obs_for_reward1, obs_for_reward2)#, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['defender'] = updated_reward['defender']

        obs['defender'] = self.get_detector_obs(opponent_obs, opponent_info) 
        done['defender'] = detector_done
        info['defender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['defender'])}
        info['defender'].update(opponent_info)
        # Change the criteria to determine wether the game is done
        if detector_done:
            done['__all__'] = True
        #from IPython import embed; embed()

        info['__all__'] = {'action_mask':self.action_mask}
    
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["detector"])
        # print_cache(self._env.l1)
        self._env.print_sample_multiagent(obs, reward, done, info, self.opponent_agent)
        return obs, reward, done, info

@hydra.main(config_path="../config", config_name="macta")
def main(cfg):
    locked_lines1 = random.choices(['0','1'], weights=[0.3, 0.7], k=1)[0]
    locked_lines2 = random.choices(['0','1'], weights=[0.3, 0.7], k=1)[0]
    print(locked_lines1+locked_lines2, "    ", type(locked_lines1+locked_lines2))
    env = CacheAttackerDefenderEnv(cfg.env_config)
    env.opponent_weights = [0,1]
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__':False}
    prev_a = 10
    test_action = format(random.randint(0, 15), '04b')
    bin_rep = format(15, '04b') 
    detector_action = str(bin_rep)
    print("check defender action: ", detector_action, type(detector_action))
    for k in range(1):
        i = 0
        while not done['__all__']:
            i += 1
            print("step: ", i)
            action = {'attacker':9 if (prev_a==10 or i<64) else 10, #np.random.randint(low=9, high=11),
                      'benign':np.random.randint(low=2, high=5),
                      'defender': np.random.randint(low=0, high=255)} #generate 8 bit random numbers to represent lock bits
            prev_a = action['attacker']
            print("actions:  ", action)
            obs, reward, done, info = env.step(action)
            #print("obs: ", obs['detector'])
            print("action: ", action)
            # print("victim: ", env.victim_address, env._env.victim_address)

    #         #print("done:", done)
    #         # print("reward:", reward)
    #         #print(env.victim_address_min, env.victim_address_max)
    #         #print("info:", info )
    #         if info['attacker'].get('invoke_victim') or info['attacker'].get('is_guess')==True:
    #             # print(info['attacker'])
    #             pass
    #     obs = env.reset()
    #     done = {'__all__':False}







def print_cache(cache):
    # Print the contents of a cache as a table
    # If the table is too long, it will print the first few sets,
    # break, and then print the last set
    table_size = 5
    ways = [""]
    sets = []
    set_indexes = sorted(cache.data.keys())
    
    if len(cache.data.keys()) > 0:
        way_no = 0

        # Label the columns
        for way in range(cache.associativity):
            ways.append("Way " + str(way_no))
            way_no += 1

        # Print either all the sets if the cache is small, or just a few
        # sets and then the last set
        sets.append(ways)
        if len(set_indexes) > table_size + 4 - 1:
            for s in range(min(table_size, len(set_indexes) - 4)):
                set_ways = cache.data[set_indexes[s]].keys()
                temp_way = ["Set " + str(s)]
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[s]][w].address)
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

            for i in range(3):
                temp_way = ['.']
                for w in range(cache.associativity):
                    temp_way.append('')
                sets.append(temp_way)

            set_ways = cache.data[set_indexes[len(set_indexes) - 1]].keys()
            temp_way = ['Set ' + str(len(set_indexes) - 1)]
            for w in range(0, cache.associativity):
                temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w][1].address)
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w].address)
            sets.append(temp_way)
            
        else:
            for s in range(len(set_indexes)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

                # add additional rows only if the replacement policy = lru_lock_policy
                if cache.rep_policy == lru_lock_policy:
                    lock_info = ["Lock bit"]

                    lock_vector_array = cache.set_rep_policy[set_indexes[s]].lock_vector_array

                    for w in range(0, len(lock_vector_array)):
                        lock_info.append(lock_vector_array[w])
                    sets.append(lock_info)

                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                    sets.append(timestamp)
                elif cache.rep_policy == new_plru_pl_policy: # add a new row to the table to show the lock bit in the plru_pl_policy cache
                    lock_info = ["Lock bit"]

                    lockarray = cache.set_rep_policy[set_indexes[s]].lockarray

                    for w in range(0, len(lockarray)):
                        if lockarray[w] == 2:
                            lock_info.append("unlocked")
                        elif lockarray[w] == 1:
                            lock_info.append("locked")
                        elif lockarray[w] == 0:
                            lock_info.append("unknown")
                        else:
                            lock_info.append(lockarray[w])
                    sets.append(lock_info)
                elif cache.rep_policy == lru_policy:  # or cache.rep_policy == lru_lock_policy:
                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                            
                    sets.append(timestamp)
                    # print(timestamp)

        table = UnixTable(sets)
        table.title = cache.name
        table.inner_row_border = True
        print(table.table)
        return set_indexes



if __name__ == "__main__":
    main()