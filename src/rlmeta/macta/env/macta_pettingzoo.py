import os
import copy
import sys

from typing import Any, Dict, Sequence, Tuple
from collections import deque

# from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
# from pettingzoo.test import parallel_api_test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import hydra
import gym
from gymnasium import spaces
# from gym import spaces
from gymnasium.spaces import Discrete, Box 

from .cache_guessing_game_env import CacheGuessingGameEnv

from omegaconf.omegaconf import open_dict
sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_simulator import *
import replacement_policy



class CacheAttackerDetectorEnv(AECEnv):

    metadata = {"name": "macta_defender"}
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True,
                 ) -> None:
        #env_config["cache_state_reset"] = False
        self.agents = ['opponent', 'detector']
        self.possible_agents = self.agents[:]
        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        self._env = CacheGuessingGameEnv(env_config)
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": self._env.observation_space,
        #             "action_mask": spaces.Box(low=0, high=1, shape=(self._env.action_space.n,), dtype=np.int8),
        #         }
        #     )
        #     for i in self.agents
        # }
        
        self.action_spaces = {'opponent': self._env.action_space,
                            'detector' : spaces.Discrete(2)}
        self.observation_spaces = {'opponent': self._env.observation_space,
                                    'detector': self._env.observation_space}
       


        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        self.action_mask = {
            'detector': True,
            'attacker': self.opponent_agent == 'attacker',
            'benign': self.opponent_agent == 'benign'
        }
        self.step_count = 0
        self.max_step = 64
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0, 1])
        self.detector_reward_scale = 0.1  #1.0

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):      
        return self.observation_spaces[agent]

    def observe(self, agent):
        return np.array(self.observations[agent])
        # return self.observations[agent]
    
    def reset(self, victim_address=-1, seed=None, options=None):

       
        """
        returned obs = { agent_name : obs }
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        # print("inside reset -->  the opponent agent is :  ", self.opponent_agent, "   ", self.opponent_weights)
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True)
        self.victim_address = self._env.victim_address
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        # self.observations = {agent: deque([[-1, -1, -1, -1]] * self.max_step) for agent in self.agents}
        self.observations = {}
        self.observations['detector'] = np.array(list(reversed(self.detector_obs)))
        self.observations['opponent'] = opponent_obs
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        # info = {}
        # return self.observations, self.infos
    
    def get_detector_obs(self, opponent_obs, opponent_info):
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        if not np.any(cur_opponent_obs == -1):
            # Make sure the observation does not leak information for detector
            # TODO should we include step number? - yes we should - the guess step should not be observed by the detector
            # attacker obs: r, victim_accessed, original action, current step
            # detector obs: r, domain_id, memory address, 0
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                cur_opponent_obs[1] = self.random_domain  #1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 1 - self.random_domain  #0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = self.step_count  #0#self.step_count
            self.detector_obs.append(cur_opponent_obs)
            self.detector_obs.popleft()
        return np.array(list(reversed(self.detector_obs)))

    def compute_reward(self, agent, action, reward, opponent_done, opponent_attack_success=False):
        attacker_reward = reward['opponent']
        detector_reward = 0
        detector_correct = False
        if agent == 'detector':
            detector_flag = False
            # detector_correct = False

            if action == 1:
                detector_flag = True
                # terminate the episode
                # detector flag the opponent as an attacker
                if self.opponent_agent == 'benign':
                    #detector_reward = - self.max_step + self.step_count - 1 #punish false positive
                    detector_reward = -10 * self.max_step
                elif self.opponent_agent == 'attacker':
                    detector_reward = max(self.max_step - self.step_count, 0)
                    detector_correct = True
            else:
                # else receive a timestep penalty
                if self.opponent_agent == 'benign':
                    detector_reward = 0  # no reward if benign agent
                    detector_correct = True
                else:
                    detector_reward = 0  # no reward if attacker has done nothing meaningful

            if action == 0 and self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
                # attacker episode terminate, and has attacked *successfully*
                detector_reward = -20
            elif action == 0 and self.opponent_agent == 'attacker' and opponent_done:
                # attacker episode terminates, but has done nothing successfully
                detector_reward = 0

            # attacker_reward = reward['attacker']

            if action == 1 and detector_correct:
                # the attacker should not receive as much reward if being detected
                attacker_reward -= 10
            #else:
            #    attacker_reward += 0.1

        rew = {}
        rew['detector'] = detector_reward * self.detector_reward_scale
        rew['opponent'] = attacker_reward

        info = {}
        info['guess_correct'] = detector_correct
        return rew, info

    def close(self):
        pass

    def render(self):
        pass

    def step(self, action):
        agent = self.agent_selection
        #if self.step_count <=8: action['detector']=0
        # print("action:  ", action, "   agent: ", agent, "   step:", self.step_count)
        # Attacker update
        # action_info = action.get('info')
        # #if self.opponent_agent == 'benign':
        # if action_info:
        #     benign_reset_victim = action_info.get('reset_victim_addr', False)
        #     benign_victim_addr = action_info.get('victim_addr', None)
        #     if self.opponent_agent == 'benign' and benign_reset_victim:
        #         self._env.set_victim(benign_victim_addr)
        #         self.victim_address = self._env.victim_address
        if agent == "opponent":
            opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action)
        if agent == "detector":
            opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.get_infos()


        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=True)
            self.victim_address = self._env.victim_address
            self.step_count -= 1  # The reset/guess step should not be counted
        if self.step_count >= self.max_step:
            detector_done = True
        else:
            detector_done = False
        if agent == "detector":
            if action == 1:  # Raise Hard Flag
                detector_done = True  # Terminate the episode
        # attacker
        self.observations['opponent'] = opponent_obs
        self.rewards['opponent'] = opponent_reward
        self.terminations['opponent'] = detector_done  #Figure out correctness
        self.infos['opponent'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info
        updated_reward, updated_info = self.compute_reward(agent, action, self.rewards, opponent_done, opponent_attack_success)
        self.rewards['opponent'] = updated_reward['opponent']
        self.rewards['detector'] = updated_reward['detector']
        self.terminations['detector'] = detector_done
        if agent == "detector":
            self.observations['detector'] = self.get_detector_obs(opponent_obs, opponent_info)
            self.infos['detector'] = {
                "guess_correct": updated_info["guess_correct"],
                "is_guess": bool(action)
            }
            self.step_count += 1
        self.infos['detector'].update(opponent_info)
        # Change the criteria to determine wether the game is done
        # if detector_done:
        #     done['__all__'] = True
        # #from IPython import embed; embed()

        # info['__all__'] = {'action_mask': self.action_mask}

        # for k, v in info.items():
        #     info[k].update({'action_mask': self.action_mask})
        self.agent_selection = self._agent_selector.next()
        # print(self.rewards)
        # print(self.observations)
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
        
@hydra.main(config_path="../config", config_name="macta")
def main(cfg):
    env = CacheAttackerDefenderEnv(cfg.env_config)
    env.opponent_weights = [0,1]
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    
    action_space = env.action_space
    env.reset()
    done = {'__all__':False}
    prev_a = 10
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print("probably the simulation is over : ", termination)
            action = None
            break
        else:
            action = env.action_space(agent).sample()
            # this is where you would insert your policy
            # action = {'attacker':np.random.randint(low=4, high=11), #np.random.randint(low=9, high=11),
            #           'benign':np.random.randint(low=0, high=3),
            #           'detector': np.random.randint(low=0, high=255)}
        env.step(action)
    env.close()
    # env.opponent_weights = [1,0]
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         print("probably the simulation is over : ", termination)
    #         action = None
    #         break
    #     else:
    #         if agent == 'attacker': 
    #             action = np.random.randint(low=4, high=11)
    #         elif agent == 'benign' : 
    #             action = np.random.randint(low=0, high=3)
    #         elif agent == 'detector': 
    #             action = np.random.randint(low=0, high=255)
    #         else:
    #             action = 0
    #         # this is where you would insert your policy
    #         # action = {'attacker':np.random.randint(low=4, high=11), #np.random.randint(low=9, high=11),
    #         #           'benign':np.random.randint(low=0, high=3),
    #         #           'detector': np.random.randint(low=0, high=255)}
    #     env.step(action)
    # env.close()

    # print("check defender action: ", detector_action, type(detector_action))
    # print("checking the action space:  ", env.defender_action_space, type(env.defender_action_space))
    # for k in range(1):
    #     i = 0
    #     while not done['__all__']:
    #         i += 1
    #         print("step: ", i)
    #         action = {'attacker':9 if (prev_a==10 or i<65) else 10, #np.random.randint(low=9, high=11),
    #                   'benign':np.random.randint(low=2, high=5),
    #                   'detector': np.random.randint(low=0, high=255)} #generate 8 bit random numbers to represent lock bits
    #         prev_a = action['attacker']
    #         obs, reward, done, info = env.step(action)
    #         #print("obs: ", obs['detector'])
    #         print("action: ", action)
    #         print("victim: ", env.victim_address, env._env.victim_address)

    #         #print("done:", done)
    #         print("reward:", reward)
    #         #print(env.victim_address_min, env.victim_address_max)
    #         #print("info:", info )
    #         if info['attacker'].get('invoke_victim') or info['attacker'].get('is_guess')==True:
    #             print(info['attacker'])
    #     obs, infos = env.reset()
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
