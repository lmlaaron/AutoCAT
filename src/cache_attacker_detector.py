import copy

from typing import Any, Dict, Sequence, Tuple
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gym

from cache_guessing_game_env_impl import CacheGuessingGameEnv


class CacheAttackerDetectorEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any],
                 keep_latency: bool = True) -> None:
        #env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
       
        self.opponent_agent = random.choices(['benign','attacker'], weights=[0.5,0.5], k=1)[0] 
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 64
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)

    def reset(self, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        self.opponent_agent = random.choices(['benign','attacker'], weights=[0.5,0.5], k=1)[0]
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True)
        self.victim_address = self._env.victim_address
        detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)

        obs = {}
        obs['detector'] = np.array(list(reversed(detector_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_detector_obs(self, opponent_obs, opponent_info):
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        if not np.any(cur_opponent_obs==-1):
            # TODO should we include step number?
            # attacker obs: r, victim_accessed, original action, current step
            # detector obs: r, domain_id, memory address, 0
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[1] = 1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = 0
            self.detector_obs.append(cur_opponent_obs)
            self.detector_obs.popleft()
        return np.array(list(reversed(self.detector_obs)))

    def compute_reward(self, action, reward, opponent_done, opponent_attack_success=False):
        action_detector = action['detector']
        action_attacker = action['attacker']
        
        # determine detector's reward 
        detector_flag = False
        detector_correct = False
        if action_detector == 1:
            detector_flag = True
            # detector flag the opponent as an attacker
            if self.opponent_agent == 'benign':
                detector_reward = -1
            elif self.opponent_agent == 'attacker':
                detector_reward = 1
                detector_correct = True
        else:
            # else receive a timestep penalty
            if self.opponent_agent == 'benign':
                detector_reward = 1
                detector_correct = True
            else:
                detector_reward = -1
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
                # attacker has attacked *successfully*
                detector_reward = -20
        
        attacker_reward = reward['attacker']
        
        # determine detector's reward
        if detector_correct:
            attacker_reward -= 0.1
        else:
            attacker_reward += 0.1
        
        rew = {}
        rew['detector'] = detector_reward
        rew['attacker'] = attacker_reward
        return rew

    def step(self, action):
        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}

        # Attacker update
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=True)
            self.victim_address = self._env.victim_address
        if self.step_count > self.max_step:
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
        updated_reward = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['detector'] = updated_reward['detector']
        obs['detector'] = self.get_detector_obs(opponent_obs, opponent_info) 
        done['detector'] = detector_done
        info['detector'] = {"guess_correct":reward['detector']>0.5, "is_guess":bool(action['detector'])}
        
        # Change the criteria to determine wether the game is done
        if detector_done:
            done['__all__'] = True
        
        info['__all__'] = {'action_mask':self.action_mask}
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        return obs, reward, done, info

if __name__ == '__main__':
    env = CacheAttackerDetectorEnv({})
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__':False}
    i = 0
    while not done['__all__']:
        i += 1
        action = {'attacker':np.random.randint(low=3, high=5),
                  'benign':np.random.randint(low=0, high=1),
                  'detector':np.random.randint(low=0, high=2)}
        obs, reward, done, info = env.step(action)
        print("step: ", i)
        print("obs: ", obs['attacker'], obs['detector'])
        print("action: ", action)
        print("victim: ", env.victim_address, env._env.victim_address)

        #print("done:", done)
        print("reward:", reward)
        #print("info:", info )
        if info['attacker'].get('invoke_victim'):
            print(info['attacker'])
