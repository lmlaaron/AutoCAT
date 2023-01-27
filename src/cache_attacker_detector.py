import copy # NOTE: how it works here? https://docs.python.org/3/library/copy.html

# provide support for type hints
# https://typing.readthedocs.io/en/latest/source/libraries.html#why-provide-type-annotations
from typing import Any, Dict, Sequence, Tuple 

# Returns a new deque object initialized left-to-right (using append()) with data from iterable. 
# If iterable is not specified, the new deque is empty
from collections import deque 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gym # https://gymnasium.farama.org/api/env/

from cache_guessing_game_env_impl import CacheGuessingGameEnv


class CacheAttackerDetectorEnv(gym.Env):
    def __init__(self,
                 env_config: Dict[str, Any], #Dict = generic type defined in 'typing' that represents a dict
                 keep_latency: bool = True,
                 ) -> None:
        #env_config["cache_state_reset"] = False

        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency # NOTE: purpose of this parameter? 
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)

        # NOTE: why use underbar expression here? 
        self._env = CacheGuessingGameEnv(env_config)
        
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space # The Space object corresponding to valid observations
        self.action_space = self._env.action_space # # The Space object corresponding to valid actions, 

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        
        # dictionary.get(keyname, value)
        # If the "opponent_weights" key is not exist in the "env_config" dict, 
        # [0.5, 0.5] will be assigned to "self.opponent_weights" instead. 
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        
        # k=1 specifies that only one element is selected from the list based on the weights 
        # [0] index used to extract the selected element
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        
        # If the self.opponent_agent is 'attacker' then value for 'attacker' key will be True
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 64
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        self.detector_reward_scale = 0.1 #1.0

    # API method from gym.Resets the environment to an initial state, required before calling step. 
    # Returns the first agent observation for an episode and information,
    def reset(self, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        
        # "reset" method resets the environment's random number generator(s) if seed is an integer 
        # or if the environment has not yet initialized a random number generator
        opponent_obs = self._env.reset(victim_address=victim_address, reset_cache_state=True)
        self.victim_address = self._env.victim_address
        self.detector_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        obs = {}
        
        # 'detector' key will have value which a numpy array of shape (64, 4) with values from self.detector_obs
        # shape (64, 4) = 64 rows, 4 columns
        obs['detector'] = np.array(list(reversed(self.detector_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_detector_obs(self, opponent_obs, opponent_info):
        
        # NOTE: copy the first element of opponent_obs using deepcopy?
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        
        # np.any() returns True if at least one element of an array is True 
        # and False if all elements are False.
        # https://numpy.org/doc/stable/reference/generated/numpy.any.html
        if not np.any(cur_opponent_obs==-1):
            
            # Make sure the observation does not leak information for detector
            # TODO should we include step number? - yes we should - the guess step should not be observed by the detector
            # attacker obs: r, victim_accessed, original action, current step
            # detector obs: r, domain_id, memory address, 0
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                cur_opponent_obs[1] = self.random_domain #1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 1-self.random_domain#0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = self.step_count #0#self.step_count
            
            # adds list "cur_opponent_obs" to the right end of deque "self.detector_obs",
            # then remove the leftmost item from deque
            # thus the size of deque is maintained 
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
            # terminate the episode #NOTE: which line is used? look for line where self._env.reset(...)
            # detector flag the opponent as an attacker
            if self.opponent_agent == 'benign':
                #detector_reward = - self.max_step + self.step_count - 1 #punish false positive
                detector_reward = - 10 * self.max_step
            elif self.opponent_agent == 'attacker':
                detector_reward = max(self.max_step - self.step_count, 0)
                detector_correct = True
        else:
            # else receive a timestep penalty
            if self.opponent_agent == 'benign':
                detector_reward = 0 # no reward if benign agent
                detector_correct = True
            else:
                detector_reward = 0 # no reward if attacker has done nothing meaningful
        

        if action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
            # attacker episode terminate, and has attacked *successfully*
            detector_reward = -20
        elif action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done:
            # attacker episode terminates, but has done nothing successfully
            detector_reward = 0
        

        attacker_reward = reward['attacker']
        
        if action_detector == 1 and detector_correct:
            # the attacker should not receive as much reward if being detected
            attacker_reward -= 10 
        #else:
        #    attacker_reward += 0.1
        
        rew = {}
        rew['detector'] = detector_reward * self.detector_reward_scale
        rew['attacker'] = attacker_reward

        info = {}
        info['guess_correct'] = detector_correct
        return rew, info

    # API method from gym. Updates an environment with actions returning the next agent observation, 
    # the reward for taking that actions, if the environment has terminated or truncated
    def step(self, action): 
        self.step_count += 1
        #if self.step_count <=8: action['detector']=0
        obs = {}
        reward = {}
        done = {'__all__':False} # NOTE: why use dunder expression? 
        info = {}
        
        # Attacker update
        # if the value of key 'info' is not exist in dict "action" returns None
        action_info = action.get('info')
        #if self.opponent_agent == 'benign':
        
        # if action_info is not None, then proceed to the next lines
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            if self.opponent_agent == 'benign' and benign_reset_victim:
                
                # assign benign_victim_addr to victim_address 
                # updates are made to both set_victim methods in _env and here
                self._env.set_victim(benign_victim_addr) 
                self.victim_address = self._env.victim_address
                
        #NOTE: Need to check with _env
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        
        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=True)
            self.victim_address = self._env.victim_address
            
            # NOTE: why reducing the step_count here?
            self.step_count -= 1 # The reset/guess step should not be counted
        if self.step_count >= self.max_step:
            detector_done = True
        else:
            detector_done = False
        if action["detector"] == 1: # Raise Hard Flag
            detector_done = True # Terminate the episode
            
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
        updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['detector'] = updated_reward['detector']
        obs['detector'] = self.get_detector_obs(opponent_obs, opponent_info) 
        done['detector'] = detector_done
        info['detector'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['detector'])}
        info['detector'].update(opponent_info)
        
        # Change the criteria to determine wether the game is done
        if detector_done:
            done['__all__'] = True
        #from IPython import embed; embed()

        info['__all__'] = {'action_mask':self.action_mask}
    
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["detector"])
        return obs, reward, done, info

# to indicate that following lines should only be executed 
# if the script is run directly, and not imported as a module into another script.
if __name__ == '__main__':
    env = CacheAttackerDetectorEnv({})
    env.opponent_weights = [0.5, 0.5] #[0,1]
    
    action_space = env.action_space 
    obs = env.reset()
    done = {'__all__':False}
    i = 0
    for k in range(2):
      while not done['__all__']:
        i += 1
        action = {'attacker':np.random.randint(low=3, high=6),
                  'benign':np.random.randint(low=2, high=5),
                  'detector':np.random.randint(low=0, high=1)}
        obs, reward, done, info = env.step(action)
        print("step: ", i)
        print("observation of detector: ", obs['detector'])
        print("action: ", action)
        print("victim: ", env.victim_address, env._env.victim_address)
        #print("done:", done)
        print("reward:", reward)
        print('env.victim_address_min, max: ', env.victim_address_min, env.victim_address_max)
        
        #print("info:", info )
        if info['attacker'].get('invoke_victim'):
            print('info[attacker]: ', info['attacker'])
      obs = env.reset()
      done = {'__all__':False}
