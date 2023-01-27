import copy # NOTE: https://docs.python.org/3/library/copy.html

# provide support for type hints
from typing import Any, Dict, Sequence, Tuple 

# deque: returns a new deque obj initialized left-to-right (using append()) with data from iterable. 
# If iterable is not specified, the new deque is empty
from collections import deque 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gym 
from gym import spaces
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cache_simulator import *
import replacement_policy

''' Defender's observation:  
    self.observation_space = spaces.Discrete(
      [3, # cache_latency (for victim's if invoke_victm, otherwise attaker's)
      20, # domain_id
      2,  # victim_address( if invoke_victim) or attacker_address
      3,  # step_count
      4])  # defender's action
  
    Defender's actions: 
    No of actions matches with the lockbits for n-ways 
     1. unlock a set using lock_bit == 0000, means all way_no are unlocked
     2. lock a set using lock_bit == 0001, means lock way_no = 3
     3. lock a set using lock_bit == 0010, means lock way_no = 2
     ...
     16. lcok a set using lock_bit == 1111, means lock way_no = 0,1,2,3

    Reward function:
     Gets large reward if attacker makes a wrong guess
     Gets large reward if attacker attacker fails to make a guess in episode in time
     Gets large penalty if attacker succeed to guess a victim's secret
     Gets large penalty if attacker succeed to guess a victim no access
     Gets small penalty if opponent_agent is benign
     Gets small penalty every time defender's action other than action = 0 (no locking)

    Starting state:
     fresh cache with nolines
  
    Episode termination:
     when the attacker make a guess TODO: should allow multiple guesses per episode
     when there is length violation
     when there is guess before victim violation
     episode terminates '''
    
class CacheAttackerDefenderEnv(gym.Env):
    #Dict = generic type defined in 'typing' that represents a dict
    def __init__(self, env_config: Dict[str, Any], keep_latency: bool = True) -> None:
        #env_config["cache_state_reset"] = False
        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency # TODO: purpose of this parameter? 
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)
        
        # naming using underscore(_) datacamp.com/tutorial/role-underscore-python
        self._env = CacheGuessingGameEnv(env_config) 
        self.validation_env = CacheGuessingGameEnv(env_config)
        
        # NOTE: needs to be of Gym's class "spaces"
        self.observation_space = self._env.observation_space 
        
        # NOTE: needs to be of Gym's class "spaces"
        self.action_space = spaces.Discrete(16)
        #self.action_space = self._env.action_space 
        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        
        # NOTE: below lines are added to use cache_simulator fuctions
        self.configs = self._env.configs
        self.logger = self._env.logger
        self.hierarchy = self._env.hierarchy
        self.l1 = self._env.l1
        
        # If the "opponent_weights" key is not exist in the "env_config" dict, 
        # [0.5, 0.5] will be assigned to "self.opponent_weights" instead. 
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        
        # random.choices returns a list with a random selection from the given sequence
        # k=1 specifies that only one element is selected from the list based on the weights 
        # [0] index used to extract the selected element
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        
        # If self.opponent_agent is 'attacker' then 'attacker' key will be True
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 5
        
        # NOTE: update deque size 4 to 5
        #self.defender_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        
        self.random_domain = random.choice([0,1]) # Returns a random element from the given sequence
        self.defender_reward_scale = 0.1 

    # API method from gym.Resets the environment to an initial state, required before calling step. 
    # Returns the first agent observation for an episode and info
    def reset(self, victim_address=-1):
        
        """ returned obs = { agent_name : obs } """
        
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        
        # "reset" method resets the environment's random number generator(s) if seed is an integer 
        # or if the environment has not yet initialized a random number generator
        opponent_obs = self._env.reset(victim_address=victim_address, reset_cache_state=True)
        
        self.victim_address = self._env.victim_address
        
        # NOTE: defender will deque size of 5 for observation
        #self.defender_obs = deque([[-1, -1, -1, -1]] * self.max_step)
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        
        self.random_domain = random.choice([0,1])
        obs = {}
        
        # 'defender' key have value as a numpy array of shape (x,y) 
        # w/ values from self.defender_obs. shape (x,5) = x rows, 5 columns
        obs['defender'] = np.array(list(reversed(self.defender_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_defender_obs(self, opponent_obs, opponent_info):
        
        ''' Defender's observation: self.observation_space = spaces.Discrete(
         [3, # cache_latency (for victim's if invoke_victm, otherwise attaker's)
          20, # domain_id
          2,  # victim_address( if invoke_victim) or attacker_address
          3,  # step_count
          4])  # defender's action'''
             
        # NOTE: first element of opponent_obs is 'attacker'?
        # NOTE: increased the column size of cur_opponent_obs to match defender_obs
        # opponent_obs from _env has modified to have row size of 5
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        #print('cur_opponent_obs: ', cur_opponent_obs)
        
        # np.any() returns True if at least 1 element of an array is True 
        # and False if all elements are False. 
        if not np.any(cur_opponent_obs==-1):
            
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                cur_opponent_obs[1] = self.random_domain #1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 1-self.random_domain#0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = self.step_count #0#self.step_count
            
            # NOTE: defender will use its previous actions for observation
            cur_opponent_obs[4] = action['defender']
            
            # adds list "cur_opponent_obs" to the right end of deque "self.defender_obs",
            # then remove the leftmost item from deque, thus deque size is maintained 
            self.defender_obs.append(cur_opponent_obs)
            self.defender_obs.popleft()
        return np.array(list(reversed(self.defender_obs)))

    def compute_reward(self, action, reward, opponent_done, opponent_attack_success=False):
        ''' Reward function:
             Gets large reward if attacker makes a wrong guess
             Gets large reward if attacker attacker fails to make a guess in episode in time
             Gets large penalty if attacker succeed to guess a victim's secret
             Gets large penalty if attacker succeed to guess a victim no access
             Gets small penalty if opponent_agent is benign
             Gets small penalty every time defender's action other than action = no locking '''
            
        action_defender = action['defender']
        defender_success = False
        defender_reward = 0
        
        if action_defender is not None:
            
            ''' reward conditions '''
            # Gets large reward if attacker makes a wrong guess
            # TODO: what is the condition for attacker make a wrong guess??
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == False:
                defender_reward = 20
                defender_success = True
                
            # Gets large reward if attacker attacker fails to make a guess in episode in time
            if self.opponent_agent == 'attacker' and opponent_done:
                defender_reward = 20
                defender_success = True
                
            ''' penalty conditions '''
            # Gets large penalty if attacker succeed to guess a victim's secret
            # attacker episode terminate, and has attacked *successfully*
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == True:
                defender_reward = -20    
        
            # TODO: what is the condition for victim makes no access??
            # in _env, self.allow_empty_victim_access == True
            # Gets large penalty if attacker succeed to guess a victim no access
            
        
            # Gets small penalty if opponent_agent is benign
            if self.opponent_agent == 'benign':
                defender_reward = -2
        
            # Gets small penalty every time defender's action other than action = no locking
            if action_defender != 0:
                defender_reward = -2
        
        attacker_reward = reward['attacker']
        
        reward = {}
        reward['defender'] = defender_reward * self.defender_reward_scale
        reward['attacker'] = attacker_reward
        info = {}
        info['defender_success'] = defender_success
        return reward, info

    # API method from gym. Updates an environment with actions returning the next agent observation, 
    # the reward for taking that actions, if the environment has terminated or truncated
    def step(self, action): 
        
        ''' Defender's actions: No of actions matches with the lockbits for n-ways 
            1. unlock a set using lock_bit == 0000, means all way_no are unlocked
            2. lock a set using lock_bit == 0001, means lock way_no = 3
            3. lock a set using lock_bit == 0010, means lock way_no = 2
            ...
            16. lcok a set using lock_bit == 1111, means lock way_no = 0,1,2,3 '''
        
        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False} # NOTE: why use dunder expression? 
        info = {}
        
        # Attacker update. if key 'info' is not exist in dict "action", returns None
        action_info = action.get('info')
        
        # if action_info is not None, then proceed to the next lines
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            if self.opponent_agent == 'benign' and benign_reset_victim:
                
                # assign benign_victim_addr to victim_address 
                # updates are made to both set_victim methods in _env and here
                self._env.set_victim(benign_victim_addr) 
                self.victim_address = self._env.victim_address
                
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        
        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=True)
            self.victim_address = self._env.victim_address
            
            # NOTE: why reducing the step_count here?
            self.step_count -= 1 # The reset/guess step should not be counted
            
        if self.step_count >= self.max_step:
            defender_done = True
        else:
            defender_done = False
        
        # NOTE: specify the lock_bits matches to the action
        lock_bits = ('0000','0001','0010','0011','0100','0101','0110','0111',
                        '1000','1001','1010','1011','1100','1101','1110','1111')
        set_no = 0
        
        if self.configs['cache_1']['rep_policy'] == 'lru_lock_policy':
            if action["defender"] == 0: 
                lock_bit = lock_bits[0]
            elif action['defender'] == 1:    
                lock_bit = lock_bits[1]
            elif action['defender'] == 2:
                lock_bit = lock_bits[2]
            elif action['defender'] == 3:
                lock_bit = lock_bits[3]
            elif action['defender'] == 4:
                lock_bit = lock_bits[4]
            elif action['defender'] == 5:
                lock_bit = lock_bits[5]
            elif action['defender'] == 6:
                lock_bit = lock_bits[6]
            elif action['defender'] == 7:    
                lock_bit = lock_bits[7]
            elif action['defender'] == 8:
                lock_bit = lock_bits[8]
            elif action['defender'] == 9:
                lock_bit = lock_bits[9]
            elif action['defender'] == 10:
                lock_bit = lock_bits[10]
            elif action['defender'] == 11:
                lock_bit = lock_bits[11]
            elif action['defender'] == 12:
                lock_bit = lock_bits[12] 
            elif action['defender'] == 13:    
                lock_bit = lock_bits[13]
            elif action['defender'] == 14:
                lock_bit = lock_bits[14]
            else:
                lock_bit = lock_bits[15]
            
            self.l1.lock(self, set_no, lock_bit)
            defender_done = False # will not terminate the episode
            
        # attacker
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = defender_done #Figure out correctness
        info['attacker'] = opponent_info
        
        #benign
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = defender_done #Figure out correctness
        info['benign'] = opponent_info
        opponent_attack_success = opponent_info.get('defender_success', False)

        # obs, reward, done, info 
        updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['defender'] = updated_reward['defender']
        obs['defender'] = self.get_defender_obs(opponent_obs, opponent_info) 
        done['defender'] = defender_done
        info['defender'] = {"defender_success":updated_info["defender_success"], "is_guess":bool(action['defender'])}
        info['defender'].update(opponent_info)
        
        # Change the criteria to determine wether the game is done
        if defender_done:
            done['__all__'] = True

        info['__all__'] = {'action_mask':self.action_mask}
    
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["defender"])
        return obs, reward, done, info

# to indicate that following lines should only be executed 
# if the script is run directly, and not imported as a module into another script.
if __name__ == '__main__':
    env = CacheAttackerDefenderEnv({})
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
                  'defender':np.random.randint(low=0, high=15)} 
        obs, reward, done, info = env.step(action)
        print("step: ", i)
        print("observation of defender: ", obs['defender'])
        print("action: ", action)
        #print("victim: ", env.victim_address, env._env.victim_address)
        #print("done:", done)
        print("reward:", reward)
        #print('env.victim_address_min, max: ', env.victim_address_min, env.victim_address_max)
        print("info:", info )
        #if info['attacker'].get('invoke_victim'):
        #    print('info[attacker]: ', info['attacker'])
      obs = env.reset()
      done = {'__all__':False}
