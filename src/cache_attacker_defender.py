import copy
from typing import Any, Dict, Sequence, Tuple 
from collections import deque 
#import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import random
import gym 
from gym import spaces
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cache_simulator import *
import replacement_policy
import hydra

class CacheAttackerDefenderEnv(gym.Env):

    def __init__(self, env_config: Dict[str, Any], keep_latency: bool = True) -> None:
        #env_config["cache_state_reset"] = False
        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency # NOTE: purpose of this parameter? 
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)
        
        self._env = CacheGuessingGameEnv(env_config) 
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space 
        self.action_space = spaces.Discrete(16)
        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        #self.configs = self._env.configs
        self.logger = self._env.logger
        self.hierarchy = self._env.hierarchy
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 5
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1]) # Returns a random element from the given sequence
        self.defender_reward_scale = 0.1 
        
        self.repl_policy = self._env.configs['cache_1']['rep_policy']
        self.l1 = self._env.l1

    def reset(self, victim_address=-1):
        
        """ returned obs = { agent_name : obs } """
        '''Episode termination:
            when the attacker make a guess NOTE: multiple guesses per episode implented in _env
            when there is length violation
            when there is guess before victim violation
            episode terminates '''
        
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        opponent_obs = self._env.reset(victim_address=victim_address, reset_cache_state=True)
        self.victim_address = self._env.victim_address
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        obs = {}
        obs['defender'] = np.array(list(reversed(self.defender_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_defender_obs(self, opponent_obs, opponent_info, action):
        
        ''' Defender's observation: self.observation_space = spaces.Discrete(
         [3, # cache_latency (for victim's if invoke_victm, otherwise attaker's)
          20, # domain_id
          2,  # victim_address (if invoke_victim = True) or attacker_address
          3,  # step_count
          4])  # defender's action'''
             
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
        #print('cur_opponent_obs: ', cur_opponent_obs)
        
        if not np.any(cur_opponent_obs==-1):
            
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                cur_opponent_obs[1] = self.random_domain #1
                cur_opponent_obs[2] = opponent_info['victim_address']
            else:
                cur_opponent_obs[1] = 1-self.random_domain#0
                cur_opponent_obs[2] = opponent_info['attacker_address']
            cur_opponent_obs[3] = self.step_count #0#self.step_count
            cur_opponent_obs[4] = action['defender']
            
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
            #if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == False:
            #    defender_reward = 20
            #    defender_success = True
                
            # Gets large reward if attacker attacker fails to make a guess in episode in time
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == False:
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
        info['guess_correct'] = defender_success
        return reward, info

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
        action_info = action.get('info')
        
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            
            if self.opponent_agent == 'benign' and benign_reset_victim:
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
        
        #action['defender'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15)
        actions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15)
        #assert(False)
        print_cache(self.l1)
        # = self._env.l1 --> self.hierarchy['cache_1'] in CacheGuessingGameEnv(env_config) 
        if self.repl_policy == replacement_policy.lru_lock_policy:
            assert(self.l1.rep_policy == lru_lock_policy)
            #print_cache(self.l1)
            
            set_no = 0
            lock_bit = '1111'
            if action['defender'] in actions:
                lock_bit = bin(action['defender'])[2:].zfill(4)
                #print(action['defender'])
                #print(lock_bit)
                #assert(False)
                self.l1.lock(self, set_no, lock_bit)
            #self.l1.lock(set_no, lock_bit)
            #print(lock_bit)
            #print(action['defender'])
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
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info 
        updated_reward, updated_info = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['defender'] = updated_reward['defender']
        obs['defender'] = self.get_defender_obs(opponent_obs, opponent_info, action) 
        done['defender'] = defender_done
        info['defender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['defender'])}
        info['defender'].update(opponent_info)
        #info['defender'].update(lock_bit)
        
        
        # Change the criteria to determine wether the game is done
        if defender_done:
            done['__all__'] = True

        info['__all__'] = {'action_mask':self.action_mask}
    
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["defender"])
        return obs, reward, done, info
 
@hydra.main(config_path="./rlmeta/config", config_name="ppo_lock")
def main(cfg):
    # TODO: pass the env_config information 
    # TODO: include lock_bits passed per each defender action
    env = CacheAttackerDefenderEnv(cfg.env_config)
    env.opponent_weights = [0.5, 0.5] #[0,1]
    action_space = env.action_space 
    obs = env.reset()
    done = {'__all__':False}
    #policy = replacement_policy.lru_lock_policy
    _env = CacheGuessingGameEnv()
    set_no = 0
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
        print("reward:", reward)
        print('attackers info:', info['attacker'])
        #print('benigns info:', info['benign'])
        print('defenders info:', info['defender'])
        #if info['attacker'].get('invoke_victim'):
        #    print('info[attacker]: ', info['attacker'])
      obs = env.reset()
      done = {'__all__':False}

if __name__ == "__main__":
    #mp.set_start_method("spawn")
    main()
