import copy, random, gym, hydra
from typing import Any, Dict
from collections import deque 
import numpy as np
from gym import spaces
from cache_guessing_game_env_defense import CacheGuessingGameEnv # TODO: make sure check the import file name change!!
from cache_simulator import *

class CacheAttackerDefenderEnv(gym.Env):

    def __init__(self, env_config: Dict[str, Any], keep_latency: bool = True) -> None:
        self.reset_observation = env_config.get("reset_observation", False)
        self.keep_latency = keep_latency # NOTE: purpose of this parameter? 
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)
        self._env = CacheGuessingGameEnv(env_config) 
        self.validation_env = CacheGuessingGameEnv(env_config)
        #self.observation_space = self._env.observation_space 
        self.window_size = 64 # self.cache_size * 8 + 8 
        self.feature_size = 5
        self.action_space = spaces.Discrete(256) # 16 for 4-way, 256 for 8-way
        
        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.victim_address = self._env.victim_address
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.associativity = env_config.get('associativity', 4)
    
        self.attacker_address_space = range(self.attacker_address_min, self.attacker_address_max +1)
        self.victim_address_space = range(self.victim_address_min, self.victim_address_max + 1) 
        self.max_box_value = max(self.window_size + 2,  2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) + 1)
        self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, self.feature_size))
        
        self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.max_step = 20
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1]) 
        self.defender_reward_scale = 1 #0.1 
        self.repl_policy = env_config.get('rep_policy', 'lru_lock_policy')
        
    
    def reset(self, victim_address=-1):
       
        ''' returned obs = { agent_name : obs } '''
        ''' Episode termination: when there is length violation '''
        self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        opponent_obs = self._env.reset(victim_address=-1, reset_cache_state=False)
        self.victim_address = victim_address 
        self.random_domain = random.choice([0,1])
        self.defender_obs = deque([[-1, -1, -1, -1, -1]] * self.max_step)
        self.step_count = 0 
        obs = {}
        obs['defender'] = np.array(list(reversed(self.defender_obs)))
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def get_defender_obs(self, opponent_obs, opponent_info, action):
        
        ''' Defender's observation: [cache_latency, domain_id, address, step_count, defender's actions]'''
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])
         
        if not np.any(cur_opponent_obs==-1):
            
            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']
                if cur_opponent_obs[2] == 4:
                    print('victims latency: ', cur_opponent_obs[0])
                    #print('attackers latency: ', cur_opponent_obs[0])
                else:
                    #print('victims latency: ', cur_opponent_obs[0])
                    print('attackers latency: ', cur_opponent_obs[0])
                cur_opponent_obs[1] = self.random_domain #1
                cur_opponent_obs[2] = opponent_info['victim_address']
                print('victim_address: ', cur_opponent_obs[2], '\n')
                
            else:
                print('attackers latency: ', cur_opponent_obs[0])
                cur_opponent_obs[1] = 1-self.random_domain#0
                cur_opponent_obs[2] = opponent_info['attacker_address']
                
            cur_opponent_obs[3] = self.step_count 
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
            # TODO: attacker make a wrong guess
     
            # attacker attacker fails to make a guess in episode in time
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == False:
                defender_reward = 20
                defender_success = True
                
            ''' penalty conditions '''
            # attacker succeed to guess a victim's secret
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success == True:   
                defender_reward = -20    
        
            # TODO: attacker succeed to guess a victim no access
            
            # opponent_agent is benign
            if self.opponent_agent == 'benign':
                defender_reward = -2
        
            # penalized whenever defender's action is not locking
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
            2. or lock a set using lock_bit == 0001, means lock way_no = 3 '''
        #print_cache(self._env.l1)
        
        obs = {}
        reward = {}
        done = {'__all__':False} 
        info = {}
        action_info = action.get('info')
        
        if isinstance(action, np.ndarray):
            action = action.item()
        #print_cache(self._env.l1)
        
        ''' defender's action '''
        set_no = 0
        #lock_bit = bin(action['defender'])[2:].zfill(4)
        lock_bit = bin(action['defender'])[2:].zfill(int(self.associativity))
        assert len(lock_bit) == int(self.associativity)
        self._env.l1.lock(set_no, lock_bit) # TODO: create a new function in _env then use it. do not call functions inside a wrapper
        #print_cache(self._env.l1) #TODO
        
        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)
            
            if self.opponent_agent == 'benign' and benign_reset_victim:
                self._env.set_victim(benign_victim_addr) 
                self.victim_address = self._env.victim_address
                
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        #print_cache(self._env.l1)
        #print_cache(self._env.l1)
        if opponent_done:
            
            opponent_obs = self._env.reset(reset_cache_state=False)
            self.victim_address = self._env.victim_address
            self.step_count -= 1 # The reset/guess step should not be counted 
            defender_done = False
            
            
        if self.step_count >= self.max_step:
            defender_done = True # will not terminate the episode
        else:
            defender_done = False
        
        # attacker
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = opponent_done 
        info['attacker'] = opponent_info
        
        #benign
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = opponent_done 
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
        
        self.step_count += 1
        #criteria to determine wether the game is done
        if self.step_count >= self.max_step:
            opponent_done = True 
            done['__all__'] = True

        info['__all__'] = {'action_mask':self.action_mask}
    
        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["defender"])
        print_cache(self._env.l1) # TODO: create a new function in _env then use it. do not call functions inside a wrapper
        return obs, reward, done, info
 
@hydra.main(config_path="./rlmeta/config", config_name="ppo_lock")
# check env_config in ppo_lock for config_empty or config_noempty 
def main(cfg):
    env = CacheAttackerDefenderEnv(cfg.env_config)
    _env = CacheGuessingGameEnv(cfg.env_config)
    env.opponent_weights = [0, 1] #[0.5, 0.5] #[0,1]
    obs = _env.reset(victim_address=-1) #=5)
    done = {'__all__':False}
    
    ''' for unit test '''
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_000.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_050.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_050-1.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_075.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_075-1.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_100.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_eff-1.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_eff-2.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_eff-3.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_eff-4.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_eff-5.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s8w_000.txt')
    test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s8w_025.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s8w_0125.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s16w_0125.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s8w_050.txt')
    #test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_reset.txt')
    trace = test_action.read().splitlines()
    actions_list = [list(map(int, x.split())) for x in trace]
    
    
    actions = [{'attacker': values[0], 'benign': values[1], 'defender': values[2]} for values in actions_list]
    i = 0
    for k in range(1):
        while not done['__all__']:
            print("STEP: ", i)
            action = actions[i]
            print('defenders action: ', action['defender'])
            print('attackers action: ', action['attacker'])
            obs, reward, done, info = env.step(action)
            
            
            print("observation of defender: ", '\n', obs['defender'])
            #print("action: ", action)
            print("reward:", reward)
            print('-----------------------------------------------')
            #print('attackers info:', info['attacker'])
            #print('defenders info:', info['defender'])
            #print(done)
            i += 1
        obs = env.reset()
        done = {'__all__':False}

if __name__ == "__main__":
    main()
