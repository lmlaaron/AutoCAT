import copy

from typing import Any, Dict, Sequence, Tuple
from collections import deque
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import gym
from gym import spaces

from cache_guessing_game_env_impl import CacheGuessingGameEnv



#class CacheAttackerDetectorEnv(gym.Env):
class CacheCovertSenderReceiverEnv(gym.Env):
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
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        #print(self._env.victim_address_space)
        #assert(-1)


        self.victim_secret_min = self._env.victim_address_min 
        self.victim_secret_max = self._env.victim_address_max

        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address



        # add noise
        self.noise_prob = env_config.get("noise_prob", 0) 
        self.noise_address_min = env_config.get("noise_address_min", self._env.victim_address_min)
        self.noise_address_max = env_config.get("noise_address_max", self._env.victim_address_max) 


        self.sender_action_space = spaces.Discrete(self.victim_address_max - self.victim_address_min + 1)
        #self.receiver_action_space = self._env.action_space
        #self.action_space = self._env.action_space

        #self.opponent_weights = env_config.get("opponent_weights", [0.5,0.5]) 
        #self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0] 
        #self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.action_mask = { 'sender': True, 'receiver': True} #self.receiver_agent}
        self.step_count = 0
        self.max_step = 32
        self.detector_obs = np.array([1.0, 0.0]) #deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        self.detector_reward_scale = 0.1 #1.0
        

    def reset(self, victim_address=-1):
        """
        returned obs = { agent_name : obs }
        """
        #####self.opponent_agent = random.choices(['benign','attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'sender': True, 'receiver': True} #{'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        #print("ma victim_address " + str(victim_address) )
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=False)
        self.victim_address = self._env.victim_address
        self.detector_obs = np.array([1.0,0.0])#deque([[-1, -1, -1, -1]] * self.max_step)
        self.random_domain = random.choice([0,1])
        obs = {}
        obs['sender'] = self.detector_obs #opponent_obs  #np.array(list(reversed(self.detector_obs)))
        obs['receiver'] = opponent_obs
        return obs


    def get_sender_obs(self, receiver_obs, receiver_info):
        #self._env = opponent_info['victim_secret']
        return np.array([receiver_info['victim_secret'], self.step_count])
    ###def get_detector_obs(self, opponent_obs, opponent_info):
    ###    cur_opponent_obs = copy.deepcopy(opponent_obs[0])
    ###    if not np.any(cur_opponent_obs==-1):
    ###        # Make sure the observation does not leak information for detector
    ###        # TODO should we include step number? - yes we should - the guess step should not be observed by the detector
    ###        # attacker obs: r, victim_accessed, original action, current step
    ###        # detector obs: r, domain_id, memory address, 0
    ###        if opponent_info.get('invoke_victim'):
    ###            cur_opponent_obs[0] = opponent_info['victim_latency']
    ###            cur_opponent_obs[1] = self.random_domain #1
    ###            cur_opponent_obs[2] = opponent_info['victim_address']
    ###        else:
    ###            cur_opponent_obs[1] = 1-self.random_domain#0
    ###            cur_opponent_obs[2] = opponent_info['attacker_address']
    ###        cur_opponent_obs[3] = self.step_count #0#self.step_count
    ###        self.detector_obs.append(cur_opponent_obs)
    ###        self.detector_obs.popleft()
    ###    return np.array(list(reversed(self.detector_obs)))

    def compute_reward(self, action, reward, receiver_done, receiver_attack_success=False):
        action_sender = action['sender']
        action_receiver = action['receiver']

        receiver_reward = reward['receiver']
        sender_reward = 0 #reward['sender']

        rew = {}
        rew['sender'] = sender_reward #* self.detector_reward_scale
        rew['receiver'] = receiver_reward

        info = {}
        info['guess_correct'] = True #detector_correct
        return rew, info
    ##    action_detector = action['detector']
    ##    action_attacker = action['attacker']
    ##    
    ##    # determine detector's reward 
    ##    detector_flag = False
    ##    detector_correct = False
    ##    
    ##    if action_detector == 1:
    ##        detector_flag = True
    ##        # terminate the episode
    ##        # detector flag the opponent as an attacker
    ##        if self.opponent_agent == 'benign':
    ##            #detector_reward = - self.max_step + self.step_count - 1 #punish false positive
    ##            detector_reward = - 10 * self.max_step
    ##        elif self.opponent_agent == 'attacker':
    ##            detector_reward = max(self.max_step - self.step_count, 0)
    ##            detector_correct = True
    ##    else:
    ##        # else receive a timestep penalty
    ##        if self.opponent_agent == 'benign':
    ##            detector_reward = 0 # no reward if benign agent
    ##            detector_correct = True
    ##        else:
    ##            detector_reward = 0 # no reward if attacker has done nothing meaningful
    ##    

    ##    if action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
    ##        # attacker episode terminate, and has attacked *successfully*
    ##        detector_reward = -20
    ##    elif action_detector == 0 and self.opponent_agent == 'attacker' and opponent_done:
    ##        # attacker episode terminates, but has done nothing successfully
    ##        detector_reward = 0
    ##    

    ##    attacker_reward = reward['attacker']
    ##    
    ##    if action_detector == 1 and detector_correct:
    ##        # the attacker should not receive as much reward if being detected
    ##        attacker_reward -= 10 
    ##    #else:
    ##    #    attacker_reward += 0.1
    ##    
    ##    rew = {}
    ##    rew['detector'] = detector_reward * self.detector_reward_scale
    ##    rew['attacker'] = attacker_reward

    ##    info = {}
    ##    info['guess_correct'] = detector_correct
    ##    return rew, info

    def step(self, action):

        # sender does a step
        self._env.sender_step(action['sender'])
        # added something in the middle
        #return obs, reward, done, info

        self.step_count += 1
        #if self.step_count <=8: action['detector']=0
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}
        ##### Attacker update
        ####action_info = action.get('info')
        #####if self.opponent_agent == 'benign':
        ######if action_info:
        ######    benign_reset_victim = action_info.get('reset_victim_addr', False)
        ######    benign_victim_addr = action_info.get('victim_addr', None)
        ######    if self.opponent_agent == 'benign' and benign_reset_victim:
        ######        self._env.set_victim(benign_victim_addr) 
        ######        self.victim_address = self._env.victim_address

        ##### inject noise
        if random.random() < self.noise_prob:
            noise_action = random.randint(self.noise_address_min, self.noise_address_max)
            self._env.noise_step(noise_action)

        ##### receiver does a step
        receiver_obs, receiver_reward, receiver_done, receiver_info = self._env.step(action['receiver'])#self.opponent_agent])
        
        if receiver_done:
            receiver_obs = self._env.reset(reset_cache_state=False)
            self.victim_address = self._env.victim_address
            #self.step_count -= 1 # The reset/guess step should not be counted
        if self.step_count >= self.max_step:
            sender_done = True
            receiver_reward = self._env.length_violation_reward
        elif receiver_done:
            # this time is length violation but receiver may not have receiver done reward
            sender_done = True
        else:
            sender_done = False
        #### attacker
        ###obs['attacker'] = opponent_obs
        ###reward['attacker'] = opponent_reward
        ###done['attacker'] = detector_done #Figure out correctness
        ###info['attacker'] = opponent_info
        # receiver
        obs['receiver'] = receiver_obs
        reward['receiver'] = receiver_reward
        done['receiver'] = receiver_done
        info['receiver'] = receiver_info

        #######benign
        ######obs['benign'] = opponent_obs
        ######reward['benign'] = opponent_reward
        ######done['benign'] = detector_done #Figure out correctness
        ######info['benign'] = opponent_info
        receiver_attack_success = receiver_info.get('guess_correct', False)

        # obs, reward, done, info 
        updated_reward, updated_info = self.compute_reward(action, reward, receiver_done, receiver_attack_success)
        reward['receiver'] = updated_reward['receiver']
        reward['sender'] = updated_reward['receiver']#updated_reward['sender']
        obs['sender'] = self.get_sender_obs(receiver_obs, receiver_info) 
        done['sender'] = sender_done
        info['sender'] = {"guess_correct":updated_info["guess_correct"], "is_guess":bool(action['sender'])}
        info['sender'].update(receiver_info)
        # Change the criteria to determine wether the game is done
        if sender_done or receiver_done:
            done['__all__'] = True
        #from IPython import embed; embed()

        info['__all__'] = {'action_mask':self.action_mask}

        #print("obs: ", obs['sender'])

        for k,v in info.items():
            info[k].update({'action_mask':self.action_mask})
        #print(obs["detector"])
        return obs, reward, done, info



@hydra.main(config_path="./rlmeta/config", config_name="ppo_exp")
def main(cfg):
    env = CacheCovertSenderReceiverEnv(cfg.env_config)
    #env = CacheAttackerDetectorEnv({})
    ####env.opponent_weights = [0,1]
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__':False}
    i = 0
    for k in range(2):
      actions =[
        {'receiver': 0, 'sender': 0},
        {'receiver': 1, 'sender': 0},
        {'receiver': 2, 'sender': 0},
        {'receiver': 3, 'sender': 0},
        {'receiver': 4, 'sender': 0},
        {'receiver': 5, 'sender': 0},
        {'receiver': 6, 'sender': 0},
        {'receiver': 7, 'sender': 0},
        {'receiver': 8, 'sender': 1 + env._env.victim_secret - env._env.victim_address_min},
        {'receiver': 1, 'sender': 0},
        {'receiver': 2, 'sender': 0},
        {'receiver': 3, 'sender': 0},
        {'receiver': 4, 'sender': 0},
        {'receiver': 5, 'sender': 0},
        {'receiver': 6, 'sender': 0},
        {'receiver': 7, 'sender': 0},
        {'receiver': 10 + env._env.victim_secret - env._env.victim_address_min, 'sender': 0}
      ]
      while (not done['__all__'] ) and i < len(actions):
        #i += 1
        #action = {'receiver':np.random.randint(low=3, high=6),
        #          'sender':np.random.randint(low=0, high=1)}
        action = actions[i]
        obs, reward, done, info = env.step(action)
        print("step: ", i)
        print("obs: ", obs['sender'])
        print("action: ", action)
        print("victim address: ",  env._env.victim_address)

        #print("done:", done)
        print("reward:", reward)
        print(env.victim_address_min, env.victim_address_max)
        #print("info:", info )
        #if info['receiver'].get('invoke_victim'):
        #    print(info['receiver'])
        i += 1
      obs = env.reset()
      done = {'__all__':False}


if __name__ == '__main__':
    main()
