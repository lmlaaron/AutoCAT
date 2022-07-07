import copy

from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
       
        self.opponent_agent = 'attacker' # 'benign', 'attacker' 

        self.step_count = 0
    
    def get_detector_obs(self, obs):
        # should return something like
        # {
        # 'agent_id':'player_0',
        # 'obs': []
        # 'action_mask': []
        # }
        # TODO: implement this method 
        return obs

    def reset(self, victim_address=-1):
        self.step_count = 0
        obs = {}
        # reset opponent? 
        # can be either attacker or benign agent
        # Attacker reset
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True
                                      )
        obs['opponent'] = opponent_obs

        # Victim reset
        # TODO:check the randomized initialization behavior
        self.victim_address = self._env.victim_address
        #print("victim reset:", self.victim_address)
        
        # TODO:Detector reset - check this function
        detector_obs = opponent_obs # so far the detector share the same observation space as 
        obs['detector'] = detector_obs
        # returned obs should be a dictionary obs = {'agent_id':'','obs':obs,'mask':,mask}
        return obs
    
    def compute_reward(self, action, reward, opponent_done, opponent_attack_success=False):
        # TODO finish up criteria
        # when detector decides to alarm, query wether the opponent is an attacker or benign agent
        # if the attacker is correctly detected, then modify the attacker's reward
        # if attacker attacks correctly as the detector alarms, detector wins.
        action_detector = action['detector']
        action_opponent = action['opponent']

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
            detector_reward = -0.01
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
                # attacker has attacked *successfully*
                detector_reward = -1
        
        #TODO
        opponent_reward = reward['opponent']
        # determine attacker's reward
        # modify the attacker's reward according to the detector results
        #if detector_flag and detector_correct:
        #    opponent_reward -= 1
        
        rew = {}
        rew['detector'] = detector_reward
        rew['opponent'] = opponent_reward
        return rew

    def step(self, action):
        # TODO should action be a dict or list 
        # the action should selected outside the environment, which is produced by the detector's objective agent
        # this can be produced by benign agent or malicious agent(two policy classes)
        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}

        # Attacker update
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action['opponent'])
        obs['opponent'] = opponent_obs
        reward['opponent'] = opponent_reward
        done['opponent'] = opponent_done
        info['opponent'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # TODO:Detector update, check the update
        # obs, reward, done, info 
        updated_reward = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['opponent'] = updated_reward['opponent']
        reward['detector'] = updated_reward['detector']
        obs['detector'] = opponent_obs # so far the detector shares the same observation as attacker
        done['detector'] = None
        info['detector'] = None
        
        # Change the criteria to determine wether the game is done
        if opponent_done:
            done['__all__'] = True
        
        # returned should be dictionaries
        # obs = {'agent_id', 'obs', 'mask'}
        # reward = {'agent_id':float}
        # done = {'agent_id':bool, '__all__':bool}
        
        return obs, reward, done, info

if __name__ == '__main__':
    env = CacheAttackerDetectorEnv({})
    action_space = env.action_space
    obs = env.reset()
    done = {'__all__':False}
    i = 0
    from IPython import embed; embed()
    while not done['__all__']:
        i += 1
        obs, reward, done, info = env.step({'opponent':np.random.randint(low=0, high=128),
                                            'detector':0})
        print("step: ", i)
        print("obs: ", obs)
        print("done:", done)
        print("reward:", reward)
        print("info:", info )

