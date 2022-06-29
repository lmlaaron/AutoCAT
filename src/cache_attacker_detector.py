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
        env_config["cache_state_reset"] = False

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

        self.step_count = 0
    
    def get_detector_obs(self):
        # should return something like
        # {
        # 'agent_id':'player_0',
        # 'obs': []
        # 'action_mask': []
        # }
        return None

    def reset(self, victim_address=-1):
        self.step_count = 0
        obs = {}
        # reset opponent? 
        # can be either attacker or benign agent
        # Attacker reset
        attacker_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True
                                      )
        obs['attacker'] = attacker_obs

        # Victim reset
        # TODO:check the randomized initialization behavior
        self.victim_address = self._env.victim_address
        print("victim reset:", self.victim_address)
        
        # TODO:Detector reset - check this function
        detector_obs = self.get_detector_obs()
        obs['detector'] = detector_obs
        # returned obs should be a dictionary obs = {'agent_id':'','obs':obs,'mask':,mask}
        return obs
    
    def compute_reward(self, action):
        # TODO finish up criteria
        # when detector decides to alarm, query wether the opponent is an attacker or benign agent
        # if the attacker is correctly detected, then modify the attacker's reward
        return {}

    def step(self, action):
        # TODO should action be a dict or list 

        self.step_count += 1
        obs = {}
        reward = {}
        done = {'__all__':False}
        info = {}

        # Attacker update
        attacker_obs, attacker_reward, attacker_done, attacker_info = self._env.step(action['attacker'])
        obs['attacker'] = attacker_obs
        reward['attacker'] = attacker_reward
        done['attacker'] = attacker_done
        info['attacker'] = attacker_info
        
        # TODO:Detector update, check the update
        # obs, reward, done, info 
        obs['detector'] = None
        reward['detector'] = 1
        done['detector'] = None
        info['detector'] = None
        
        # Change the criteria to determine wether the game is done
        if attacker_done:
            done['__all__'] = True
        
        # returned should be dictionaries
        # obs = {'agent_id', 'obs', 'mask'}
        # reward = {'agent_id':float}
        # done = {'agent_id':bool, '__all__':bool}
        
        return obs, reward, done, info


if __name__ == '__main__':
    env = CacheAttackerDetectorEnv({})
    obs = env.reset()
    done = {'__all__':False}
    i = 0
    while not done['__all__']:
        i += 1
        obs, reward, done, info = env.step({'attacker':np.random.randint(9)})
        print("step: ", i,"obs: ", obs, "done:", done, "info", info )

