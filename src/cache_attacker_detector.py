import copy

from typing import Any, Dict, Sequence, Tuple

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
       
        self.opponent_agent = random.choice(['benign','attacker']) # 'benign', 'attacker' 
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0

    def reset(self, victim_address=-1):
        """
        returned obs = {'agent_id':obs}
        """
        self.opponent_agent = random.choice(['benign','attacker']) 
        self.action_mask = {'detector':True, 'attacker':self.opponent_agent=='attacker', 'benign':self.opponent_agent=='benign'}
        self.step_count = 0
        self.victim_address = self._env.victim_address

        obs = {}
        opponent_obs = self._env.reset(victim_address=victim_address,
                                       reset_cache_state=True
                                      )
        detector_obs = opponent_obs # so far the detector share the same observation space as 
        obs['detector'] = detector_obs
        obs['attacker'] = opponent_obs
        obs['benign'] = opponent_obs
        return obs
    
    def compute_reward(self, action, reward, opponent_done, opponent_attack_success=False):
        # TODO finish up criteria
        # when detector decides to alarm, query wether the opponent is an attacker or benign agent
        # if the attacker is correctly detected, then modify the attacker's reward
        # if attacker attacks correctly as the detector alarms, detector wins.
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
            else:
                detector_reward = -1
            if self.opponent_agent == 'attacker' and opponent_done and opponent_attack_success:
                # attacker has attacked *successfully*
                detector_reward = -20
        
        attacker_reward = reward['attacker']
        
        # determine detector's reward
        # if detector_flag and detector_correct:
        #    attacker_reward -= 0.1
        
        rew = {}
        rew['detector'] = detector_reward
        rew['attacker'] = attacker_reward
        return rew
    
    def get_detector_obs(self):
        pass

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
        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = opponent_done
        info['attacker'] = opponent_info
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = opponent_done
        info['benign'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info 
        updated_reward = self.compute_reward(action, reward, opponent_done, opponent_attack_success)
        reward['attacker'] = updated_reward['attacker']
        reward['detector'] = updated_reward['detector']
        obs['detector'] = opponent_obs # TODO(John):so far the detector shares the same observation as attacker
        done['detector'] = opponent_done
        info['detector'] = {"guess_correct":reward['detector']>0.5, "is_guess":bool(action['detector'])}
        
        # Change the criteria to determine wether the game is done
        if opponent_done:
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

