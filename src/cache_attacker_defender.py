import copy
import random
import gym
import hydra
from typing import Any, Dict
from collections import deque
import numpy as np
from gym import spaces
from cache_guessing_game_env_defense import AttackerCacheGuessingGameEnv
from cache_simulator import *


class CacheAttackerDefenderEnv(gym.Env):

    def __init__(self, env_config: Dict[str, Any]) -> None:
        self.reset_observation = env_config.get("reset_observation", False)
        self.env_config = env_config
        self.episode_length = env_config.get("episode_length", 80)
        self.threshold = env_config.get("threshold", 0.8)
        self._env = AttackerCacheGuessingGameEnv(env_config)
        self.validation_env = AttackerCacheGuessingGameEnv(env_config)
        self.window_size = env_config.get('window_size', 64)
        self.feature_size = env_config.get('feature_size', 6)
        cache_config = env_config.get('cache_configs', {})
        cache_1_config = cache_config.get('cache_1', {})
        self.blocks = cache_1_config.get('blocks', 8)
        self.associativity = cache_1_config.get('associativity', 4)
        self.n_sets = int(self.blocks / self.associativity)
        self.action_space_size = env_config.get('action_space_size', 16)  # action_space_size = 2 ^ associativity
        self.action_space = spaces.Discrete(self.action_space_size)
        self.victim_address_min = env_config.get('victim_addr_s', 9)
        self.victim_address_max = env_config.get('victim_addr_e', 9)
        self.victim_address = self._env.victim_address
        self.attacker_address_max = env_config.get('attacker_addr_s', 0)
        self.attacker_address_min = env_config.get('attacker_addr_e', 7)
        self.attacker_address_space = range(self.attacker_address_min, self.attacker_address_max + 1)
        self.victim_address_space = range(self.victim_address_min, self.victim_address_max + 1)
        self.max_box_value = max(self.window_size + 2, 2 * len(self.attacker_address_space)
                                 + 1 + len(self.victim_address_space) + 1)
        self.observation_space = spaces.Box(low=-1, high=self.max_box_value,
                                            shape=(self.window_size, self.feature_size))
        self.opponent_weights = env_config.get("opponent_weights", [0.5, 0.5])
        self.opponent_agent = random.choices(['benign', 'attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender': True, 'attacker': self.opponent_agent == 'attacker',
                            'benign': self.opponent_agent == 'benign'}
        self.step_count = 0
        self.max_step = env_config.get('max_step', 20)
        self.defender_obs = deque([[-1] * (4 + self.n_sets)] * self.max_step)
        self.random_domain = random.choice([0, 1])
        self.defender_reward_scale = env_config.get('defender_reward_scale', 1)
        self.opponent_weights = env_config.get('opponent_weights', [1, 0])
        self.def_success_reward = env_config.get('def_success_reward', 20)
        self.def_fail_reward = env_config.get('def_fail_reward', -20)
        self.def_benign_reward = env_config.get('def_benign_reward', 0)
        self.def_action_reward = env_config.get('def_action_reward', 0)
        self.def_latency_reward = env_config.get('def_latency_reward', -0.01)

        assert 4 + int(self.n_sets) == self.feature_size, f'feature size mismatches w/ width of obs_space'

    def reset(self, victim_address=-1) -> dict:

        """ returned obs = { agent_name : obs } """
        ''' Episode termination: when there is length violation '''
        self.opponent_agent = random.choices(['benign', 'attacker'], weights=self.opponent_weights, k=1)[0]
        self.action_mask = {'defender': True, 'attacker': self.opponent_agent == 'attacker',
                            'benign': self.opponent_agent == 'benign'}
        opponent_obs = self._env.reset(victim_address=-1, reset_cache_state=False)
        self.victim_address = victim_address
        self.random_domain = random.choice([0, 1])
        self.defender_obs = deque([[-1] * (4 + self.n_sets)] * self.max_step)
        self.step_count = 0
        obs = {'defender': np.array(list(reversed(self.defender_obs))), 'attacker': opponent_obs,
               'benign': opponent_obs}
        return obs

    def get_defender_obs(self, opponent_obs, opponent_info, action):

        """ Defender's observation: [cache_latency, domain_id, address, step_count, defender's actions]"""
        cur_opponent_obs = copy.deepcopy(opponent_obs[0])

        if not np.any(cur_opponent_obs == -1):

            if opponent_info.get('invoke_victim'):
                cur_opponent_obs[0] = opponent_info['victim_latency']

                if cur_opponent_obs[2] == 4:
                    print('victims latency: ', cur_opponent_obs[0])

                else:
                    print('attackers latency: ', cur_opponent_obs[0])

                cur_opponent_obs[1] = self.random_domain  # 1
                cur_opponent_obs[2] = opponent_info['victim_address']
                print('victim_address: ', cur_opponent_obs[2], '\n')

            else:
                print('attackers latency: ', cur_opponent_obs[0])
                cur_opponent_obs[1] = 1 - self.random_domain  # 0
                cur_opponent_obs[2] = opponent_info['attacker_address']

            cur_opponent_obs[3] = self.step_count

            # accommodate different no of actions of defender according to set no
            for i in range(0, self.n_sets):
                cur_opponent_obs[i + 4] = action['defender'][i]
            self.defender_obs.append(cur_opponent_obs)
            self.defender_obs.popleft()

        latency = int(cur_opponent_obs[0])

        return np.array(list(reversed(self.defender_obs)), dtype=object), latency

    def compute_reward(self, action, latency, reward, opponent_done, opponent_attack_success=False):

        action_defender = action['defender']
        defender_success = False
        defender_reward = 0

        if action_defender is not None:

            # 1. Gets large reward if attacker gets a wrong guess in time
            if self.opponent_agent == 'attacker' and opponent_done:
                if not opponent_attack_success:
                    defender_reward = self.def_success_reward
                    defender_success = True

            # 2. Gets large penalty if attacker gets correct guess in time
            elif self.opponent_agent == 'attacker' and opponent_done:
                if opponent_attack_success:
                    defender_reward = self.def_fail_reward

            # 3. Gets 0 penalty if the identify opponent agent as benign
            elif self.opponent_agent == 'benign':
                defender_reward = self.def_benign_reward

            # 4. Gets 0 penalty for any defenders action
            elif isinstance(action_defender, int):
                defender_reward = self.def_action_reward

            # 5. Gets small penalty for no of cache misses 
            elif self.opponent_agent == 'attacker' or self.opponent_agent == 'benign':
                if latency == 1:
                    defender_reward = self.def_latency_reward

        reward['defender'] = defender_reward * self.defender_reward_scale
        info = {'guess_correct': defender_success}

        return reward, info

    def step(self, action):

        """ Defender's actions: No of actions matches with the lockbits for n-ways
            1. unlock a set using lock_bit == 0000, means all way_no are unlocked
            2. or lock a set using lock_bit == 0001, means lock way_no = 3 """
        # print_cache(self._env.l1)

        obs = {}
        reward = {}
        done = {'__all__': False}
        info = {}
        action_info = action.get('info')

        if isinstance(action, np.ndarray):
            action = action.item()

        ''' defender's action '''
        n_sets = self.n_sets

        for set_index in range(0, n_sets):
            print('set_index', set_index, 'defenders action: ', action['defender'][set_index])
            if self.associativity == 1:
                lock_bit = int(action['defender'][set_index])
            else:
                lock_bit = bin(int(action['defender'][set_index]))[2:].zfill(int(self.associativity))
                # print(lock_bit)
                assert len(lock_bit) == int(self.associativity), f"Lock bit length does not match associativity"

            self._env.lock_l1(set_index, lock_bit)

        if action_info:
            benign_reset_victim = action_info.get('reset_victim_addr', False)
            benign_victim_addr = action_info.get('victim_addr', None)

            if self.opponent_agent == 'benign' and benign_reset_victim:
                self._env.set_victim(benign_victim_addr)
                self.victim_address = self._env.victim_address

        opponent_obs, opponent_reward, opponent_done, opponent_info = self._env.step(action[self.opponent_agent])

        if opponent_done:
            opponent_obs = self._env.reset(reset_cache_state=False)
            self.victim_address = self._env.victim_address
            self.step_count -= 1  # The reset/guess step should not be counted
            defender_done = False

        elif self.step_count >= self.max_step:
            defender_done = True  # will not terminate the episode
        else:
            defender_done = False

        # attacker
        obs['attacker'] = opponent_obs
        reward['attacker'] = opponent_reward
        done['attacker'] = opponent_done
        info['attacker'] = opponent_info

        # benign
        obs['benign'] = opponent_obs
        reward['benign'] = opponent_reward
        done['benign'] = opponent_done
        info['benign'] = opponent_info
        opponent_attack_success = opponent_info.get('guess_correct', False)

        # obs, reward, done, info 
        obs['defender'], latency = self.get_defender_obs(opponent_obs, opponent_info, action)
        updated_reward, updated_info = self.compute_reward(action, latency, reward,
                                                           opponent_done, opponent_attack_success)

        reward['attacker'] = updated_reward['attacker']
        reward['defender'] = updated_reward['defender']

        done['defender'] = defender_done
        info['defender'] = {"guess_correct": updated_info["guess_correct"], "is_guess": bool(action['defender'])}
        info['defender'].update(opponent_info)

        self.step_count += 1
        # criteria to determine weather the game is done
        if self.step_count >= self.max_step:
            done['__all__'] = True

        info['__all__'] = {'action_mask': self.action_mask}

        for k, v in info.items():
            info[k].update({'action_mask': self.action_mask})

        print_cache(self._env.l1)  # TODO: create a new function in _env then use it. do not call funcs inside a wrapper
        return obs, reward, done, info


@hydra.main(config_path="./rlmeta/config", config_name="ppo_lru_lock")
def main(cfg):
    # checks the parameter setting for training and cache configuration
    env = CacheAttackerDefenderEnv(cfg.env_config)
    _env = AttackerCacheGuessingGameEnv(cfg.env_config)
    done = {'__all__': False}

    ''' for unit test '''
    test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s2w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s4w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_1s8w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_2s4w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_4s1w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_4s2w.txt')
    # test_action = open('/home/geunbae/CacheSimulator/env_test/rep_policy/rldefense/lru_lock_8s1w.txt')
    trace = test_action.read().splitlines()
    actions_list = [list(map(int, x.split())) for x in trace]
    actions = [{'attacker': values[0], 'benign': values[1], 'defender': values[2:]} for values in actions_list]
    i = 0
    for k in range(1):
        while not done['__all__'] and i < len(actions):
            print("STEP: ", i)
            action = copy.deepcopy(actions[i])
            print('attackers action: ', action['attacker'])
            obs, reward, done, info = env.step(action)
            for set_index in range(0, env.n_sets):
                action['defender'] = [actions[i]['defender'][set_index]]
                # print('defenders action at set_index {}:'.format(set_index), action['defender'])
            # print("observation of defender: ", '\n', obs['defender'])
            # print('attackers info:', info['attacker'])
            # print('defenders info:', info['defender'])
            # print("action: ", action)
            print("reward:", reward)
            i += 1

        done = {'__all__': False}


if __name__ == "__main__":
    main()
