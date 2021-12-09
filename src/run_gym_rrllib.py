# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
#from CacheSimulator.src.gym_cache.envs.cache_simulator_wrapper import CacheSimulatorWrapper
import gym
import ray
import ray.tune as tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
import sys
sys.path.append("../src")
from models.dqn_model import DNNEncoder 

# the actual model used by the RLlib
class TestModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        hidden_dim = 256 
        super(TestModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.a_model = nn.Sequential(
            DNNEncoder(
                input_dim=int(np.product(self.obs_space.shape)),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, num_outputs)
        )
        self.v_model = nn.Sequential(
            DNNEncoder(
                input_dim=int(np.product(self.obs_space.shape)),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            ),
            nn.Linear(hidden_dim, 1)
        )
        self._last_flat_in = None
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._output = self.a_model(self._last_flat_in)
        return self._output, state 
    def value_function(self):
        return self.v_model(self._last_flat_in).squeeze(1)

ModelCatalog.register_custom_model("test_model", TestModel)
# RLlib does not work with gym registry, must redefine the environment in RLlib
# from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
# from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
# from cache_guessing_game_env_fix_impl import * # for prime and probe attack
from cache_guessing_game_env_impl import *
# (Re)Start the ray runtime.
if ray.is_initialized():
  ray.shutdown()

'''
CacheSimulatorDiversityWrapper
description:
a gym compatible environment that records legit patterns in the buffer (under some condition)
when step() a guess, first look up whether the pattern is in the register 
if it is then the reward will be very negative
otherwise it will be positive
at the end output everything in the register 
'''
class CacheSimulatorDiversityWrapper(gym.Env):
    def __init__(self, env_config):
        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.pattern_buffer = []
        self.repeat_pattern_reward = -10000
        self.correct_thre = 0.95
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_buffer = []
    def reset(self):
        self.action_buffer = []
        return self._env.reset()
    '''
    chcek whether the sequence in the action_buffer is seen before
    returns true if it is
    returns false and update the pattern buffer 
    '''
    # shall we use trie structure instead ???
    def check_and_save(self):
        if self.action_buffer in self.pattern_buffer:
            return True
        else:
            if self._env.calc_correct_rate() > self.correct_thre:
                if self.validation_env.calc_correct_seq(self.action_buffer) > self.correct_thre:
                    self.pattern_buffer.append(self.action_buffer)
            return False        
    def step(self,action):
        state, reward, done, info = self._env.step(action)
        latency = state[0]
        self.action_buffer.append((action,latency)) #latnecy is part of the attack trace
        #make sure the current existing correct guessing rate is high enough beofre 
        # altering the reward
        if done == False:
            return state, reward, done, info
        else:
            action = self._env.parse_action(action)
            is_guess = action[1]
            if is_guess == True:
                is_exist = self.check_and_save()
                if is_exist == True:
                    reward = self.repeat_pattern_reward #-10000
                return state, reward, done, info
            else:
                return state, reward, done, info
    def reset(self):
        print(self.pattern_buffer)
        self.action_buffer = []
        self.validation_env.reset()
        return self._env.reset()

ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
tune.register_env("cache_guessing_game_env_fix", CacheSimulatorDiversityWrapper)

# Two ways of training
# method 2b
config = {
    'env': 'cache_guessing_game_env_fix',
    'env_config': {
        'verbose': 1,
        "force_victim_hit": False,
        'flush_inst': False,
        "allow_victim_multi_access": False,
        "attacker_addr_s": 0,
        "attacker_addr_e": 3,
        "victim_addr_s": 0,
        "victim_addr_e": 1,
        "reset_limit": 1,
        "cache_configs": {
                # YAML config file for cache simulaton
            "architecture": {
              "word_size": 1, #bytes
              "block_size": 1, #bytes
              "write_back": True
            },
            "cache_1": {#required
              "blocks": 2, 
              "associativity": 2,  
              "hit_time": 1 #cycles
            },
            "mem": {#required
              "hit_time": 1000 #cycles
            }
        }
    }, 
    #'gamma': 0.9, 
    'num_gpus': 1, 
    'num_workers': 1, 
    'num_envs_per_worker': 1, 
    #'entropy_coeff': 0.001, 
    #'num_sgd_iter': 5, 
    #'vf_loss_coeff': 1e-05, 
    'model': {
        'custom_model': 'test_model',#'rnn', 
        #'max_seq_len': 20, 
        #'custom_model_config': {
        #    'cell_size': 32
        #   }
    }, 
    'framework': 'torch',
}
trainer = PPOTrainer(env=CacheSimulatorDiversityWrapper, config=config)
analysis= tune.run(
    PPOTrainer,
    local_dir="~/ray_results", 
    name="test_experiment",
    config=config)
