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
import copy

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
        #self.recent_model = []

    def forward(self, input_dict, state, seq_lens):
        #if obs[-1] > 0.99:
        #    self.recent_model.append((copy.deepcopy(self.a_model), copy.deepcopy(self.v_model)))
        #    if len(self.recent_model) > 5:
        #        self.recent_model.pop()
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._output = self.a_model(self._last_flat_in)
        return self._output, state 
    def value_function(self):
        return self.v_model(self._last_flat_in).squeeze(1)

    #def custom_loss(self, policy_loss, loss_input):

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
    def __init__(self, env_config, keep_latency=False):
        self.env_config = env_config
        # two choices for memorize the table
        # 1. keep both the action and the actual latency
        #     in this case the pattern has to be stored
        # 2. just keep the action but not the latency
        #      in this case, the model has to be stored
        self.keep_latency = keep_latency
        self._env = CacheGuessingGameEnv(env_config)
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.pattern_buffer = []
        self.enable_diversity = False
        self.repeat_pattern_reward = -10000
        self.correct_thre = 0.98
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_buffer = []
        #self.latency_buffer = []
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        self.pattern_init_state_buffer = []
    '''
    reset function
    '''
    def reset(self):
        self.action_buffer = []
        self.validation_env.reset()
        rtn = self._env.reset()
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        print("number of found patterns:" + str(len(self.pattern_buffer)))
        return rtn
    '''
    calculate the guessing correct rate of action_buffer
    '''
    def calc_correct_seq(self, action_buffer):
        correct_guess = 0
        total_guess = 0
        for _ in range(1):
            if self.replay_action_buffer(action_buffer):
                correct_guess += 1 
            total_guess += 1
        rtn = 1.0 * correct_guess / total_guess 
        print('calc_correct_seq ' + rtn)
        return rtn
    '''
    given the action and latency sequence, replay the cache game
    1. for known initial state, the loop runs once
    2. for randomized initial state, the loop runs multiple times
    ''' 
    def replay_action_buffer(self, action_buffer, init_state = None):
        replay_env = self.validation_env 
        while True:
            repeat = False                  # flag for not finding correct pattern
            replay_env.reset()
            if init_state != None:
               l1, victim_address = init_state
               replay_env.verbose = 1
               replay_env.l1 = l1
               replay_env.victim_address = victim_address
            for step in action_buffer:
               action = step[0]
               latency = step[1] 
               print(replay_env.parse_action((action)))
               obs, reward, _, _ = replay_env.step(action)
               print(obs)
               print(latency) 
               if obs[0] != latency:
                   repeat = True
            if repeat == True:
                continue
            else:
                break 
        if reward > 0:
            print("replay buffer correct guess")
            return True         # meaning correct guess
        else:
            return False        # meaning wrong guess
    '''
    chcek whether the sequence in the action_buffer is seen before
    returns true if it is
    returns false and update the pattern buffer 
    '''
    # shall we use trie structure instead ???
    def check_and_save(self):
        
        #return False
        '''
        if self.keep_latency == True:
            if self.action_buffer in self.pattern_buffer:
                return True
            else:
                if self._env.calc_correct_rate() > self.correct_thre:
                    print(self.action_buffer)
                    c = self.calc_correct_seq(self.action_buffer)
                    if c > self.correct_thre:
                        #print("calc_correct_seq")
                        self.pattern_buffer.append(self.action_buffer)
                        self.pattern_init_state_buffer.append(copy.deepcopy(self.pattern_init_state))
                        self.replay_pattern_buffer()
                return False        
        else:
        '''
        if self.action_buffer in self.pattern_buffer:
            return True
        else:
            if self._env.calc_correct_rate() > self.correct_thre:
                if self.keep_latency == True:  # need to calculate the latency of specific pattern
                    print(self.action_buffer)
                    c = self.calc_correct_seq(self.action_buffer)
                    if c > self.correct_thre:
                        #print("calc_correct_seq")
                        self.pattern_buffer.append(self.action_buffer)
                        self.pattern_init_state_buffer.append(copy.deepcopy(self.pattern_init_state))
                        self.replay_pattern_buffer()
                else:                           # just keep the pattern and the model
                    #c = self.calc_correct_seq(self.action_buffer) 
                    #if c > self.correct_thre: 
                    self.pattern_buffer.append(self.action_buffer)
                    print(self.pattern_buffer)
                    self._env.clear_guess_buffer_history()        
            return False        
    '''
    step function
    '''        
    def step(self,action):
        state, reward, done, info = self._env.step(action)
        #state = [state self._env.calc_correct_rate()]
        if self.keep_latency == True:
            latency = state[0]
        else:
            latency = -1
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
                if is_exist == True and self.enable_diversity==True:
                    reward = self.repeat_pattern_reward #-10000
                return state, reward, done, info
            else:
                return state, reward, done, info
    ## pretty print the pattern buffer
    '''
    replay function
    '''
    def replay_pattern_buffer(self):
        print("replay pattern buffer")
        i = 0
        print(self.pattern_buffer)
        print(self.pattern_init_state_buffer)
        for i in range(0, len(self.pattern_buffer)):
            actions = self.pattern_buffer[i]
            self.replay_action_buffer(actions, init_state = self.pattern_init_state_buffer[i])
            print("attack pattern " + str(i))
            i += 1
        if i == 5:
            exit() 

ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
tune.register_env("cache_simulator_diversity_wrapper", CacheSimulatorDiversityWrapper)

# Two ways of training
# method 2b
config = {
    'env': 'cache_simulator_diversity_wrapper',
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

'''
overwrite the default PPO trainer to trigger customized model savings
'''
#class PPOTrainerDiversity(PPOTrainer):
#    #def setup
#    def step(self):
#        result = super().step()
#        #if super().env._env.calc_correct_rate() > 0.95:
#        print("appending model in PPOTrainerDiversity")
#        self.workers.foreach_env_with_context(print)
#        assert(False)
#        #assert(self.workers.local_worker().env != None) 
#        #print(self.workers.local_worker()) 
#        #print(self.workers) 
#        #assert(False)
#        #self.env.model_buffer.append(self.get_policy()) 
#            #result.update(should_checkpoint=True)
#        return result
#
#trainer = PPOTrainerDiversity(env="cache_simulator_diversity_wrapper", config=config)
analysis= tune.run(
    PPOTrainer,
    local_dir="~/ray_results", 
    name="test_experiment",
    config=config)
