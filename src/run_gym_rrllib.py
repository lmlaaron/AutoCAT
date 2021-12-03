# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
import ray.tune as tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelCatalog
#from models.dqn_model import DQNModel

import sys
sys.path.append("../src")
from models.dqn_model import DNNEncoder, DQNModel 

# the actual model used by the RLlib
class TestModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        hidden_dim = 256 
        super(TestModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        ##self.a_model = DQNModel(
        ##    input_dim=int(np.product(self.obs_space.shape)),
        ##    hidden_dim=hidden_dim,
        ##    action_dim=num_outputs,
        ##)
        ##self.v_model = DQNModel(
        ##    input_dim=int(np.product(self.obs_space.shape)),
        ##    hidden_dim=hidden_dim,
        ##    action_dim=1,
        ##)
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

#RLlib does not work with gym registry, must redefine the environment in RLlib
# from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
#from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
# from cache_guessing_game_env_fix_impl import * # for prime and probe attack
from cache_guessing_game_env_impl import *
# (Re)Start the ray runtime.
if ray.is_initialized():
  ray.shutdown()

ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv) #Fix)
# Two ways of training
# 1. directly use trainer
# 2. use tune API

#method 1
##trainer = PPOTrainer(env=CacheGuessingGameEnv, config={
##    "env_config":{},
##    "model": {
##        "use_lstm": True
##    }
##})
####trainer.train()
#method 
##analysis = tune.run(
##    #DQNTrainer,
##    PPOTrainer, 
##    local_dir="~/ray_results", 
##    name="test_experiment",
##    #checkpoint_at_end=True,
##    #stop={
##    #    "episodes_total": 500,
##    #},
##    config={
##        "num_gpus": 1,
##        #"seed": 0xCC,
##        "env": "cache_guessing_game_env_fix",
##        #"rollout_fragment_length": 5,
##        #"train_batch_size": 5,
##        #"sgd_minibatch_size": 5,
##        "model": { #see https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings
##            #"fcnet_hiddens": [8192, 512], #, 4096, 512],
##            # Activation function descriptor.
##            # Supported values are: "tanh", "relu", "swish" (or "silu"),
##            # "linear" (or None).
##            #"fcnet_activation": "relu",
##            #"use_lstm": True,
##            # specify our custom model 
##            # Extra kwargs to be passed to your model's c'tor
##            #"custom_model_config": {
##                #"input_files": '' ,
##            #}
##            ####"custom_model_config": {
##            ####    "input_dim": 64,
##            ####    "hidden_dim": 512,
##            ####    "action_dim": 3 * 5 * 18 * 2 * 16,
##            ####},
##        },
##    }
##)

# method 2b
config = {
    'env': 'cache_guessing_game_env_fix',
    #'env_config': {'repeat_delay': 2}, 
    #'gamma': 0.9, 
    'num_gpus': 1, 
    'num_workers': 0, 
    'num_envs_per_worker': 20, 
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
    'framework': 'torch'
}
tune.run(
    "PPO",
    local_dir="~/ray_results", 
    name="test_experiment",
    config=config)
