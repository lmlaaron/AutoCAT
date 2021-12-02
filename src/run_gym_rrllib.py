# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer
import ray.tune as tune
from typing import Optional
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import numpy as np
from ray.rllib.models import ModelCatalog
#from models.dqn_model import DQNModel
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super(ResidualBlock, self).__init__()
        self.dim = dim
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, self.dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.dim, self.dim))
        self.layers = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class DNNEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_blocks: Optional[int] = 1) -> None:
        super(DNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_blocks):
            layers.append(ResidualBlock(self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQNModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_blocks: Optional[int] = 1) -> None:
        super(DQNModel, self).__init__()
        TorchModelV2.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_blocks = num_blocks
        self.backbone = DNNEncoder(self.input_dim, self.hidden_dim,
                                   self.hidden_dim, self.num_blocks)
        self.linear_a = nn.Linear(self.hidden_dim, self.action_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        a = self.linear_a(h)
        v = self.linear_v(h)
        return v + a - a.mean(-1, keepdim=True)


class TestModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        print(num_outputs)
        print(obs_space)
        # 
        #    hidden_dim = model_config.hidden_dim
        #else
        hidden_dim = 32
        super(TestModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        #self.linear = nn.Linear(64, 3 * 5 * 18 * 2 * 16)
        #layers = []
        #linear = SlimFC(int(np.product(self.obs_space.shape)), hidden_dim,
        #initializer=normc_initializer(1.0),
        #activation_fn="relu")
        #layers.append(linear)
        #layers.append(nn.ReLU())
        #self.model = nn.Sequential(*layers)
        #self.model = DNNEncoder(int(np.product(self.obs_space.shape)), 64, hidden_dim, 1)
        #self.a_model = SlimFC(hidden_dim, int(np.product(self.action_space.shape)))
        #self.v_model = SlimFC(hidden_dim, 1)
        print(self.obs_space)
        self.a_model = nn.Sequential(
            SlimFC(
            in_size=int(np.product(self.obs_space.shape)), 
            out_size=256,#np.product(action_space.shape)),
            initializer=normc_initializer(1.0),
            activation_fn="tanh"),
            SlimFC(256, 256,initializer=normc_initializer(1.0),activation_fn="tanh"),
            SlimFC(256, num_outputs,initializer=normc_initializer(0.01),activation_fn=None))
        self.v_model = nn.Sequential(
            SlimFC(
            in_size=int(np.product(self.obs_space.shape)),
            out_size=256,
            initializer=normc_initializer(1.0),
            activation_fn="tanh"),
            SlimFC(256, 256,initializer=normc_initializer(1.0),activation_fn="tanh"),
            SlimFC(256, 1,initializer=normc_initializer(0.01),activation_fn=None))
        self._append_free_log_std = AppendBiasLayer(num_outputs)
        #nn.init.constant_(a_model.bias, 0.0) 
        #nn.init.constant_(linear.bias, 0.0)
        #normc_initializer(1.0)
        #nn.init.constant_(self.a_model.bias, 0.0)
        #self._v = None
        ##self.linear = nn.Linear(self)
        self.fcnet= TorchFC(
            self.obs_space,
            self.action_space,
            num_outputs,
            model_config,
            name="fcnet")
        self._last_flat_in = None
    def forward(self, input_dict, state, seq_lens):
        self._output, t = self.fcnet(input_dict, state, seq_lens)
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._output = self.a_model(self._last_flat_in)
        return self._output, state 
    def value_function(self):
        return self.v_model(self._last_flat_in).squeeze(1)
        #return self.fcnet.value_function()


ModelCatalog.register_custom_model("my_torch_model", DQNModel)
ModelCatalog.register_custom_model("test_model", TestModel)
tune.run(
    "PPO",
    local_dir="~/ray_results", 
    name="test_experiment",
    config=config)
#RLlib does not work with gym registry, must redefine the environment in RLlib
# from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
#from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
# from cache_guessing_game_env_fix_impl import * # for prime and probe attack
from cache_guessing_game_env_impl import *

# (Re)Start the ray runtime.
if ray.is_initialized():
  ray.shutdown()

ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)

# Two ways of training
# 1. directly use trainer
# 2. use tune API

#####method 1
####trainer = PPOTrainer(env=CacheGuessingGameEnv, config={
####    "env_config":{},
####    "model": {
####        "use_lstm": True
####    }
####})
####trainer.train()
from ray.rllib.examples.models.custom_loss_model import CustomLossModel, \
    TorchCustomLossModel
ModelCatalog.register_custom_model(
        "custom_loss", TorchCustomLossModel)

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
ModelCatalog.register_custom_model("fcnet", TorchFC)

from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel)

from ray.rllib.examples.models.fast_model import FastModel, TorchFastModel
ModelCatalog.register_custom_model(
        "fast_model", TorchFastModel)

#method 
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv) #Fix)
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
    'gamma': 0.9, 
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
