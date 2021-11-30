# bootstrap naive RL runs with ray[rllib]
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

from ray.rllib.models import ModelCatalog
#from models.dqn_model import DQNModel

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

class DQNModel(TorchModelV2):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_blocks: Optional[int] = 1) -> None:
        super(DQNModel, self).__init__()

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


ModelCatalog.register_custom_model("my_torch_model", DQNModel)


#RLlib does not work with gym registry, must redefine the environment in RLlib
# from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
# from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
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

#method 2
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv) #Fix)
analysis = tune.run(
    #DQNTrainer,
    PPOTrainer, 
    local_dir="~/ray_results", 
    name="test_experiment",
    #checkpoint_at_end=True,
    #stop={
    #    "episodes_total": 500,
    #},
    config={
        "num_gpus": 1,
        #"seed": 0xCC,
        "env": "cache_guessing_game_env_fix",
        #"rollout_fragment_length": 5,
        #"train_batch_size": 5,
        #"sgd_minibatch_size": 5,
        "model": { #see https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings
            #"fcnet_hiddens": [8192, 512], #, 4096, 512],
            # Activation function descriptor.
            # Supported values are: "tanh", "relu", "swish" (or "silu"),
            # "linear" (or None).
            "fcnet_activation": "relu",
            #"use_lstm": True,
            # specify our custom model 
            #"custom_model": "my_torch_model",
            # Extra kwargs to be passed to your model's c'tor
            #"custom_model_config": {
            #    "input_dim": 64,
            #    "hidden_dim": 512,
            #    "action_dim": 3 * 5 * 18 * 2 * 16,
            #},
        },
    }
)