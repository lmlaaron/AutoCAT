# bootstrap naive RL runs with ray[rllib]
# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
import ray.tune as tune

#RLlib does not work with gym registry, must redefine the environment in RLlib
from cache_guessing_game_env_fix_impl_evict import * # for evict time attack
#from cache_guessing_game_env_fix_impl_flush import * # for flush reload attack
#from cache_guessing_game_env_fix_impl import * # for prime and probe attack 

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
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnvFix)
analysis = tune.run(
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
        },
    }
)