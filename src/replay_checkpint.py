'''
Author Mulong Luo
Date 2022.1.24
usage: resotre the ray checkpoint to replay the agent and extract the attack pattern
'''

import gym
from starlette.requests import Request
import requests

import ray
from ray import serve
from run_gym_rrllib import * # need this to import the config and PPOtrainer

print(config)
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
#exit(0)
from ray.rllib.agents.ppo import PPOTrainer
trainer0 = PPOTrainer(config=config)
trainer0.train()
checkpoint_path = trainer0.save()
trainer = PPOTrainer(config=config)
trainer.restore(checkpoint_path)


env = CacheGuessingGameEnv(config["env_config"])
obs = env.reset()
for _ in range(10):
    print(f"-> Sending observation {obs}")
    action = trainer.compute_single_action(obs)
    print(f"<- Received response {action}")
    obs, reward, done, info = env.step(action)