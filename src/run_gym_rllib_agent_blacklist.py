# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
#from CacheSimulator.src.gym_cache.envs.cache_simulator_wrapper import CacheSimulatorWrapper
#from CacheSimulator.src.replay_checkpint import replay_agent
import gym
import ray
import ray.tune as tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
import sys
import copy
import torch

def replay_agent(trainer, env):
    # no cache randomization
    # rangomized inference ( 10 times)
    pattern_buffer = []
    num_guess = 0
    num_correct = 0
    for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
        for repeat in range(1):
            obs = env.reset(victim_address=victim_addr)
            action_buffer = []
            done = False
            while done == False:
                print(f"-> Sending observation {obs}")
                action = trainer.compute_single_action(obs, explore=False) # randomized inference
                print(f"<- Received response {action}")
                obs, reward, done, info = env.step(action)
                action_buffer.append((action, obs[0]))
            if reward > 0:
                correct = True
                num_correct += 1
            else:
                correct = False
            num_guess += 1
            pattern_buffer.append((victim_addr, action_buffer, correct))
    pprint.pprint(pattern_buffer)
    return 1.0 * num_correct / num_guess, pattern_buffer

if __name__ == "__main__":
    import signal
    import sys
    import pickle
    from test_custom_policy_diversity_works import *
    #tune.run(PPOTrainer, config=config)#config={"env": 'Freeway-v0', "num_gpus":1})
    from ray.tune.logger import pretty_print
    trainer = PPOCustomTrainer(config=config)
    env = CacheGuessingGameEnv(config["env_config"])
    
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    i = 0
    thre =0.98
    while True:
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))
        i += 1
        if i % 1 == 0:
            accuracy, patterns = replay_agent(trainer, env)
            if accuracy > thre:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)
                # add  this agent to blacklist
                trainer.get_policy().push_current_model()
