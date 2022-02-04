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

def replay_agent(trainer, env, randomize_init=False, non_deterministic=False):
    # no cache randomization
    # rangomized inference ( 10 times)
    pattern_buffer = []
    num_guess = 0
    num_correct = 0
    if randomize_init == False and non_deterministic == False:
        repeat_times = 1
    else:
        repeat_times = 50

    for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
        for repeat in range(repeat_times):
            obs = env.reset(victim_address=victim_addr)
            if randomize_init:
                env._randomize_cache("union")
            action_buffer = []
            done = False
            while done == False:
                print(f"-> Sending observation {obs}")
                action = trainer.compute_single_action(obs, explore = non_deterministic) # randomized inference
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
    #tune.register_env("cache_guessing_game_env_fix", CacheSimulatorMultiGuessWrapper)
    #from run_gym_rllib_simd import *
    #config['num_workers'] = 6
    #config['num_envs_per_worker']= 2
    env = CacheGuessingGameEnv(config["env_config"])
    #env = CacheSimulatorMultiGuessWrapper(config["env_config"]) 
    trainer = PPOCustomTrainer(config=config)
    
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        policy = trainer.get_policy()
        for model in policy.past_models:
            print(model.state_dict()['_hidden_layers.1._model.0.weight'])
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    i = 0
    thre =0.95 #0.98
    #buf = []

    while True:
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))
        i += 1
        if i % 1 == 0:  # give enought interval to achieve small verificaiton overhead
            accuracy, patterns = replay_agent(trainer, env, randomize_init=True, non_deterministic=True)
            # just with lower reward
            # HOW TO PREVENT THE SAME AGENT FROM BEING ADDED TWICE????
            # HOW TO TELL IF THEY ARE CONSIDERED THE SAME AGENT?
            # HOW TO FORCE TRAINER TO KNOW THAT THEY ARE STILL DISCOVERING THE SAME AGENT???
            if accuracy > thre:
                # if the agent is different from the known agent
                policy = trainer.get_policy()
                if policy.existing_agent(env, trainer) == False:
                    checkpoint = trainer.save()
                    print("checkpoint saved at", checkpoint)
                    # this agent might have high accuracy but 
                    # it ccould be that it is still the same agent
                    # add  this agent to blacklist
                    trainer.get_policy().push_current_model()
                    #buf.append(copy.deepcopy(trainer.get_weights()))

    policy = trainer.get_policy()
    for model in policy.past_models:
        print(model.state_dict()['_hidden_layers.1._model.0.weight'])
    #for weight in policy.past_weights:
    #    print(weight['_value_branch._model.0.bias'])
        #print(weight['default_policy']['_value_branch._model.0.bias'])
    #print(policy.model.state_dict()['_hidden_layers.1._model.0.weight'])

    #for w in buf:
    #    print(w['default_policy']['_value_branch._model.0.bias'])