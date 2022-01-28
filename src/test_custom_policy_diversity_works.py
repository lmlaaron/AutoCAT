# using ray 1.9 to run
# python 3.9

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.ppo import PPOTrainer
import gym
import ray.tune as tune
from torch.nn import functional as F
from typing import Optional, Dict
import torch.nn as nn
import ray
from collections import deque
#from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import Deprecated
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    PolicyID, LocalOptimizer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import copy
import numpy as np
import sys
sys.path.append("../src")
torch, nn = try_import_torch()
from cache_guessing_game_env_impl import *

def custom_init(policy: Policy, obs_space: gym.spaces.Space, 
              action_space: gym.spaces.Space, config: TrainerConfigDict)->None:
        #pass
        policy.past_len = 5        
        policy.past_models = deque(maxlen =policy.past_len)
        policy.timestep = 0

def copy_model(model: ModelV2) -> ModelV2:
    copdied_model= TorchModelV2(
        obs_space = model.obs_space,
        action_space = model.action_space, 
        num_outputs = model.num_outputs,
        model_config = model.model_config,
        name = 'copied')
    
    return copied_model

def compute_div_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch):
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    valid_mask = torch.ones_like(values, dtype=torch.bool)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 
    print('log_probs')
    #print(log_probs)
    divs = []
    div_metric = nn.KLDivLoss(size_average=False, reduce=False)
    #div_metric = nn.CrossEntropyLoss()
    #if len(policy.past_models) > 1:
    #    assert(policy.past_models[0].state_dict() == policy.past_models[1].state_dict())
    
    for idx, past_model in enumerate(policy.past_models):
        #assert(False)
        past_logits, _ = past_model.from_batch(train_batch)
        past_values = past_model.value_function()
        past_valid_mask = torch.ones_like(past_values, dtype=torch.bool)
        past_dist = dist_class(past_logits, past_model)
        past_log_probs = past_dist.logp(train_batch[SampleBatch.ACTIONS])#.reshape(-1) 
        div = div_metric(log_probs * train_batch[Postprocessing.ADVANTAGES], past_log_probs* train_batch[Postprocessing.ADVANTAGES])
        #div = div_metric(log_probs, past_log_probs) * train_batch[Postprocessing.ADVANTAGES]
        #div = dist.multi_kl(past_dist) * train_batch[Postprocessing.ADVANTAGES]
        #assert(
        
        if idx == 0 and True:#policy.timestep % 10 == 0:
            print('past_model.state_dict()')
            #print(past_model.state_dict())
            print('model.state_dict()')
            #print(model.state_dict())
            #div = past_dist.multi_kl(dist)
            print('div')
            #print(div)
    
        div = div.mean(0)
        divs.append(div)
    print('divs')
    #print(divs)
    div_loss = 0
    div_loss_orig = 0

    for div in divs:
        div_loss += div
        div_loss_orig += div
    div_loss = div_loss / policy.past_len
    print('policy.past_len')
    print(policy.past_len)
    return div_loss

import pickle
def custom_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch) -> TensorType:
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()
    policy.timestep += 1
    #if len(policy.devices) > 1:
        # copy weights of main model (tower-0) to all other towers type
    if policy.timestep % 100 == 0:
        copied_model = pickle.loads(pickle.dumps(model))
        copied_model.load_state_dict(model.state_dict())
        policy.past_models.append(copied_model)
    
    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    
    #print('log_probs')
    #print(log_probs)
    
    pi_err = -torch.sum(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES],
                            valid_mask))
    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    values.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0
    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))
    div_loss = compute_div_loss(policy, model, dist_class, train_batch)
    total_loss = (pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"] - 1000 * div_loss )
    print('pi_err')
    #print(pi_err)
    print('value_err')
    #print(value_err)
    print('div_loss')
    print(div_loss)
    print('pi_err')
    print(pi_err)
    print('total_loss')
    print(total_loss)
    
    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err
    return total_loss


CustomPolicy = A3CTorchPolicy.with_updates(
    name="MyCustomA3CTorchPolicy",
    loss_fn=custom_loss,
    #make_model= make_model,
    before_init=custom_init)
CustomTrainer = A2CTrainer.with_updates(
    get_policy_class=lambda _: CustomPolicy)
#PPOCustomPolicy = PPOTorchPolicy.with_updates(
#    name="MyCustomA3CTorchPolicy",
#    loss_fn=custom_loss,
#    #make_model= make_model,
#    before_init=custom_init)

from typing import Dict, List, Type, Union
from ray.rllib.utils.annotations import override

class CustomPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.past_len = 5        
        self.past_models = deque(maxlen =self.past_len)
        self.timestep = 0
        super(CustomPPOTorchPolicy, self).__init__(observation_space, action_space, config)

    #@override(PPOTorchPolicy)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution],
             train_batch: SampleBatch, extern_trigger = False ) -> Union[TensorType, List[TensorType]]:
        #return custom_loss(self, model, dist_class, train_batch)
    
        self.timestep += 1
        if self.timestep % 20 == 0 and extern_trigger == False:
            copied_model = pickle.loads(pickle.dumps(model))
            copied_model.load_state_dict(model.state_dict())
            self.past_models.append(copied_model)
        
        total_loss = PPOTorchPolicy.loss(self, model, dist_class, train_batch)
        #self.past_len
        div_loss = compute_div_loss(self, model, dist_class, train_batch)
        print('total_loss')
        print(total_loss)
        print('div_loss')
        print(div_loss)
        #assert(False)
        ret_loss = total_loss - 0.1 * div_loss
        return ret_loss
        '''
        new_loss = []
        if issubclass(type(total_loss),TensorType):
            return total_loss - compute_div_loss(self, model, dist_class, train_batch)
        else:            
            for loss in total_loss:
                new_loss.append(loss - compute_div_loss(self, model, dist_class, train_batch))
            return new_loss
        '''

    def push_current_model(self):
        model = pickle.loads(pickle.dumps(self.model))
        model.load_state_dict(model.state_dict())
        self.past_models.append(model)


PPOCustomTrainer = PPOTrainer.with_updates(
    get_policy_class=lambda _: CustomPPOTorchPolicy)



#tune.run(CustomTrainer, config={"env": 'Frostbite-v0', "num_gpus":0})#, 'model': { 'custom_model': 'test_model' }})
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
# Two ways of training
# method 2b
config = {
    'env': 'cache_guessing_game_env_fix', #'cache_simulator_diversity_wrapper',
    'env_config': {
        'verbose': 0,
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
        #'custom_model': 'test_model',#'rnn', 
        #'max_seq_len': 20, 
        #'custom_model_config': {
        #    'cell_size': 32
        #   }
    }, 
    'framework': 'torch',
}

if __name__ == "__main__":
    tune.run(PPOCustomTrainer, config=config)#config={"env": 'Freeway-v0', "num_gpus":1})
