# look at https://github.com/ray-project/ray/blob/ea2bea7e309cd60457aa0e027321be5f10fa0fe5/rllib/examples/custom_env.py#L2
#from CacheSimulator.src.gym_cache.envs.cache_simulator_wrapper import CacheSimulatorWrapper
import gym
import ray
import ray.tune as tune
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
import sys
import copy
import torch
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

#from ray.rllib.offline import JsonReader
from torch.autograd import Variable




# the actual model used by the RLlib
from collections import deque
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
        self.past_len = 5
        self.past_models = deque(maxlen=self.past_len)
        self.past_mean_rewards = deque(maxlen=self.past_len)
        self._last_flat_in = None
        #self.reader = JsonReader(self.input_files)
        #self.div_coef = 
        #self.rescale_factor = 
        #self.recent_model = []

    def forward(self, input_dict, state, seq_lens):
        #if obs[-1] > 0.99:
        #    self.recent_model.append((copy.deepcopy(self.a_model), copy.deepcopy(self.v_model)))
        #    if len(self.recent_model) > 5:
        #        self.recent_model.pop()
        obs = input_dict["obs_flat"].float()
        return self._forward(obs, input_dict, state, seq_lens)

    def _forward(self, obs, input_dict, state, seq_lens):
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._output = self.a_model(self._last_flat_in)
        return self._output, state 

    def value_function(self):
        return self.v_model(self._last_flat_in).squeeze(1)

    # modified from https://github.com/ray-project/ray/blob/ac3371a148385027cf034e152d50f528acb853de/rllib/examples/models/custom_loss_model.py
    '''
    let's assume log_prob in diveristy and logits in ray are the same thing
    '''
    def custom_loss(self, policy_loss, loss_inputs):
        #div_loss, div_loss_orig = compute_div_loss(states, log_probs)
        """Calculates a custom loss on top of the given policy_loss(es).
        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs (TensorStruct): Struct of np.ndarrays holding the
                entire train batch.
        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        # Get the next batch from our input files.
        #batch = self.reader.next()
        batch = loss_inputs
        # Define a secondary loss by building a graph copy with weight sharing.

        #batch["obs"]
        #batch["obs"].float()
        #torch.from_numpy(batch["obs"])
        #torch.from_numpy(batch["obs"]).float()
        #batch["obs"].float().to(policy_loss[0].device)
        #torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device)
        obs = restore_original_dimensions(
            batch["obs"].float().to(policy_loss[0].device),
            #torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device),
            self.obs_space,
            tensorlib="torch")
        logits, states = self.forward({"obs_flat": obs}, [], None)
        #logits, states = self.forward(loss_inputs, [], None)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        # ae_loss = squared_diff(
        #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        #compute the diverity losss
        # treating logits as log_prob; maybe need correction
        #div_loss, div_loss_orig = self.compute_div_loss(states, logits)
        
        #now compute
        div_metric = nn.KLDivLoss(size_average=False, reduce=False)
        div_loss = 0
        past_ratios = [1.0 for r in range(len(self.past_models))] 
        divs = []
        for idx, past_model in enumerate(self.past_models):
           target_probs, states = past_model.forward({"obs_flat": obs}, [], None)        
           div = div_metric(logits, target_probs).sum(1)
           div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
           div = div.mean(0)
           divs.append(div)
        
        #self.div_loss_metric = div_loss.item()
        #self.policy_loss_metric = np.mean(
        #    [loss.item() for loss in policy_loss])

        divs_sort_idx = np.argsort([d.data[0] for d in divs])
        div_loss_orig = 0
        for idx in divs_sort_idx:
            #if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
            #    div_loss += float(-past_ratios[idx]) * divs[idx]
            #elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
            #    div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            #elif self.rel and self.past_mean_reward_min != self.current_perf:
            #    div_loss += float(-past_ratios[idx]) * divs[idx]
            #else:
            div_loss += divs[idx]
            div_loss_orig += divs[idx]

        #if self.use_clip_div_loss:
        #    div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        #else:
        div_loss = div_loss / self.past_len# float(len(self.past_models))
        # Add the imitation loss to each already calculated policy loss term.
        # Alternatively (if custom loss has its own optimizer):
        # return policy_loss + [10 * self.imitation_loss]
        #return [loss_ - self.rescale_factor * self.div_coef * div_loss for loss_ in policy_loss]
        return [ loss_ - div_loss for loss_ in policy_loss ]

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.div_loss_metric,
        }

    '''
    Author: Zhang-Wei Hong 
    Email: williamd4112@gapp.nthu.edu.tw>
    Description: source code for paper
    “Diversity-driven exploration strategy for deep reinforcement learning” (NIPS 2018) 
    '''        
    '''
    def compute_div_loss(self, states, log_probs):
        div_metric = nn.KLDivLoss(size_average=False, reduce=False)
        # Div loss
        div_loss = 0
        #if args.use_neg_ratio or args.use_ratio or args.rel:
        #    max_perf = current_max_perf if args.use_history_max else current_perf
        #    min_perf = current_min_perf if args.use_history_max else current_perf
        #    past_mean_reward_max = max(max(past_mean_rewards), max_perf)
        #    past_mean_reward_min = min(min(past_mean_rewards), min_perf)
        #    past_mean_reward_rng = past_mean_reward_max - past_mean_reward_min + 1e-9

        #if args.use_neg_ratio:
        #    past_ratios = [((r - past_mean_reward_min) / past_mean_reward_rng) * 2 - 1 for r in past_mean_rewards]
        #elif args.use_ratio:
        #    past_ratios = [((r - past_mean_reward_min) / past_mean_reward_rng)  for r in past_mean_rewards]
        #elif args.rel:
        #    past_ratios = [((r - current_perf) / past_mean_reward_rng)  for r in past_mean_rewards]
        #else:
        #    past_ratios = [1.0 for r in range(len(past_models))]
        past_ratios = [1.0 for r in range(len(self.past_models))]

        divs = []
        for idx, past_model in enumerate(self.past_models):
            past_model.forward(states,)?????
            _, target_inds = t_probs.max(1)
            target_inds = target_inds.data
            action_size = t_probs.size()
            target_probs = Variable(torch.zeros(action_size[0], action_size[1]).cuda().scatter_(1, target_inds.unsqueeze(1), 1.0), requires_grad=False)

            div = div_metric(log_probs, target_probs).sum(1)
            div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
            div = div.mean(0)
            divs.append(div)

        divs_sort_idx = np.argsort([d.data[0] for d in divs])

        div_loss_orig = 0
        for idx in divs_sort_idx:
            if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            elif self.rel and self.past_mean_reward_min != self.current_perf:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            else:
                div_loss += divs[idx]
            div_loss_orig += divs[idx]

        if self.use_clip_div_loss:
            div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        else:
            div_loss = div_loss / float(len(self.past_models))

        return div_loss, div_loss_orig / float(len(self.past_models))
        

        for idx, past_model in enumerate(self.past_models):
            temperature = args.temp
            actions = Variable(rollouts.actions.view(-1, action_shape), volatile=True)
            _, t_action_log_probs, _, t_log_probs, t_probs = past_model.evaluate_actions(Variable(states.data, volatile=True), actions, temperature=temperature)
            target_log_probs = t_log_probs
            target_log_probs = Variable(target_log_probs.data, requires_grad=False)

            if self.div_soft:
                target_probs = Variable(t_probs.data, requires_grad=False)
            else:
                _, target_inds = t_probs.max(1)
                target_inds = target_inds.data
                action_size = t_probs.size()
                target_probs = Variable(torch.zeros(action_size[0], action_size[1]).cuda().scatter_(1, target_inds.unsqueeze(1), 1.0), requires_grad=False)
            div = div_metric(log_probs, target_probs).sum(1)
            div = torch.clamp(div, min=-self.div_threshold, max=self.div_threshold)
            div = div.mean(0)
            divs.append(div)

        divs_sort_idx = np.argsort([d.data[0] for d in divs])
        if self.knn != 0:
            divs_sort_idx = divs_sort_idx[:self.knn]

        div_loss_orig = 0
        for idx in divs_sort_idx:
            if self.use_neg_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            elif self.use_ratio and self.past_mean_reward_min != self.past_mean_reward_max:
                div_loss += (1.0 - float(past_ratios[idx])) * divs[idx]
            elif self.rel and self.past_mean_reward_min != self.current_perf:
                div_loss += float(-past_ratios[idx]) * divs[idx]
            else:
                div_loss += divs[idx]
            div_loss_orig += divs[idx]

        if self.use_clip_div_loss:
            div_loss = torch.clamp(div_loss / float(len(self.past_models)), min=-self.div_max, max=self.div_max)
        else:
            div_loss = div_loss / float(len(self.past_models))

        return div_loss, div_loss_orig / float(len(self.past_models))
        '''

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
          Description:
            A method to detect potential cache attacks based on autocorrelogram
            An autocorrelogram is a chart showing the autocorrelation coefficient values 
            for a sequence of lag values. An oscillation pattern is inferred when the autocorrelation 
            coefficient shows significant periodicity with peaks sufficiently high for certain lag values 
            (i.e., the values of X correlates highly with itself at lag distances of k1, k2 etc.).
            
            Cache based timing channels rely on the latency of events to perform timing modulation.
            
            Steps:
            1. First construct a conflict event train (cache conflict misses)
            2. The conflict misses that are observed within each observation window (typically one OS time quantum Q) 
            are used to construct a conflict miss event train plot.
            3. Compute the auto-correlation coefficients for the conflict miss event train plot at the lag value of (Cache sets)
            
            Note:
            To evade detection, the trojan/spy may (with some effort) may deliberately introduce noise through 
            creating cache conflicts with other contexts. This may potentially lower autocorrelation coefficients, 
            but we note that the trojan/spy may face a much bigger problem in reliable transmission due to higher 
            variability in cache access latencies.
            
            Parameters:
            1. self.cc_hunter_episode: The length of the episode to be used for detecting cache conflicts and officially call it done
            2. self.cc_hunter_episode_scale: The scaling factor for the cchunter agent on top of the full victim address range
    
'''
class CCHunterWrapper(gym.Env): #TODO(LISA)
    def __init__(self, env_config, keep_latency=True):
        self.keep_latency = keep_latency
        self.env_config = env_config
        self.cc_hunter_episode_scale = 10
        self._env = CacheGuessingGameEnv(env_config)
        #     in this case the pattern has to be stored
        self.validation_env = CacheGuessingGameEnv(env_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_buffer = []
        #self.latency_buffer = []
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        self.pattern_init_state_buffer = []
        self.victim_address_min = self._env.victim_address_min
        self.victim_address_max = self._env.victim_address_max
        self.attacker_address_max = self._env.attacker_address_max
        self.attacker_address_min = self._env.attacker_address_min
        self.victim_address = self._env.victim_address
        self.cc_hunter_episode = self.cc_hunter_episode_scale * (self.victim_address_max - self.victim_address_min + 1)
     
    def reset(self):
        self.action_buffer = []
        # import pdb; pdb.set_trace()
        self.cc_hunter_length = self._env.cache_size
        self.cc_hunter_buffer = []
        self.validation_env.reset() # Is this needed?
        rtn = self._env.reset(victim_address = -1, if_only_reinitialize_rl_related_variables = True)
        self.pattern_init_state = (copy.deepcopy(self._env.l1), self._env.victim_address)
        # print("number of found patterns:" + str(len(self.pattern_buffer)))
        return rtn
    
    def calculate_autocorrelation_coefficients(self, x, lags):
        """
        Calculate the autocorrelation coefficients for the given data and lags.
        """
        n = len(x)
        series = pd.Series([i[0] for i in x])
        # print("Series is:\n", series)
        # print("series correlation:\n",series.autocorr())
        # data = np.asarray(x)
        # print(data)
        # x_mean = np.mean(data)
        # y_mean = np.mean(data)
        # rho = np.zeros(lags)
        # for lag in range(0, lags):
        #     x_m = data[:-lag]
        #     y_m = data[lag:]
        #     x_m -= x_mean
        #     y_m -= y_mean
        #     rho[lag] = np.sum(x_m * y_m) / (n - lag)
        return series.autocorr(lags)
    
    def autocorrelogram(self, x, plot_autocorrelogram=False):
        autocorrelogram = []
        for i in range(self._env.cache_size * 100): # we may also consider a wider range of lags
            autocorrelogram.append(self.calculate_autocorrelation_coefficients(x, i))
        # plot the autocorrelogram
        if plot_autocorrelogram:
            import matplotlib.pyplot as plt
            plt.plot(autocorrelogram)
        return autocorrelogram
    
    def cc_hunter_attack(self, x, threshold = 0.8, plot_autocorrelogram=False):
        autocorrelogram = self.autocorrelogram(x, plot_autocorrelogram)
        # detect the attack
        cc_hunter_attack = False
        cc_hunter_attack_list = []
        for i in range(0,self._env.cache_size):
            # Note: we should not consider the auto-correlation coefficient at lag 0
            if autocorrelogram[i] > threshold:
                cc_hunter_attack_list.append(i)
        cc_hunter_attack = True if len(cc_hunter_attack_list) > 1 else False 
        #Future: tune the periodicity threshold. It is currently set to 0.8, 
        # which is a reasonable threshold for detecting cache conflicts pattern repetition
        print(cc_hunter_attack)
        return cc_hunter_attack, cc_hunter_attack_list
        
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
    step function
    '''        
    def step(self,action):
        state, reward, done, info = self._env.step(action) 
        # TODO(LISA): when done, the agent will call reset()
        # (LISA) state gives a history of the [hit_info, addr, ****]
        # add to buffer 
        # done is true if the game is over, and cache will be reset, and the episode will start again
        # if we want to keep the cache state after the game is over, we need to override the done signal
        
        #state = [state self._env.calc_correct_rate()]
        if self.keep_latency == True:
            latency = state[0]
        else:
            latency = -1
        self.cc_hunter_buffer.append((latency, state[1])) 
        # import pdb; pdb.set_trace()
        
        # Get the past trace of (latency, addr) appended to the cc_hunter_buffer

        self.action_buffer.append((action,latency)) #latnecy is part of the attack trace
        #make sure the current existing correct guessing rate is high enough beofre 
        # altering the reward
        if done == False:
            return state, reward, done, info
        else: # DONE
            length = len(self.cc_hunter_buffer)
            self._env.reset(victim_address = -1, if_only_reinitialize_rl_related_variables = True)
            print('cc hunter buffer length is', length)
            if length < self.cc_hunter_episode:
                done = False
            else:
                if self.cc_hunter_attack(self.cc_hunter_buffer) == True:
                    reward -= 0 # TODO
            return state, reward, done, info
            
    
    


ray.init(include_dashboard=False, ignore_reinit_error=True, num_gpus=1)


# Two ways of training
config = {
    'env': 'cchunter_wrapper', #'cache_simulator_diversity_wrapper',
    'env_config': {
        'verbose': 1,
        "force_victim_hit": False,
        'flush_inst': False,
        "allow_victim_multi_access": False,
        "attacker_addr_s": 0,
        "attacker_addr_e": 3,#3,
        "victim_addr_s": 0,
        "victim_addr_e": 1,#1,
        "reset_limit": 1,
        "cache_configs": {
                # YAML config file for cache simulaton
            "architecture": {
              "word_size": 1, #bytes
              "block_size": 1, #bytes
              "write_back": True
            },
            "cache_1": {#required
              "blocks": 4,#2, 
              "associativity": 1,#2,  
              "hit_time": 1 #cycles
            },
            "mem": {#required
              "hit_time": 1000 #cycles
            }
        }
    }, 
    #'gamma': 0.9, 
    'num_gpus': 1, 
    'num_workers': 8, 
    'num_envs_per_worker': 2, 
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

tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
tune.register_env("cchunter_wrapper", CCHunterWrapper)

#trainer = PPOTrainer(config=config)
#exit(0)
if __name__ == "__main__":
    import signal
    import sys

    #tune.run(PPOTrainer, config=config)#config={"env": 'Freeway-v0', "num_gpus":1})
    from ray.tune.logger import pretty_print
    trainer = PPOTrainer(config=config)
    # trainer = SACTrainer(config=config)

    # def signal_handler(sig, frame):
    #     print('You pressed Ctrl+C!')
    #     checkpoint = trainer.save()
    #     print("checkpoint saved at", checkpoint)
    #     sys.exit(0)
        
    def signal_handler(sig, frame):
        import pickle
        print('You pressed Ctrl+C!')
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        i = checkpoint.rfind('/')
        config_name = checkpoint[0:i] + '/../env.config' 
        config_name_full = checkpoint[0:i] + '/../env.config_full'
        print("env config saved at ", config_name)
        with open(config_name, 'wb') as handle:
            pickle.dump(config["env_config"], handle)
        with open(config_name_full, 'wb') as handle:
            pickle.dump(config, handle)
        # policy = trainer.get_policy()
        # for model in policy.past_models:
        #     print(model.state_dict()['_hidden_layers.1._model.0.weight'], protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit(0)



    signal.signal(signal.SIGINT, signal_handler)
    
    while True:
       # Perform one iteration of training the policy with PPO
       result = trainer.train()
       print(pretty_print(result))

       #if True: #i % 100 == 0:
       #    checkpoint = trainer.save()
       #    print("checkpoint saved at", checkpoint)


    # restore checkpoint
    #trainer2 = PPOTrainer(config=config)
    #trainer2.restore(checkpoint)
