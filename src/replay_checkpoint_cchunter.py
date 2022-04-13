'''
Author Yueying Li, Mulong Luo
Date 2022.3.23
usage: restore the ray checkpoint to replay the agent and extract the attack pattern
'''
import os
from copy import deepcopy
import gym
from starlette.requests import Request
import requests
import pprint
import ray
from ray import serve
# from test_custom_policy_diversity_works import config
import sys
import pickle 
import ray.tune as tune
from ray.rllib.agents.ppo import PPOTrainer
from test_custom_policy_diversity_works import PPOCustomTrainer
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from cchunter_wrapper import CCHunterWrapper
import pandas as pd
import numpy as np

# fork of the pandas autocorrelation_plot with added "maximum number of lags" and some other utility parameters to illustrate

# from pandas.compat import lmap
os.environ['CUDA_VISIBLE_DEVICES']='0'
tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)
tune.register_env("cchunter_wrapper", CCHunterWrapper)

config = {}
config["env_config"] = {}
def autocorrelation_plot_forked(series, ax=None, n_lags=None, change_deno=False, change_core=False, **kwds):
    """
    Autocorrelation plot for time series.
    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    n_lags: maximum number of lags to show. Default is len(series)
    kwds : keywords
        Options to pass to matplotlib plotting method
    Returns:
    -----------
    class:`matplotlib.axis.Axes`
    """
    import matplotlib.pyplot as plt
    
    n_full = len(series)
    if n_full <= 2:
      raise ValueError("""len(series) = %i but should be > 2
      to maintain at least 2 points of intersection when autocorrelating
      with lags"""%n_full)
      
    # Calculate the maximum number of lags permissible
    # Subtract 2 to keep at least 2 points of intersection,
    # otherwise pandas.Series.autocorr will throw a warning about insufficient
    # degrees of freedom
    n_maxlags = n_full - 2
    
    
    # calculate the actual number of lags
    if n_lags is None:
      # Choosing a reasonable number of lags varies between datasets,
      # but if the data longer than 200 points, limit this to 100 lags as a
      # reasonable default for plotting when n_lags is not specified
      n_lags = min(n_maxlags, 100)
    else:
      if n_lags > n_maxlags:
        raise ValueError("n_lags should be < %i (i.e. len(series)-2)"%n_maxlags)
    
    if ax is None:
        ax = plt.gca(xlim=(1, n_lags), ylim=(-1.0, 1.0))

    if not change_core:
      data = np.asarray(series)
      mean = np.mean(data)
      c0 = np.sum((data - mean) ** 2) / float(n_full)
      def r(h):
          deno = n_full if not change_deno else (n_full - h)
          return ((data[:n_full - h] - mean) *
                  (data[h:] - mean)).sum() / float(deno) / c0
    else:
      def r(h):
        return series.autocorr(lag=h)
      
    x = np.arange(n_lags) + 1
    # y = lmap(r, x)
    y = np.array([r(xi) for xi in x])
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=0.95, linestyle='--', color='grey')
    # ax.axhline(y=z95 / np.sqrt(n_full), color='grey')
    ax.axhline(y=0.0, color='black')
    # ax.axhline(y=-z95 / np.sqrt(n_full), color='grey')
    # ax.axhline(y=-z99 / np.sqrt(n_full), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    
    return ax

#from run_gym_rrllib import * # need this to import the config and PPOtrainer
PLOT_CCHUNTER = True
PLOT_LEAN = True
PLOT_DIST = False
config["env_config"]["verbose"] = 1 
config["num_workers"] = 1
#config["num_envs_per_worker"] = 1

print(config)
#tune.register_env("cache_guessing_game_env_fix", CacheGuessingGameEnv)#Fix)
#exit(0)

checkpoint_path = sys.argv[1:][0]
print(checkpoint_path[0])
#exit(-1)
#'/home/ml2558/ray_results/PPO_cache_guessing_game_env_fix_2022-01-24_21-18-203pft9506/checkpoint_000136/checkpoint-136'


i = checkpoint_path.rfind('/')
config_path = checkpoint_path[0:i] + '/../env.config'
config_path_full = checkpoint_path[0:i] + '/../env.config_full'

if os.path.isfile(config_path_full): 
    print('load env configuration in', config_path_full)
    #exit(0) 
    with open(config_path_full, 'rb') as handle:
        config = pickle.load(handle)
elif os.path.isfile(config_path): 
    print('load env configuration in', config_path)
    with open(config_path, 'rb') as handle:
        config["env_config"] = pickle.load(handle)
else:
    print('env.config not found! using default one')
    print('be careful to that the env.config matches the env which generate the checkpoint')
    print(config["env_config"])

print(config)
# config['num_workers'] = 1
config['num_envs_per_worker'] = 1
config["num_workers"] = 4


trainer = PPOTrainer(config=config)
trainer.restore(checkpoint_path)

# local_worker = trainer.workers.local_worker()
#env = local_worker.env_context


env = CCHunterWrapper(config["env_config"])


def replay_agent():
    # no cache randomization
    # rangomized inference ( 10 times)
    pattern_buffer = []
    num_guess_all = 0
    num_correct_all = 0
    pattern_dict = {}
    for victim_addr in range(env.victim_address_min, env.victim_address_max + 1):
        num_correct = 0
        num_guess = 0
        for repeat in range(1):
            if PLOT_CCHUNTER == True:
                obs = env.reset()
            else:
                obs = env.reset(victim_address=victim_addr)
            #env._randomize_cache()#"union")#"victim")
            action_buffer = []
            done = False
            legend=[]
            step = 0
            
            while done == False:
                step += 1
                #print(f"-> Sending observation {obs}")
                action = trainer.compute_single_action(obs, explore=False) 
                # deterministic inference
                
                # print the log likelihood for each action
                # see https://github.com/ray-project/ray/blob/7f1bacc7dc9caf6d0ec042e39499bbf1d9a7d065/rllib/policy/policy.py#L228
                
                local_worker = trainer.workers.local_worker()
                pp = local_worker.preprocessors["default_policy"]
                ###print(obs)
                observation = pp.transform(obs)
                episodes = None
                policy = trainer.get_policy()
                logp = policy.compute_log_likelihoods( 
                    actions = [i for i in range(0, env.action_space.n)],
                    obs_batch = [observation],
                )
                    #prev_action_batch = None,
                    #prev_reward_batch = None,
                    #action_normalized=True)
                #print(logp)
                #print(np.argmax(logp.cpu().numpy()))
                import matplotlib.pyplot as plt
                # plt.plot(logp.cpu().numpy())
                #print(action)
                # legend.append('step '+ str(step))
                #print(f"<- Received response {action}")
                obs, reward, done, info = env.step(action)
                # import pdb; pdb.set_trace()
                if 'correct_guess' in info.keys():
                    num_guess += 1
                    if info['correct_guess'] == True:
                        print(f"victim_address! {victim_addr} correct guess! {info['correct_guess']}")
                        num_correct += 1
                        correct = True
                    else:
                        correct = False
                action_buffer.append((action, obs[0]))
            # if reward > 0:
            #     correct = True
            #     num_correct += 1
            # else:
            #     correct = False
            pattern_buffer.append((victim_addr, action_buffer, num_correct))
            if pattern_dict.get((victim_addr, tuple(action_buffer), num_correct)) == None:
                pattern_dict[(victim_addr, tuple(action_buffer), num_correct)] = 1 #1 means the number of occurence
            else:
                pattern_dict[(victim_addr, tuple(action_buffer), num_correct)] += 1
            if PLOT_DIST == True:
                plt.xlabel('action label')
                plt.ylabel('logp')
                plt.legend(legend)
                plt.show()
                
        num_guess_all += num_guess
        num_correct_all += num_correct
        
    with open('temp.txt', 'a') as out:
        pprint.pprint(pattern_buffer, stream=out)
    
    print( "overall accuracy " + str(1.0 * num_correct / num_guess) )
    print( "overall bandwidth " + str(1.0 * num_guess / len(pattern_buffer[0][1])) )

    pprint.pprint(pattern_dict)
    import matplotlib.pyplot as plt
    
    address_trace = [i[0] for i in action_buffer]
    hit_trace = [i[1] for i in action_buffer][:-1]
    victim_access_or_attacker_guess_index = hit_trace.index(2)
    mask = [i != 2 for i in hit_trace] 
    address_trace_lean = [i for i, v in zip(address_trace, mask) if v]
    hit_trace_lean = [i for i, v in zip(hit_trace, mask) if v]
    import pdb; pdb.set_trace()
    
    
    if PLOT_CCHUNTER == True:
        from statsmodels.graphics.tsaplots import plot_acf
        import statsmodels.api as sm
        import pandas as pd
        if PLOT_LEAN == True:
            plt.figure()
            # Lean is the trace without the victim access or attacker guess
            plt.plot(address_trace_lean, label = 'address trace')
            plt.plot(hit_trace_lean, label = 'hit trace')
            plt.xlabel('step')
            plt.legend(['address trace', 'hit trace'])
            plt.ylabel('hit or victim access address')
            plt.savefig('cchunter_lean.png')
            
            # ax1 = plt.figure()
            data = pd.Series(hit_trace_lean)
            plt.figure()
            autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True)
            plt.savefig('cchunter_hit_trace_lean_acf.png')
            print("Figure saved as cchunter_hit_trace_lean_acf.png")
            
            plt.figure()
            data = pd.Series(address_trace_lean)
            autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True)
            plt.savefig('cchunter_address_trace_lean_acf.png')
            print("Figure saved as cchunter_address_trace_lean_acf.png")

            
        plt.plot(address_trace, label = 'address trace')
        plt.plot(hit_trace, label = 'hit trace')
        plt.xlabel('step')
        plt.ylabel('hit or victim access address')
        plt.savefig('cchunter.png')
        print("Figure saved as cchunter.png")

        plt.figure()
        data = pd.Series(address_trace)
        autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True)
        plt.savefig('cchunter_address_trace_acf.png')
        print("Figure saved as cchunter_address_trace_acf.png")
        plt.figure()
        data = pd.Series(hit_trace)
        autocorrelation_plot_forked(data, n_lags=len(data)-2, change_deno=True)
        plt.savefig('cchunter_hit_trace_acf.png')
        print("Figure saved as cchunter_hit_trace_acf.png")
            
        plt.figure()
        address_trace = pd.DataFrame(address_trace)
        plot_acf(address_trace, lags=len(address_trace)-2)
        plt.savefig('cchunter_adress_trace_acf.png')
        print("Figure saved as cchunter.png")

    print("num distinct patterns "+ str(len(pattern_dict)))
    return 1.0 * num_correct / num_guess, pattern_buffer


replay_agent()

#if __name__ == "__main__":

# cache randomization
# no randomized inference





