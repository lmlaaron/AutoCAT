# environment with flush option
# for flush and reload attack
# also the attacker and victim now shares the addr space
# why "flush" is needed?
# because the initial state may already contain some sets
# thus we definitely need random intialization in this case

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy
import numpy as np
import sys
import math
import random

import yaml, cache, argparse, logging, pprint
from terminaltables.other_tables import UnixTable
from cache_simulator import *

# this is a fixed version of the environment 
# the agent does not need to decide when to make a guess 
# or when to let the victim access
# the victim access in the middle of the window size
# and make a guess at the end
# this elimiates the 
# 1.length violation
# 2.double access violation
# 3.guess before acces violation
class CacheGuessingGameEnvFix(gym.Env):
  """
  Description:
    A L1 cache with total_size, num_ways 
    assume cache_line_size == 1B
  
  Observation:
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [3,                 #cache latency
      ]

  Actions:
    # action step contains four values
    # 1. access address
    # 2. what is the victim's accessed address?

  Reward: ????
   single_step: penalty
   episode: lower bound value of assertion??
   or just the guessing correctness   

  Starting state:
    fresh cache with nolines
  
  Episode termination:
    episode terminates
    at the end of the window
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, env_config={}):
    self.num_ways = 4
    self.cache_size = 8

    self.logger = logging.getLogger()
    self.fh = logging.FileHandler('log')
    self.sh = logging.StreamHandler()
    self.logger.addHandler(self.fh)
    self.logger.addHandler(self.sh)
    self.fh_format = logging.Formatter('%(message)s')
    self.fh.setFormatter(self.fh_format)
    self.sh.setFormatter(self.fh_format)
    self.logger.setLevel(logging.INFO)
    
    self.logger.info('Loading config...')
    self.config_file = open('../configs/config_simple_L1')
    self.configs = yaml.load(self.config_file, yaml.CLoader)
    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    self.window_size = 2 * self.cache_size + 4 #10 
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.state = [0, self.cache_size, 0] * self.window_size
    self.x_range = 10000
    high = np.array(
        [
            self.x_range,
        ],
        dtype=np.float32,
    )

    # action step contains two values
    self.action_space = spaces.MultiDiscrete(
      [self.cache_size,     #cache access
       2,                   #whether it is a flush
       self.cache_size      #what is the guess of the victim's access
      ])
    
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [
      3,                  #cache latency
      self.cache_size+1,  #attacker accessed address
      self.window_size,   #current steps
      ] * self.window_size
    )
    #self.observation_space = spaces.Discrete(3) # 0--> hit, 1 --> miss, 2 --> NA
    
    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(0,self.cache_size)
    
    self._randomize_cache()
    return

  def step(self, action):
    print('Step...')
    
    if action.ndim > 1:  # workaround for training and predict discrepency
      action = action[0]

    address = str(action[0])  # attacker address and victim address the same space [0, self.cache_size] 
    is_flush = action[1]  
    victim_addr = str(action[2]) # guessed victim address

    if is_flush == 1: # check whether it is a flush instruction
        self.l1.cflush(address, self.current_step)
        print("cflush " + address )
        r = 2
    elif self.l1.read(address, self.current_step).time > 500: # measure the access latency
      print("acceee " + address + " miss")
      r = 1 # cache miss
    else:
      print("access " + address + " hit"  )
      r = 0 # cache hit
    reward = 0

    self.current_step += 1
    done = False
    if self.current_step == self.window_size / 2:
      print("victim access %d" % self.victim_address)
      self.l1.read(str(self.victim_address), self.current_step)
      reward = 0
      done = False
      self.current_step += 1
    elif self.current_step == self.window_size:
      #if victim_addr == str(self.victim_address):   # addr guess accuracy
      if (int(victim_addr) - self.victim_address) % (self.cache_size/self.num_ways) == 0: # set guess accuracy
        print("correct guess " + victim_addr)
        reward = 200
        done = True
      else:
        print("wrong guess " + victim_addr )
        reward = -9999
        done = True

    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    #return r, reward, done, info
    current_step = self.current_step
    self.state = [r, action[0], current_step] + self.state 
    self.state = self.state[0:len(self.state)-3]
    #self.state = [r, action[0], current_step, victim_accessed]
    return np.array(self.state), reward, done, info

  def reset(self):
    print('Reset...')
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(0,self.cache_size-1)
    print("victim address %d" % self.victim_address)
    #return np.array([0, self.cache_size, 0, 0])
    self.state = [0, self.cache_size, 0] * self.window_size
    self._randomize_cache()
    return np.array(self.state)
    #self.state = [1000 ]
    #return np.array(self.state, dtype=np.float32)

  def render(self, mode='human'):
    return 

  def close(self):
    return

  def _randomize_cache(self):
    self.current_step = -self.cache_size
    for _ in range(self.cache_size):
      addr = random.randint(0,self.cache_size-1)
      self.l1.read(str(addr), self.current_step)
      self.current_step += 1