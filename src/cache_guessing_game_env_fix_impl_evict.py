#environment for evict and time option
# the victim access time is known to the attacker

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy
import numpy as np
import sys
import math
import random
import os
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
    self.config_file = open(os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1')
    self.configs = yaml.load(self.config_file, yaml.CLoader)
    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    self.window_size = 2 * self.cache_size + 4 #10 
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.state = [0, self.cache_size,  0] * self.window_size
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
       2,                   #whether it is victim access
       self.cache_size      #what is the guess of the victim's access
      ])
    
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [
      3,                  #cache latency
      self.cache_size+1,  #attacker accessed address
      #2,                   #whether it is victim access
      self.window_size + 2,   #current steps
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
    #print('Step... %d' % self.current_step)
    
    if action.ndim > 1:  # workaround for training and predict discrepency
      action = action[0]

    address = str(action[0] + self.cache_size)  # attacker address and victim address the same space [0, self.cache_size] 
    is_victim = action[1]  
    victim_addr = str(action[2]) # guessed victim address
    self.current_step += 1
 
    done = False
    if is_victim == 1 and self.current_step < self.window_size: # check whether it is victim access
      #self.current_step += 1
      if self.l1.read(str(self.victim_address), self.current_step).time > 500:
        print("victim accesses %d " % self.victim_address + " miss")
        r = 1 # cache miss
      else:
        print("victim accesses %d " % self.victim_address + " hit")
        r = 0 # cache miss
      reward = 0
      done = False
    elif self.current_step >= self.window_size - 1:
      #if victim_addr == str(self.victim_address):   # addr guess accuracy
      if (int(victim_addr) - self.victim_address) % (self.cache_size/self.num_ways) == 0: # set guess accuracy
        print("correct guess " + victim_addr)
        reward = 200
        done = True
      else:
        print("wrong guess " + victim_addr )
        reward = -9999
        done = True
      r = 2
    else: # normal attacker access
      #self.current_step += 1
      self.l1.read(address, self.current_step) # attacker access, for evict time attack, its timing is not known
      print("attacker access " + address )
      r = 2
      reward = 0
      done = False

    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    #return r, reward, done, info

    current_step = self.current_step
    self.state = [r, action[0], current_step] + self.state 
    self.state = self.state[0:len(self.state)-3]
    #print(np.array(self.state))
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
    self.current_step = -self.cache_size * 2 
    for _ in range(self.cache_size * 2):
      addr = random.randint(0,self.cache_size * 2)
      if addr == self.cache_size * 2:
        self.l1.cflush(str(addr), self.current_step)
      else:
        self.l1.read(str(addr), self.current_step)
      self.current_step += 1