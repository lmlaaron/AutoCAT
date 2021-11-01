
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy
import numpy as np
import sys
import math
import random
sys.path.insert(0, '../../../')

import yaml, cache, argparse, logging, pprint
from terminaltables.other_tables import UnixTable
from cache_simulator import *

class CacheEnv(gym.Env):
  """
  Description:
    A L1 cache with total_size, num_ways 
    assume cache_line_size == 1B
  
  Observation:
    currently assume ternary:
    (-1, H, L)
    -1: sender access
    H: receiver cache miss
    L: receiver cache hit
    But later could be continuous more multi value

  Actions:
    Type: Discrete(2 * total_size)
    0 - total_size: sender access
    total_size - 2 * total_size: receiver access

  Reward: ????
   single_step: penalty
   episode: lower bound value of assertion??
   or just the guessing correctness   

  Starting state:
    fresh cache with nolines
  
  Episode termination:
    after some threshold of accuracy for most recent 100 predictions
  """
  metadata = {'render.modes': ['human']}

  def __init__(self):
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
    self.config_file = open('/home/ml2558/CacheSimulator/configs/config_simple_L1')
    self.configs = yaml.load(self.config_file)
    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    
    self.x_range = 10000
    high = np.array(
        [
            self.x_range,
        ],
        dtype=np.float32,
    )

    #self.action_space = spaces.Discrete(self.cache_size * 2)
    # let's define the action_space as consecutive 6 accesses plus one guess of 
    # the latency of the last access
    # and define the observation space as just one latency 
    self.action_space = spaces.MultiDiscrete(
      [self.cache_size * 2,
      self.cache_size * 2,
      self.cache_size * 2,
      self.cache_size * 2,
      self.cache_size * 2,    
      self.cache_size * 2,
      2 #10000 #guess of the latency
    ])
    
    self.observation_space = spaces.Discrete(10000)
    #self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    return

  def step(self, action):
    print('Step...')

    if action.ndim > 1:  # workaround for training and predict discrepency
      action = action[0]

    for i in range(0,6):
      address = str(action[i])  # six consecutive accesses    
      r = self.l1.read(address, self.current_step) # measure the access latency

    #check the 
    guess = action[6]
    if r.time > 10:  #cache miss
      ground_truth = 1
    else:           # cache hit
      ground_truth = 0

    print(ground_truth)
    print(guess)

    if ground_truth == guess:
      reward = 1
      done = True
    else:
      reward = 0
      done = False
    
    self.current_step += 1
    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    return r.time, reward, done, info
  
  def reset(self):
    print('Reset...')
    
    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    return 0
    #self.state = [1000 ]
    #return np.array(self.state, dtype=np.float32)

  def render(self, mode='human'):
    return 

  def close(self):
    return