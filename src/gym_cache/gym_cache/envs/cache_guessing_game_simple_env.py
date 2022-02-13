
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
"""
simple guessing game has fixed length of action
always make the guess at the end
let's use three-step model first
first half let's let attacker access
then let victim just access one cache
last half let the attack access
"""
class CacheGuessingGameSimpleEnv(gym.Env):
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
    when the attacker make a guess, episode terminates
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

    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?
    self.action_space = spaces.MultiDiscrete(
      [self.cache_size,     #cache access
      2,                    #whether to make a guess
      2,                    #whether to invoke victim access
      self.cache_size       #what is the guess of the victim's access
      ])

    self.victim_address = 1 
    self.victim_accessed = False
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

    address = str(action[0]+self.cache_size)  # attacker address in range [self.cache_size, 2* self.cache_size]
    is_guess = action[1]      # check whether to guess or not
    is_victim = action[2]     # check whether to invoke victim
    victim_addr = str(action[3]) # victim address

    if self.current_step > 16:
      r = 9999#
      reward = -10000
      done = True
    else:
      if is_victim == True:
        r = 9999 #
        if self.victim_accessed == False:
          self.victim_accessed = True
          print(self.victim_address)
          self.l1.read(str(self.victim_address), self.current_step)
          self.current_step += 1
          reward = 0
          done = False
        else:
          reward = -20000
          done = True
      else:
        if is_guess == True:
          r = 9999  # 
          if self.victim_accessed == True:
            if self.victim_accessed and victim_addr == str(self.victim_address):
              reward = 1000
              done = True
            else:
              reward = -200
              done = True
          else: # guess without victim accessed first
            reward = -30000 
            done = True
        else:
          r = self.l1.read(address, self.current_step).time # measure the access latency
          self.current_step += 1
          reward = 0
          done = False

    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    return r, reward, done, info
    return np.array([0]), reward, done, info

  def reset(self):
    print('Reset...')
    
    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(0,self.cache_size) 
    return 0 #np.array([0])
    #self.state = [1000 ]
    #return np.array(self.state, dtype=np.float32)

  def render(self, mode='human'):
    return 

  def close(self):
    return