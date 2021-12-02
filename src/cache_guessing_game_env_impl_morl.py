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

class CacheGuessingGameEnv(gym.Env):
  """
  Description:
    A L1 cache with total_size, num_ways 
    assume cache_line_size == 1B
  
  Observation:
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [3,                 #cache latency
      20,                 #current steps
      2,                  #whether the victim has accessed yet
      ]

  Actions:
    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?

  Reward: ????
   single_step: penalty
   episode: lower bound value of assertion??
   or just the guessing correctness   

  Starting state:
    fresh cache with nolines
  
  Episode termination:
    when the attacker make a guess
    when there is double victim violation
    when there is length violation
    when there is guess before victim violation
    episode terminates
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
    self.window_size = self.cache_size * 2 + 8 #10 
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.state = [0, self.cache_size, 0, 0] * self.window_size
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
    self.action_spec = ['discrete', 1, [0, self.cache_size * self.cache_size * 2 * 2  ]]
    #self.action_spec = [
    #  ['discrete', 1, [0, self.cache_size-1]],
    #  ['discrete', 1, [0, 2-1]],
    #  ['discrete', 1, [0, 2-1]],
    #  ['discrete', 1, [0, self.cache_size-1]],
    #]
    
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    self.observation_space = spaces.MultiDiscrete(
      [
      3,                  #cache latency
      self.cache_size+1,  #attacker accessed address
      self.window_size + 2,   #current steps
      2,                  #whether the victim has accessed yet
      ] * self.window_size
    )
    self.state_spec = [
      ['discrete', 1, [0, 3]],
      ['discrete', 1, [0, self.cache_size + 1 ]],
      ['discrete', 1, [0, self.window_size + 2]],
      ['discrete', 1, [0, 2]],
    ] * self.window_size
 
    self.reward_spec = [[-10000, 10000], [-10000, 10000]] 
    self.current_state = np.array( [0, self.cache_size, 0, 0] * self.window_size ) 
    self.terminal = False
    #self.observation_space = spaces.Discrete(3) # 0--> hit, 1 --> miss, 2 --> NA
    
    #self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(0,self.cache_size)
    self._randomize_cache()
    return

  def step(self, action):
    print('Step...')

    ##if action.ndim > 1:  # workaround for training and predict discrepency
    ##  action = action[0]

    ##address = str(action[0]+self.cache_size)  # attacker address in range [self.cache_size, 2* self.cache_size]
    ##is_guess = action[1]      # check whether to guess or not
    ##is_victim = action[2]     # check whether to invoke victim
    ##victim_addr = str(action[3]) # victim address
    address = str(int(action / int( self.cache_size * 2 * 2) ) + self.cache_size)
    is_guess = int(action % ( self.cache_size * 2 * 2 ) / (2 * self.cache_size) )
    is_victim = int(action % ( self.cache_size * 2 ) / self.cache_size )
    victim_addr = str(action % ( self.cache_size ))

    print(self.current_step)
    print(self.window_size)
    if self.current_step > self.window_size : # if current_step is too long, terminate
      r = 2#
      print("length violation!")
      reward = -10000
      done = True
      reward = np.array([-10000, 0])
    else:
      if is_victim == True:
        r = 2 #
        #if self.victim_accessed == False:
        self.victim_accessed = True
        print("victim access %d" % self.victim_address)
        self.l1.read(str(self.victim_address), self.current_step)
        self.current_step += 1
        #reward = -10
        reward = np.array( [0, 0] )
 
        done = False
        #else:               # if double victim access, huge penalty 
        #  print("double access")
        #  reward = -20000
        #  done = True
      else:
        if is_guess == True:
          r = 2  # 
          #if self.victim_accessed == True:
          if self.victim_accessed and victim_addr == str(self.victim_address):
              print("correct guess " + victim_addr)
              #reward = 200
              reward = np.array( [0, 1] )
              done = True
          else:
              print("wrong guess " + victim_addr )
              #reward = -9999
              reward = np.array( [0, 0] )
              done = True
          #else:         # guess without victim accessed first, huge penalty
          #  print("guess without access violation")
          #  reward = -30000 
          #  done = True
        else:
          if self.l1.read(address, self.current_step).time > 500: # measure the access latency
            print("acceee " + address + " miss")
            r = 1 # cache miss
          else:
            print("access " + address + " hit"  )
            r = 0 # cache hit
          self.current_step += 1
          #reward = -1 
          reward = np.array( [-1, 0] )
          done = False

    #return observation, reward, done, info
    info = {}
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    #return r, reward, done, info
    current_step = self.current_step
    if self.victim_accessed == True:
      victim_accessed = 1
    else:
      victim_accessed = 0
    self.state = [r, int(address)-self.cache_size, current_step, victim_accessed] + self.state 
    self.state = self.state[0:len(self.state)-4]
    #self.state = [r, action[0], current_step, victim_accessed]
    return np.array(self.state), reward, done #, info

  def reset(self):
    print('Reset...')
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    self.victim_address = random.randint(0,self.cache_size-1) 
    print("victim address %d", self.victim_address)
    #return np.array([0, self.cache_size, 0, 0])
    self.state = [0, self.cache_size, 0, 0] * self.window_size
    self._randomize_cache()
    self.current_state = np.array(self.state)
    self.terminal = False
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

  def observe(self):
    ''' reset the enviroment '''
    return self.current_state