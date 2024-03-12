# Author: Mulong Luo
# date 2021.12.3
# description: environment for study RL for side channel attack
from collections import deque

import numpy as np
import random
import os
import yaml, logging
import sys
from itertools import permutations

import gym
from gym import spaces #this works for Rlmeta
# from gymnasium.spaces import Discrete, Box #this works for torchRL

from omegaconf.omegaconf import open_dict
sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cache_simulator import *
import replacement_policy

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

  Observation:
    # observation conatains four values
    # 1. cache miss or hit
    # 2. victim accessed or not
    # 3. original action
    # 4. step count

  Actions:
    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?

  Reward:

  Starting state:
    fresh cache with nolines
  
  Episode termination:
    when the attacker make a guess
    when there is length violation
    when there is guess before victim violation
    episode terminates
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, env_config={
   "length_violation_reward":-10000,
   "double_victim_access_reward": -10000,
   "force_victim_hit": False,
   "victim_access_reward":-10,
   "correct_reward":200,
   "wrong_reward":-9999,
   "step_reward":-1,
   "window_size":0,
   "attacker_addr_s":4,
   "attacker_addr_e":7,
   "victim_addr_s":0,
   "victim_addr_e":3,
   "flush_inst": False,
   "allow_victim_multi_access": True,
   "verbose":0,
   "reset_limit": 1,    # specify how many reset to end an epoch?????
   "cache_configs": {
      # YAML config file for cache simulaton
      "architecture": {
        "word_size": 1, #bytes
        "block_size": 1, #bytes
        "write_back": True
      },
      "cache_1": {#required
        "blocks": 8, 
        "associativity": 2,  
        "hit_time": 1 #cycles
      },
      "mem": {#required
        "hit_time": 1000 #cycles
      }
    }
  }
):

    #some data for pettig zoo
    temp_list = []
    self.Data1 = np.array(temp_list)
    self.Data2 = {}
    # prefetcher
    # pretetcher: "none" "nextline" "stream"
    # cf https://my.eng.utah.edu/~cs7810/pres/14-7810-13-pref.pdf
    self.prefetcher = env_config["prefetcher"] if "prefetcher" in env_config else "none"

    # remapping function for randomized cache
    self.rerandomize_victim = env_config["rerandomize_victim"] if "rerandomize_victim" in env_config else False
    self.ceaser_remap_period = env_config["ceaser_remap_period"] if "ceaser_remap_period" in env_config else 200000
    self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
    self.force_victim_hit =env_config["force_victim_hit"] if "force_victim_hit" in env_config else False
    self.length_violation_reward = env_config["length_violation_reward"] if "length_violation_reward" in env_config else -10000
    self.victim_access_reward = env_config["victim_access_reward"] if "victim_access_reward" in env_config else -10
    self.victim_miss_reward = env_config["victim_miss_reward"] if "victim_miss_reward" in env_config else -10000 if self.force_victim_hit else self.victim_access_reward
    self.double_victim_access_reward = env_config["double_victim_access_reward"] if "double_victim_access_reward" in env_config else -10000
    self.allow_victim_multi_access = env_config["allow_victim_multi_access"] if "allow_victim_multi_access" in env_config else True
    self.correct_reward = env_config["correct_reward"] if "correct_reward" in env_config else 200
    self.wrong_reward = env_config["wrong_reward"] if "wrong_reward" in env_config else -9999
    self.step_reward = env_config["step_reward"] if "step_reward" in env_config else 0
    self.reset_limit = env_config["reset_limit"] if "reset_limit" in env_config else 1
    self.cache_state_reset = env_config["cache_state_reset"] if "cache_state_reset" in env_config else True
    window_size = env_config["window_size"] if "window_size" in env_config else 0
    attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
    attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
    victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
    victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
    flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False
    self.verbose = env_config["verbose"] if "verbose" in env_config else 0
    self.logger = logging.getLogger()
    self.fh = logging.FileHandler('log')
    self.sh = logging.StreamHandler()
    self.logger.addHandler(self.fh)
    self.logger.addHandler(self.sh)
    self.fh_format = logging.Formatter('%(message)s')
    self.fh.setFormatter(self.fh_format)
    self.sh.setFormatter(self.fh_format)
    self.logger.setLevel(logging.INFO)
    if "cache_configs" in env_config:
      #self.logger.info('Load config from JSON')
      self.configs = env_config["cache_configs"]
    else:
      self.config_file_name = os.path.dirname(os.path.abspath(__file__))+'/../configs/config_simple_L1'
      self.config_file = open(self.config_file_name)
      self.logger.info('Loading config from file ' + self.config_file_name)
      self.configs = yaml.load(self.config_file, yaml.CLoader)
    self.vprint(self.configs)

    self.num_ways = self.configs['cache_1']['associativity'] 
    self.cache_size = self.configs['cache_1']['blocks']
    
    if "rep_policy" not in self.configs['cache_1']:
      self.configs['cache_1']['rep_policy'] = 'new_plru_pl'
    '''
    with open_dict(self.configs):
        self.configs['cache_1']['prefetcher'] = self.prefetcher
    print(self.prefetcher)
    #assert(False)
    '''
    if window_size == 0:
      self.window_size = self.cache_size * 8 + 8 #10 
      #self.window_size = self.cache_size * 4 + 8 #10 
    else:
      self.window_size = window_size
    self.feature_size = 4
    self.hierarchy = build_hierarchy(self.configs, self.logger)
    #self.state = [0, self.cache_size, 0, 0] * self.window_size
    # self.state = [-1, -1, -1, -1] * self.window_size # Xiaomeng
    #self.state = [0, self.cache_size, 0, 0, 0] * self.window_size

    self.state = deque([[-1, -1, -1, -1]] * self.window_size)
    self.step_count = 0

    self.attacker_address_min = attacker_addr_s
    self.attacker_address_max = attacker_addr_e
    self.attacker_address_space = range(self.attacker_address_min,
                                  self.attacker_address_max + 1)  # start with one attacker cache line
    self.victim_address_min = victim_addr_s
    self.victim_address_max = victim_addr_e
    self.victim_address_space = range(self.victim_address_min,
                                self.victim_address_max + 1)  #

    # for randomized mapping rerandomization
    #perm = permutations(list(range(self.victim_address_min, self.victim_address_max + 1 )))
    if self.rerandomize_victim == True:
      # perm = permutations(list(range(min(self.victim_address_min, self.attacker_address_min), max(self.victim_address_max, self.attacker_address_max) + 1 )))
      # self.perm = list(perm)
      addr_space = max(self.victim_address_max, self.attacker_address_max) + 1
      self.perm = [i for i in range(addr_space)]
    
    # keeping track of the victim remap length
    self.ceaser_access_count = 0
    self.mapping_func = lambda addr : addr
    self.remap()

    self.flush_inst = flush_inst
    self.reset_time = 0
    # action step contains four values
    # 1. access address
    # 2. whether to end and make a guess now?
    # 3. whether to invoke the victim access
    # 4. if make a guess, what is the victim's accessed address?
    ####self.action_space = spaces.MultiDiscrete(
    ####  [self.cache_size,     #cache access
    ####  2,                    #whether to make a guess
    ####  2,                    #whether to invoke victim access
    ####  2,                    #whether it is a flush inst 
    ####  self.cache_size       #what is the guess of the victim's access
    ####  ])
    
    ######self.action_space = spaces.Discrete(
    ######  len(self.attacker_address_space) * 2 * 2 * 2 * len(self.victim_address_space)
    ######)
    # using tightened action space
    if self.flush_inst == False:
      # one-hot encoding
      if self.allow_empty_victim_access == True:
        # | attacker_addr | v | victim_guess_addr | guess victim not access |
        self.action_space = spaces.Discrete(
          len(self.attacker_address_space) + 2 + len(self.victim_address_space) + 1
        )
      else:
        # | attacker_addr | v | victim_guess_addr | 
        self.action_space = spaces.Discrete(
          len(self.attacker_address_space) + 2 + len(self.victim_address_space)
        )
    else:
      # one-hot encoding
      if self.allow_empty_victim_access == True:
        # | attacker_addr | flush_attacker_addr | v | victim_guess_addr | guess victim not access |
        self.action_space = spaces.Discrete(
          2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) + 1
        )
      else:
        # | attacker_addr | flush_attacker_addr | v | victim_guess_addr |
        self.action_space = spaces.Discrete(
          2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) 
        )
    
    # let's book keep all obvious information in the observation space 
    # since the agent is dumb
    #self.observation_space = spaces.MultiDiscrete(
    #  [
    #    3,                                          #cache latency
    #    len(self.attacker_address_space) + 1,       #attacker accessed address
    #    self.window_size + 2,                       #current steps
    #    2,                                          #whether the victim has accessed yet
    #    #2,                                          # whether it is a cflush
    #  ] * self.window_size
    #)
    self.max_box_value = max(self.window_size + 2,  2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) + 1)#max(self.window_size + 2, len(self.attacker_address_space) + 1) 
    self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, self.feature_size))
    # self.max_box_value = max(256, self.window_size + 2,  2 * len(self.attacker_address_space) + 1 + len(self.victim_address_space) + 1)#max(self.window_size + 2, len(self.attacker_address_space) + 1) 
    # self.observation_space = spaces.Box(low=-1, high=self.max_box_value, shape=(self.window_size, self.feature_size+2))

    
    #print('Initializing...')
    self.l1 = self.hierarchy['cache_1']
    self.current_step = 0
    self.victim_accessed = False
    if self.allow_empty_victim_access == True:
      #self.victim_address = random.randint(self.victim_address_max +1, self.)
      self.victim_address = random.randint(self.victim_address_min, self.victim_address_max + 1)
    else:
      self.victim_address = random.randint(self.victim_address_min, self.victim_address_max)
    self._randomize_cache()
    
    if self.configs['cache_1']["rep_policy"] == "plru_pl": # pl cache victim access always uses locked access
      assert(self.victim_address_min == self.victim_address_max) # for plru_pl cache, only one address is allowed
      self.vprint("[init] victim access %d locked cache line" % self.victim_address_max)
      self.l1.read(hex(self.ceaser_mapping(self.victim_address_max))[2:], self.current_step, replacement_policy.PL_LOCK)#, domain_id='v')

    # internal guessing buffer
    # does not change after reset
    self.guess_buffer_size = 100
    self.guess_buffer = [False] * self.guess_buffer_size
    #return

    self.last_state = None

  def clear_guess_buffer_history(self):
    self.guess_buffer = [False] * self.guess_buffer_size

  def print_sample_multiagent(self, obs, reward, done, info, opponent_agent):
    self.vprint('attacker reward:  ', reward['attacker'], 'detector reward:  ', reward['detector'], "    done:  ", done['detector'])
    self.vprint('opponent :  ', opponent_agent)

  # remap the victim address range
  def remap(self):
    if self.rerandomize_victim == False:
      self.mapping_func = lambda addr : addr
    else:
      self.vprint("doing remapping!")
      random.shuffle(self.perm)

  # ceasar remapping
  # addr is integer not string
  def ceaser_mapping(self, addr):
    if self.rerandomize_victim == False:
      return addr
    else:
      self.ceaser_access_count += 1
      # return self.mapping_func(addr)
      return self.perm[addr]

  def step(self, action):
    set_index = '-1'
    way_index = -1

    self.vprint('Step ', self.step_count)
    info = {}
    if isinstance(action, np.ndarray):
        action = action.item()

    original_action = action
    action = self.parse_action(original_action) #, self.flush_inst)

    address = hex(action[0]+self.attacker_address_min)[2:]             # attacker address in attacker_address_space
    is_guess = action[1]                                              # check whether to guess or not
    is_victim = action[2]                                             # check whether to invoke victim
    is_flush = action[3]                                              # check whether to flush
    victim_addr = hex(action[4] + self.victim_address_min)[2:]            # victim address
    is_victim_random = action[5]
    info['attacker_address'] = action[0] #TODO check wether to +self.attacker_address_min --> this returns the cache_set number
    #TODO 6/12/2023 new info needed. attacker set,way and victim
    victim_latency = None
    # if self.current_step > self.window_size : # if current_step is too long, terminate
    if self.step_count >= self.window_size - 1:
      r = 2 #
      self.vprint("length violation!")
      reward = self.length_violation_reward #-10000 
      done = True
    else:
      if is_victim == True or is_victim_random == True:
        info['invoke_victim'] = True
        info['victim_address'] = self.victim_address # temporarily record true victim address
        if self.allow_victim_multi_access == True or self.victim_accessed == False:
          r = 2 #
          self.victim_accessed = True

          if True: #self.configs['cache_1']["rep_policy"] == "plru_pl": no need to distinuish pl and normal rep_policy
            if is_victim_random == True:
                victim_random = random.randint(self.victim_address_min, self.victim_address_max)
                self.vprint("victim random access %d " % victim_random)
                t, evict_addr, [set_index, way_index] = self.l1.read(hex(self.ceaser_mapping(victim_random))[2:], self.current_step)#, domain_id='v')
                t = t.time 
                info['victim_address'] = victim_random
            elif self.victim_address <= self.victim_address_max:
                self.vprint("victim access %d " % self.victim_address)
                t, evict_addr, [set_index, way_index] = self.l1.read(hex(self.ceaser_mapping(self.victim_address))[2:], self.current_step)#, domain_id='v')
                t = t.time # do not need to lock again
            else:
                self.vprint("victim make a empty access!") # do not need to actually do something
                t = 1 # empty access will be treated as HIT??? does that make sense???
                #t = self.l1.read(str(self.victim_address), self.current_step).time 
          if t > 500:   # for LRU attack, has to force victim access being hit
            victim_latency = 1
            self.current_step += 1
            reward = self.victim_miss_reward #-5000
            if self.force_victim_hit == True:
              done = True
              self.vprint("victim access has to be hit! terminate!")
            else:
              done = False
          else:
            victim_latency = 0
            self.current_step += 1
            reward = self.victim_access_reward #-10
            done = False
        else:
          r = 2
          #self.vprint("does not allow multi victim access in this config, terminate!")
          self.current_step += 1
          reward = self.double_victim_access_reward # -10000
          done = True
      else:
        if is_guess == True:
          r = 2  #
          # this includes two scenarios
          # 1. normal scenario
          # 2. empty victim access scenario: victim_addr parsed is victim_addr_e, 
          # and self.victim_address is also victim_addr_e + 1

          if self.victim_accessed and victim_addr == hex(self.victim_address)[2:]:
              if victim_addr != hex(self.victim_address_max + 1)[2:]: 
                self.vprint("correct guess " + victim_addr)
              else:
                self.vprint("correct guess empty access!")
              # update the guess buffer 
              self.guess_buffer.append(True)
              self.guess_buffer.pop(0)
              reward = self.correct_reward # 200
              done = True
          else:
              if victim_addr != hex(self.victim_address_max + 1)[2:]:
                self.vprint("wrong guess " + victim_addr )
              else:
                self.vprint("wrong guess empty access!")

              # update the guess buffer 
              self.guess_buffer.append(False)
              self.guess_buffer.pop(0)
              reward = self.wrong_reward #-9999
              done = True
        elif is_flush == False or self.flush_inst == False:
          lat, evict_addr, [set_index, way_index] = self.l1.read(hex(self.ceaser_mapping(int('0x' + address, 16)))[2:], self.current_step)#, domain_id='a')
          lat = lat.time # measure the access latency
          if lat > 500:
            self.vprint("access " + address + " miss")
            r = 1 # cache miss
          else:
            self.vprint("access " + address + " hit"  )
            r = 0 # cache hit
          self.current_step += 1
          reward = self.step_reward #-1 
          done = False
        else:    # is_flush == True
          self.l1.cflush(address, self.current_step)#, domain_id='X')
          #cflush = 1
          self.vprint("cflush " + address )
          r = 2
          self.current_step += 1
          reward = self.step_reward
          done = False

    # lbits = self.l1.get_locked_bits()
    #return observation, reward, done, info
    if done == True and is_guess != 0:
      info["is_guess"] = True
      if reward > 0:
        info["guess_correct"] = True
      else:
        info["guess_correct"] = False
    else:
      info["is_guess"] = False
    # the observation (r.time) in this case 
    # must be consistent with the observation space
    # return observation, reward, done?, info
    #return r, reward, done, info
    current_step = self.current_step
    if self.victim_accessed == True:
      victim_accessed = 1
    else:
      victim_accessed = 0
    

    #TODO remove the temporary test
    if is_victim or is_victim_random:
        victim_accessed = 1
    else:
        victim_accessed = 0
    

    #r = cache miss or hit, 
    ####self.state = [r, action[0], current_step, victim_accessed] + self.state 
    #Xiaomeng
    # self.state = [r, victim_accessed, original_action, current_step ] + self.state  
    # self.state = self.state[0:len(self.state)-4]
    self.state.append([r, victim_accessed, original_action, self.step_count])
    self.state.popleft()
    self.step_count += 1
    
    '''
    support for multiple guess per episode
    '''
    if done == True:
      self.reset_time += 1
      if self.reset_time == self.reset_limit:  # really need to end the simulation
        self.reset_time = 0
        done = True                            # reset will be called by the agent/framework
        self.vprint('correct rate:' + str(self.calc_correct_rate()))
      else:
        done = False                           # fake reset
        self._reset()                          # manually reset

    if victim_latency is not None:
        info["victim_latency"] = victim_latency

        if self.last_state is None:
            cache_state_change = None
        else:
            cache_state_change = victim_latency ^ self.last_state
        self.last_state = victim_latency

    else:
        if r == 2:
            cache_state_change = 0
        else:
            if self.last_state is None:
                cache_state_change = None
            else:
                cache_state_change = r ^ self.last_state
            self.last_state = r
    #things to be printed in the sample_multiagent
    self.vprint('locked bits at the end of attacker step: ' + str(self.l1.get_locked_bits()))
    self.vprint('opponent reward printing from the cache guessing game side:  ', reward, "    done:  ", done)
    table = print_cache(self.l1)
    self.vprint(table)


    info["cache_state_change"] = cache_state_change

    info["way_index"] = way_index
    info["set_index"] = int(set_index,2)
    # print("$$$$$$$$$$$ address : ", address)
    # print("print to check the info and action of the attacker :  ", info)
    # print("print to check the info and action of the attacker222:  ", type(info["set_index"]), type(info["way_index"]))
    self.Data1 = np.array(list(reversed(self.state)))
    self.Data2 = reward
    self.Data3 = done
    self.Data4 = info
    return np.array(list(reversed(self.state))), reward, done, info
  
  def get_infos(self):
    return self.Data1, self.Data2, self.Data3, self.Data4

  def get_data(self):
    return self.Data1, self.Data2
  def reset(self,
            victim_address=-1,
            reset_cache_state=False,
            reset_observation=True,
            seed = -1):
    if self.ceaser_access_count > self.ceaser_remap_period:
      self.remap() # do the remap, generating a new mapping function if remap is set true
      self.ceaser_access_count = 0

    if self.cache_state_reset or reset_cache_state or seed != -1:
      self.vprint('Reset...(also the cache state)')
      self.hierarchy = build_hierarchy(self.configs, self.logger)
      self.l1 = self.hierarchy['cache_1']
      if seed == -1:
        self._randomize_cache()
      else:
        self.seed_randomization(seed)
      self.vprint("********cache after being reset********")
      table = print_cache(self.l1)
      self.vprint(table)
    else:
      self.vprint('Reset...(cache state the same)')

    self._reset(victim_address)  # fake reset

    # self.state = [0, len(self.attacker_address_space), 0, 0] * self.window_size
    # self.state = [-1, -1,-1, -1] * self.window_size
    # self.state = [0, len(self.attacker_address_space), 0, 0, 0] * self.window_size
    if reset_observation:
        self.state = deque([[-1, -1, -1, -1]] * self.window_size)
        self.step_count = 0

    self.reset_time = 0

    if self.configs['cache_1']["rep_policy"] == "plru_pl": # pl cache victim access always uses locked access
      assert(self.victim_address_min == self.victim_address_max) # for plru_pl cache, only one address is allowed
      self.vprint("[reset] victim access %d locked cache line" % self.victim_address_max)
      lat, evict_addr, _ = self.l1.read(hex(self.ceaser_mapping(self.victim_address_max))[2:], self.current_step, replacement_policy.PL_LOCK)#, domain_id='v')
  
    self.last_state = None

    return np.array(list(reversed(self.state)))

  '''
  function to calculate the correctness rate
  '''
  def calc_correct_rate(self):
    return self.guess_buffer.count(True) / len(self.guess_buffer)

  '''
  evluate the correctness of an action sequence (action+ latency) 
  action_buffer: list [(action, latency)]
  '''
  def calc_correct_seq(self, action_buffer):
    last_action, _ = action_buffer[-1]
    last_action = self.parse_action(last_action)
    print(last_action)
    guess_addr = last_action[4]
    print(guess_addr)
    self.reset(victim_addr = guess_addr)
    self.total_guess = 0
    self.correct_guess = 0
    while self.total_guess < 20:
      self.reset(victim_addr)
      for i in range(0, len(action_buffer)):
        p = action_buffer[i]
        state, _, _, _ = self.step(p[0])
        latency = state[0]
        if latency != p[1]:
          break
      if i < len(action_buffer) - 1:
        continue
      else:
        self.total_guess += 1
        if guess_addr == self.victim_address:
          self.correct_guess += 1
    return self.correct_guess / self.total_guess

  def set_victim(self, victim_address=-1):
    self.victim_address = victim_address

  # fake reset, just set a new victim addr 
  def _reset(self, victim_address=-1):
    self.current_step = 0
    self.victim_accessed = False
    if victim_address == -1:
      if self.allow_empty_victim_access == False:
        self.victim_address = random.randint(self.victim_address_min, self.victim_address_max)
      else:  # when generating random addr use self.victim_address_max + 1 to represent empty access
        self.victim_address = random.randint(self.victim_address_min, self.victim_address_max + 1) 
    else:
      assert(victim_address >= self.victim_address_min)
      if self.allow_empty_victim_access == True:
        assert(victim_address <= self.victim_address_max + 1 )
      else:
        assert(victim_address <= self.victim_address_max ) 
      
      self.victim_address = victim_address
    if self.victim_address <= self.victim_address_max:
      self.vprint("victim address ", self.victim_address)
    else:
      self.vprint("victim has empty access")
    

  def render(self, mode='human'):
    return 

  def close(self):
    return

  def seed_randomization(self, seed=-1):    
    return self._randomize_cache(mode="union", seed= seed)


  def _randomize_cache(self, mode = "union", seed=-1):
    
    # use seed so that we can get identical initialization states
    if seed != -1:
      random.seed(seed)
  # def _randomize_cache(self, mode = "attacker"):
    if mode == "attacker":
      self.l1.read(hex(self.ceaser_mapping(0))[2:], -2)#, domain_id='X')
      self.l1.read(hex(self.ceaser_mapping(1))[2:], -1)#, domain_id='X')
      return
    
    if mode == "none":
      return
    self.current_step = -self.cache_size * 2 
    for _ in range(self.cache_size * 2):
      if mode == "victim":
        addr = random.randint(self.victim_address_min, self.victim_address_max)
      elif mode == "attacker":
        addr = random.randint(self.attacker_address_min, self.attacker_address_max)
      elif mode == "union":
        addr = random.randint(self.victim_address_min, self.victim_address_max) if random.randint(0,1) == 1 else random.randint(self.attacker_address_min, self.attacker_address_max)
        # v_addr_space = self.victim_address_max - self.victim_address_min + 1
        # a_addr_space = self.attacker_address_max - self.attacker_address_min + 1
        # addr = np.random.randint(v_addr_space + a_addr_space)
        # if addr < v_addr_space:
        #     addr += self.victim_address_min
        # else:
        #     addr += self.attacker_address_min - self.victim_address_max - 1

      elif mode == "random":
        addr = random.randint(0, sys.maxsize)
      else:
        raise RuntimeError from None
      # print("checking the mode:   ", mode, "  ", addr, "  ", self.current_step)
      # print("checking the created cache:  ", self.l1.n_blocks)
      self.l1.read(hex(self.ceaser_mapping(addr))[2:], self.current_step)#, domain_id='X')
      self.current_step += 1

  def get_obs_space_dim(self):
    return int(np.prod(self.observation_space.shape))

  def get_act_space_dim(self):
    return int(np.prod(self.action_space.shape))

  def vprint(self, *args):
    if self.verbose == 1:
      print( " "+" ".join(map(str,args))+" ")

  '''
  parse the action in the degenerate space (no redundant actions)
  returns list of 5 elements representing
  address, is_guess, is_victim, is_flush, victim_addr
  '''
  def parse_action(self, action):
    address = 0
    is_guess = 0
    is_victim = 0
    is_flush = 0
    victim_addr = 0
    is_victim_random = 0
    if self.flush_inst == False:
      if action < len(self.attacker_address_space):
        address = action
      elif action == len(self.attacker_address_space):
        is_victim = 1
      elif action == len(self.attacker_address_space)+1:
        is_victim_random = 1
      else:
        is_guess = 1
        victim_addr = action - ( len(self.attacker_address_space) + 1 + 1) # becuase the one that assigned to the is_victim_random is at the attacker address space+1  
    else:
      if action < len(self.attacker_address_space):
        address = action
      elif action < 2 * len(self.attacker_address_space):
        is_flush = 1
        address = action - len(self.attacker_address_space) 
        is_flush = 1
      elif action == 2 * len(self.attacker_address_space):
        is_victim = 1
      else:
        is_guess = 1
        victim_addr = action - ( 2 * len(self.attacker_address_space) + 1 ) 
        
    return [ address, is_guess, is_victim, is_flush, victim_addr, is_victim_random ] 


def print_cache(cache):
    # Print the contents of a cache as a table
    # If the table is too long, it will print the first few sets,
    # break, and then print the last set
    table_size = 5
    ways = [""]
    sets = []
    set_indexes = sorted(cache.data.keys())
    
    if len(cache.data.keys()) > 0:
        way_no = 0

        # Label the columns
        for way in range(cache.associativity):
            ways.append("Way " + str(way_no))
            way_no += 1

        # Print either all the sets if the cache is small, or just a few
        # sets and then the last set
        sets.append(ways)
        if len(set_indexes) > table_size + 4 - 1:
            for s in range(min(table_size, len(set_indexes) - 4)):
                set_ways = cache.data[set_indexes[s]].keys()
                temp_way = ["Set " + str(s)]
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[s]][w].address)
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

            for i in range(3):
                temp_way = ['.']
                for w in range(cache.associativity):
                    temp_way.append('')
                sets.append(temp_way)

            set_ways = cache.data[set_indexes[len(set_indexes) - 1]].keys()
            temp_way = ['Set ' + str(len(set_indexes) - 1)]
            for w in range(0, cache.associativity):
                temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w][1].address)
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w].address)
            sets.append(temp_way)
            
        else:
            for s in range(len(set_indexes)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

                # add additional rows only if the replacement policy = lru_lock_policy
                if cache.rep_policy == lru_lock_policy:
                    lock_info = ["Lock bit"]

                    lock_vector_array = cache.set_rep_policy[set_indexes[s]].lock_vector_array

                    for w in range(0, len(lock_vector_array)):
                        lock_info.append(lock_vector_array[w])
                    sets.append(lock_info)

                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                    sets.append(timestamp)
                elif cache.rep_policy == new_plru_pl_policy: # add a new row to the table to show the lock bit in the plru_pl_policy cache
                    lock_info = ["Lock bit"]

                    lockarray = cache.set_rep_policy[set_indexes[s]].lockarray

                    for w in range(0, len(lockarray)):
                        if lockarray[w] == 2:
                            lock_info.append("unlocked")
                        elif lockarray[w] == 1:
                            lock_info.append("locked")
                        elif lockarray[w] == 0:
                            lock_info.append("unknown")
                        else:
                            lock_info.append(lockarray[w])
                    sets.append(lock_info)
                elif cache.rep_policy == lru_policy:  # or cache.rep_policy == lru_lock_policy:
                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                            
                    sets.append(timestamp)
                    # print(timestamp)

        # table = UnixTable(sets)
        # table.title = cache.name
        # table.inner_row_border = True
        # table_lines = table.table
        max_col_widths = [max(len(item) for item in col) for col in zip(*sets)]
        # Construct the formatted table string
        table_string = ""
        for row in sets:
          formatted_row = [item.rjust(width) for item, width in zip(row, max_col_widths)]
          table_string += "|" + "|".join(formatted_row) + "|\n"

        # Add a separator line after the header
        table_string = table_string + "-" * (sum(max_col_widths) + len(max_col_widths) + 1) + "\n"
        return table_string





if __name__ == '__main__':
    env = CacheGuessingGameEnv()
    obs = env.reset()
    done = False
    i=0
    print(env.l1.rep_policy)
    while not done:
        i+=1
        obs, reward, done, info = env.step(np.random.randint(9))
        # print("step ", i, ":", obs, reward, done, info) 
