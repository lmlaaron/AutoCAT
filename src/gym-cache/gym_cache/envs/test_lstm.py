from gym import error, spaces, utils
from gym.utils import seeding
import numpy
import numpy as np
import sys
import math
import random
import gym

class TestLstmEnv(gym.Env):
    """
    Desciprtion:
    test the LSTM's capability to recognize 1D pattern
    Observation:
    # just 1D discrete space 
    Actions:
    # whether to bet
    Reward:
    # bet correct: 50
    # bet wrong: 0
    """

    def __init__(self):
        self.state = 0
        self.observation_space = spaces.Discrete(2) #
        self.action_space = spaces.Discrete(2) # 0 not guess, 1 guess
        return
    def step(self, action):
        if action.ndim > 1:  # workaround for training and predict discrepency
            action = action[0]
        
        if self.state < 10:
            self.state += 1
            out = 0
        else:
            self.state = 0
            out = 1

        if action == 1:
            if out == 1:
                reward = 9
            else:
                reward = -9
        else:
            reward = -1 

        self.state += 1
        return np.array([out]), reward, False, {}
    def reset(self):
        self.state = 0
        return np.array([0])
    def render(self, mode='human'):
        return 
    def close(self):
        return
