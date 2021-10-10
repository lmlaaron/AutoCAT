# bootstrap naive RL runs

import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("gym_cache:cache-v0")
#model = PPO(MlpPolicy, env, verbose=0)
#model.learn(total_timesteps=25000)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() #my agent
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
