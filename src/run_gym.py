# bootstrap naive RL runs

import gym
env = gym.make("gym_cache:cache-v0")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() #my agent
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
