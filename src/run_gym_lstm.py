# bootstrap naive RL runs
import gym
from stable_baselines import PPO2
#env = gym.make("gym_cache:cache-guessing-game-v0")
model = PPO2('MlpLstmPolicy', "gym_cache:cache-guessing-game-v0", nminibatches=1, verbose=1)
model.learn(10000)

env = model.get_env()
obs = env.reset()
total_reward = 0
num_correct =0
num_wrong = 0
for i in range(10000):
    print(i)
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs)
    obs, reward , done, _ = env.step(action)
    # Note: with VecEnv, env.reset() is automatically called
    if reward > 10 and action[0][1] == True:
        num_correct += 1
    if reward == 0 and action[0][1] == True: #is guess and wrong
        num_wrong += 1
    total_reward += reward
    print(action)
    print(reward)
    if done:
        observation = env.reset()

print(num_correct)
print(num_wrong)
print(1.0 * num_correct / (num_correct + num_wrong))