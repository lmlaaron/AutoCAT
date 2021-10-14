# test lstm for guessing the period
import gym
from stable_baselines import PPO2

model = PPO2('MlpLstmPolicy', "gym_cache:test-lstm-v0", nminibatches=1, verbose=1, policy_kwargs={"n_lstm":50})
model.learn(400000)

env = model.get_env()
observation = env.reset()
total_reward = 0
num_correct = 0
num_wrong = 0
for i in range(2000):
    print(i)
    action, _state = model.predict(observation, deterministic = False)
    observation, reward, done, info = env.step(action)
    if reward > 0:
        num_correct += 1
    if reward < 0:
        num_wrong += 1
    total_reward += reward

print(num_correct)
print(num_wrong)
print(1.0 * num_correct / (num_correct + num_wrong))
print(total_reward)