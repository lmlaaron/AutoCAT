from __future__ import annotations

import copy
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn

import rloptim.envs.atari_wrappers as atari_wrappers
import rloptim.envs.gym_wrappers as gym_wrappers

from rloptim.agents.dqn.dqn_agent import DQNAgent, RemoteDQNAgent
from rloptim.core.replay_buffer import create_prioritized_replay_buffer
from rloptim.core.model_server import ModelServer
from rloptim.core.rollouts import ParallelRollouts
from rloptim.core.trainer import AsyncTrainer
from rloptim.core.evaluator import Evaluator
from rloptim.envs.env import Env

import sys

sys.path.append("../src/gym_cache")
sys.path.append("../src")
#from envs.simple_cache_wrapper import SimpleCacheWrapperFactory
from envs.cache_simulator_wrapper_factory import CacheSimulatorWrapperFactory
from models.dqn_model import DQNModel


def main():
    model_address = "localhost:2020"
    train_address = "localhost:2021"
    eval_address = "localhost:2022"

    train_device = "cuda:0"
    infer_device = "cuda:1"

    num_rollouts = 8
    num_workers = 32

    replay_buffer_size = 1024
    alpha = 0.6
    beta = 0.4

    epochs = 1000
    steps_per_epoch = 2000
    batch_size = 512
    lr = 3e-4

    eval_episodes = 100

    max_steps = 20

    #env_factory = SimpleCacheWrapperFactory(max_steps=max_steps)
    env_factory = CacheSimulatorWrapperFactory()

    # env = atari_wrappers.make_atari("PongNoFrameskip-v4")
    # env = gym_wrappers.wrap_atari(env, max_episode_steps=2700)

    net = DQNModel(
        input_dim = 448,
        hidden_dim = 512,
        action_dim = 64
    )
    
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_agent = DQNAgent(net,
                            optimizer=optim,
                            sync_every_n_steps=100,
                            multi_step=3,
                            num_agents=num_rollouts)

    train_agent.online_net.to(train_device)
    train_agent.target_net.to(train_device)

    infer_agent = copy.deepcopy(train_agent)
    infer_agent.online_net.to(infer_device)
    infer_agent.target_net.to(infer_device)

    remote_agent1 = RemoteDQNAgent(train_agent, model_address)
    remote_agent2 = RemoteDQNAgent(train_agent, model_address)

    remote_agent1.train()
    remote_agent2.eval()

    replay_buffer = create_prioritized_replay_buffer(replay_buffer_size,
                                                     alpha,
                                                     beta,
                                                     device=train_device,
                                                     prefetch=3)

    model_server = ModelServer(model_address, infer_agent)

    trainer = AsyncTrainer(train_address,
                           model_address,
                           train_agent,
                           replay_buffer,
                           batch_size=batch_size,
                           sync_every_n_steps=10)
    evaluator = Evaluator(eval_address)

    rollouts1 = ParallelRollouts(train_address,
                                 model_address,
                                 env_factory,
                                 remote_agent1,
                                 num_rollouts=num_rollouts,
                                 num_workers=num_workers,
                                 num_episodes=None,
                                 seed=123,
                                 connect_deadline=120)

    rollouts2 = ParallelRollouts(eval_address,
                                 model_address,
                                 env_factory,
                                 remote_agent2,
                                 num_rollouts=num_rollouts,
                                 num_workers=num_workers,
                                 num_episodes=None,
                                 seed=456,
                                 connect_deadline=120)

    model_server.start()
    trainer.start_server()
    evaluator.start_server()

    rollouts1.start()
    rollouts2.start()

    for _ in range(epochs):
        trainer.run(epochs=1, steps_per_epoch=steps_per_epoch)
        time.sleep(1)
        evaluator.run(episodes=eval_episodes)
        time.sleep(1)

    rollouts1.terminate()
    rollouts2.terminate()

    trainer.stop_server()
    evaluator.stop_server()
    model_server.stop()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    # mp.set_start_method("spawn")
    main()
