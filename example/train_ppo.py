from __future__ import annotations


import copy
import time
import hydra
import logging

import torch
import torch.multiprocessing as mp
import torch.nn as nn

import rloptim.envs.atari_wrappers as atari_wrappers
import rloptim.envs.gym_wrappers as gym_wrappers
#sys.path.append("../src")

from rloptim.agents.ppo.ppo_agent import PPOAgent, RemotePPOAgent
from rloptim.core.replay_buffer import create_replay_buffer
from rloptim.core.model_server import ModelServer
from rloptim.core.rollouts import ParallelRollouts
from rloptim.core.trainer import AsyncTrainer
from rloptim.core.evaluator import Evaluator
from rloptim.envs.env import Env

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
src_dir= os.path.join(parent_dir,"src")
sys.path.insert(0, src_dir)

src_dir= os.path.join(parent_dir,"src/gym_cache")
sys.path.insert(0, src_dir)

#sys.path.append("../src/gym_cache")
#sys.path.append("../src")
#from envs.simple_cache_wrapper import SimpleCacheWrapperFactory
from envs.cache_simulator_wrapper_factory import CacheSimulatorWrapperFactory

from models.ppo_model import PPOModel

@hydra.main(config_path="./conf", config_name="conf_ppo")
def main(cfg):
    logging.info(f"configs:\n{cfg}")
    #model_address = "localhost:2020"
    #train_address = "localhost:2021"
    #eval_address = "localhost:2022"

    train_device = "cuda:0"
    infer_device = "cuda:1"

    num_rollouts = 8
    num_workers = 32

    replay_buffer_size = 65536 #or bigger 

    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    entropy_ratio = 0.01
    

    epochs = 1000
    steps_per_epoch = 200
    batch_size = 2048
    lr = 3e-4

    eval_episodes = 100

    logging.info(f"num_rollouts = {num_rollouts}; num_workers = {num_workers}")
    logging.info(f"replay_buffer_size = {replay_buffer_size}")
    logging.info(f"epochs = {epochs}; steps_per_epoch = {steps_per_epoch}; batch_size = {batch_size}; lr = {lr}; eval_episodes = {eval_episodes}")

    max_steps = 20

    logging.info(f"Agent Model: input_dim = {input_dim}; hidden_dim = {hidden_dim}; action_dim = {action_dim}")

    penalty_for_step = -0.01
    reward_correct_guess = 1
    reward_wrong_guess = -10

    env_factory = CacheSimulatorWrapperFactory(cfg)

    input_dim = env_factory.get_obs_dim() #448
    hidden_dim = 512
    action_dim = env_factory.get_action_dim() #64

    logging.info(f"penalty_for_step = {penalty_for_step}; reward_correct_guess = {reward_correct_guess}; reward_wrong_guess = {reward_wrong_guess}")
    # env = atari_wrappers.make_atari("PongNoFrameskip-v4")
    # env = gym_wrappers.wrap_atari(env, max_episode_steps=2700)

    net = PPOModel(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   action_dim=action_dim)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_agent = PPOAgent(net,
                           optimizer=optim,
                           gamma=gamma,
                           gae_lambda=gae_lambda,
                           eps_clip=eps_clip,
                           entropy_ratio=entropy_ratio)

    train_agent.model.to(train_device)

    infer_agent = copy.deepcopy(train_agent)
    infer_agent.model.to(infer_device)

    remote_agent1 = RemotePPOAgent(train_agent, cfg.model_address)
    remote_agent2 = RemotePPOAgent(train_agent, cfg.model_address)

    remote_agent1.train()
    remote_agent2.eval()

    replay_buffer = create_replay_buffer(replay_buffer_size,
                                         device=train_device,
                                         prefetch=3)

    model_server = ModelServer(cfg.model_address, infer_agent)

    trainer = AsyncTrainer(cfg.train_address,
                           cfg.model_address,
                           train_agent,
                           replay_buffer,
                           batch_size=batch_size,
                           sync_every_n_steps=10)
    evaluator = Evaluator(cfg.eval_address)

    rollouts1 = ParallelRollouts(cfg.train_address,
                                 cfg.model_address,
                                 env_factory,
                                 remote_agent1,
                                 num_rollouts=num_rollouts,
                                 num_workers=num_workers,
                                 num_episodes=None,
                                 seed=123,
                                 connect_deadline=120)

    rollouts2 = ParallelRollouts(cfg.eval_address,
                                 cfg.model_address,
                                 env_factory_verbose,
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

    for epoch in range(epochs):
        trainer.run(epochs=1, steps_per_epoch=steps_per_epoch)
        time.sleep(1)
        evaluator.run(episodes=eval_episodes)
        time.sleep(1)
        if epoch % 20 == 0:
            train_agent.save(f"model-{epoch}.pt")
            print(f"Save agent Epochs = {epoch}")

    rollouts1.terminate()
    rollouts2.terminate()

    trainer.stop_server()
    evaluator.stop_server()
    model_server.stop()


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    # mp.set_start_method("spawn")
    main()