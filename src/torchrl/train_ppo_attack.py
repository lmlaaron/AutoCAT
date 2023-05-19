import copy
import logging
import os
import time

import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
import sys

import torch
import torch.multiprocessing as mp

import torchrl.collectors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from cache_guessing_game_env_impl import CacheGuessingGameEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import UnsqueezeTransform, Compose, TransformedEnv, \
    CatFrames, EnvCreator, ParallelEnv

import model_utils

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    if cfg.seed is not None:
        random_utils.manual_seed(cfg.seed)

    print(f"workding_dir = {os.getcwd()}")

    def make_env():
        return ParallelEnv(cfg.collector.num_workers, EnvCreator(lambda: GymWrapper(CacheGuessingGameEnv(OmegaConf.to_container(cfg.env_config)))))

    env = make_env()

    dummy_env = GymWrapper(CacheGuessingGameEnv(OmegaConf.to_container(cfg.env_config)))

    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env.action_spec.space.n).to(cfg.train_device)

    optimizer = torch.optim.Adam(train_model.parameters(), **cfg.optimizer)

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.replay_buffer_size), sampler=SamplerWithoutReplacement(), batch_size=cfg.batch_size)

    actor = train_model.get_actor()

    value_net = train_model.get_value()
    value_head = train_model.get_value_head()
    loss_fn = ClipPPOLoss(
        actor,
        value_head,
    )
    gae = GAE(value_network=value_net, gamma=0.99, lmbda=0.95)
    dataloader = torchrl.collectors.SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )
    pbar = tqdm.tqdm(dataloader, total=cfg.num_epochs)
    for data in pbar:
        pbar.set_description(f"reward: {data['next', 'reward'].mean(): 4.4f}")
        for i in range(cfg.num_epochs):
            # we can safely flatten the data, GAE supports that
            data = gae(data.view(-1))
            rb.extend(data.view(-1))
            for batch in rb:
                loss_vals = loss_fn(batch)
                loss_val = sum(loss_vals.values())
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
        dataloader.update_policy_weights_()
        # testdata = env.rollout(actor)

if __name__ == "__main__":
    main()
