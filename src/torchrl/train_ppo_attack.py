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
from tensordict import TensorDict

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

from torchrl.record.loggers.wandb import WandbLogger

@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    if cfg.seed is not None:
        random_utils.manual_seed(cfg.seed)

    print(f"workding_dir = {os.getcwd()}")

    def make_env():
        return ParallelEnv(cfg.collector.num_workers, EnvCreator(lambda: GymWrapper(CacheGuessingGameEnv(OmegaConf.to_container(cfg.env_config)))))

    logger = WandbLogger(exp_name='rl4cache')

    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    num_epochs = cfg.num_epochs
    batch_size = cfg.batch_size

    env = make_env()

    dummy_env = GymWrapper(CacheGuessingGameEnv(OmegaConf.to_container(cfg.env_config)))

    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env.action_spec.space.n).to(cfg.train_device)

    optimizer = torch.optim.Adam(train_model.parameters(), **cfg.optimizer)

    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg.replay_buffer_size), sampler=SamplerWithoutReplacement(), batch_size=batch_size)

    actor = train_model.get_actor()

    value_net = train_model.get_value()
    value_head = train_model.get_value_head()
    loss_fn = ClipPPOLoss(
        actor,
        value_head,
    )
    gae = GAE(value_network=value_net, gamma=0.99, lmbda=0.95)
    datacollector = torchrl.collectors.SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    total_batches = total_frames // frames_per_batch
    num_batches = -(frames_per_batch // -batch_size)
    total_updates = total_batches * num_epochs * num_batches
    pbar = tqdm.tqdm(total=total_updates)
    frames = 0
    for data in datacollector:
        frames += data.numel()
        pbar.set_description(f"reward: {data['next', 'reward'].mean(): 4.4f}")

        td_log = TensorDict({}, batch_size=[num_epochs, num_batches])

        for i in range(num_epochs):
            # we can safely flatten the data, GAE supports that
            data = gae(data.view(-1))
            rb.extend(data.view(-1))
            for j, batch in enumerate(rb):
                pbar.update(1)
                loss_vals = loss_fn(batch)
                for key, lv in loss_vals.items():
                    td_log[i, j][key] = lv.mean().detach()
                loss_val = sum(loss_vals.values())
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
        datacollector.update_policy_weights_()
        logger.log_scalar("frames", frames)
        for key, val in td_log.items():
            logger.log_scalar(key, val.mean())
        # testdata = env.rollout(actor)

if __name__ == "__main__":
    main()
