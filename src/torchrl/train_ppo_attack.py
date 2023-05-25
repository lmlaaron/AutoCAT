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
from torchrl.envs import set_exploration_type, ExplorationType

import model_utils

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.record.loggers.wandb import WandbLogger

@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    if cfg.seed is not None:
        random_utils.manual_seed(cfg.seed)

    print(f"workding_dir = {os.getcwd()}")

    logger = WandbLogger(exp_name='rl4cache')

    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    num_epochs = cfg.num_epochs
    eval_freq = cfg.eval_freq
    device = cfg.device
    env_config = cfg.env_config
    env_config = OmegaConf.to_container(env_config)

    def make_env():
        return GymWrapper(CacheGuessingGameEnv(env_config), device=device)

    env = make_env()

    dummy_env = make_env()

    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env.action_spec.space.n).to(device)

    optimizer = torch.optim.Adam(train_model.parameters(), **cfg.optimizer)

    replay_buffer_size = cfg.rb.size
    prefetch = cfg.rb.prefetch
    batch_size = cfg.rb.batch_size
    if replay_buffer_size is None:
        replay_buffer_size = frames_per_batch
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_buffer_size), sampler=SamplerWithoutReplacement(), batch_size=batch_size, prefetch=prefetch)

    actor = train_model.get_actor()

    value_net = train_model.get_value()
    value_head = train_model.get_value_head()
    loss_fn = ClipPPOLoss(
        actor,
        value_head,
    )
    gae = GAE(value_network=value_net, gamma=0.99, lmbda=0.95)
    datacollector = torchrl.collectors.MultiSyncDataCollector(
        [EnvCreator(make_env)] * cfg.collector.num_workers,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )
    total_batches = total_frames // frames_per_batch
    num_batches = -(frames_per_batch // -batch_size)
    total_updates = total_batches * num_epochs * num_batches
    pbar = tqdm.tqdm(total=total_updates)
    frames = 0
    test_rewards = []
    for k, data in enumerate(datacollector):
        if k % eval_freq == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                tdout = env.rollout(1000, actor)
                test_rewards.append(tdout.get(('next', 'reward')).mean())
                logger.log_scalar(
                    "test_reward",
                    test_rewards[-1]
                    )
                logger.log_scalar(
                    "test traj len",
                    tdout.numel(),
                )
            del tdout

        frames += data.numel()

        td_log = TensorDict({}, batch_size=[num_epochs, num_batches])

        for i in range(num_epochs):
            # we can safely flatten the data, GAE supports that
            data = gae(data.view(-1))
            rb.extend(data.view(-1))
            for j, batch in enumerate(rb):
                batch = batch.to(device)
                pbar.update(1)
                loss_vals = loss_fn(batch)
                for key, lv in loss_vals.items():
                    td_log[i, j][key] = lv.mean().detach()
                loss_val = sum(loss_vals.values())
                loss_val.backward()
                pbar.set_description(
                    f"collection {k}, epoch {i}, batch {j}, "
                    f"reward: {data['next', 'reward'].mean(): 4.4f}, "
                    f"loss critic: {loss_vals['loss_critic'].item(): 4.4f}, "
                    f"test reward: {test_rewards[-1]: 4.4f}"
                )
                optimizer.step()
                optimizer.zero_grad()
        datacollector.update_policy_weights_()
        logger.log_scalar("frames", frames)
        logger.log_scalar("train_reward", data.get(('next', 'reward')).mean())
        for key, val in td_log.items():
            logger.log_scalar(key, val.mean())

        # testdata = env.rollout(actor)

if __name__ == "__main__":
    main()
