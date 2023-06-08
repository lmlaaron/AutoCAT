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
    CatFrames, EnvCreator, ParallelEnv, RewardSum, StepCounter, ObservationNorm
from torchrl.envs import set_exploration_type, ExplorationType

import model_utils

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.record.loggers.wandb import WandbLogger

@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):

    print(f"workding_dir = {os.getcwd()}")

    logger = WandbLogger(exp_name="-".join([cfg.logger.exp_name, cfg.logger.exp_suffix]))

    frames_per_batch = cfg.collector.frames_per_batch
    total_frames = cfg.collector.total_frames
    num_epochs = cfg.num_epochs
    eval_freq = cfg.eval_freq
    device = cfg.device
    env_config = cfg.env_config
    env_config = OmegaConf.to_container(env_config)
    num_workers = cfg.collector.num_workers

    def make_env():
        return TransformedEnv(
            GymWrapper(CacheGuessingGameEnv(env_config), device=device),
            Compose(
                RewardSum(),
                StepCounter(),
            )
        )

    env = make_env()

    dummy_env = make_env()

    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env.action_spec.space.n).to(device)


    replay_buffer_size = cfg.rb.size
    prefetch = cfg.rb.prefetch
    batch_size = cfg.rb.batch_size
    if replay_buffer_size is None:
        replay_buffer_size = frames_per_batch
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(replay_buffer_size),
        sampler=SamplerWithoutReplacement(),
        batch_size=batch_size,
        prefetch=prefetch)

    actor = train_model.get_actor()

    # get a rollout
    rollout = ParallelEnv(2, make_env).rollout(10, actor)
    print("rollout", rollout)

    value_net = train_model.get_value()
    value_head = train_model.get_value_head()
    loss_fn = ClipPPOLoss(
        actor,
        value_head,
        entropy_coef=cfg.entropy_coeff,
    )
    optimizer = torch.optim.Adam(loss_fn.parameters(), **cfg.optimizer)
    gae = GAE(value_network=value_net, gamma=0.99, lmbda=0.95)
    datacollector = torchrl.collectors.MultiSyncDataCollector(
        [EnvCreator(make_env)] * num_workers,
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
    ep_reward = []
    for k, data in enumerate(datacollector):

        frames += data.numel()

        episode_reward = data.get(("next", "episode_reward"))[data.get(("next", "done"))]
        if episode_reward.numel():
            ep_reward.append(episode_reward.mean())

        if k % eval_freq == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                tdout = env.rollout(1000, actor, break_when_any_done=False)
                test_rewards.append(tdout.get(('next', 'reward')).mean())
                done = tdout['next', 'done']
                sc = tdout['next', 'step_count'][done].float().mean()
                er = tdout['next', 'episode_reward'][done].mean()
                logger.log_scalar(
                    "test reward",
                    test_rewards[-1],
                    step=frames,
                    )
                logger.log_scalar(
                    "test_traj_len",
                    sc,
                    step=frames,
                )
                logger.log_scalar(
                    "test_episode_reward",
                    er,
                    step=frames,
                )
            del tdout

        td_log = TensorDict({'grad norm': torch.zeros(num_epochs, num_batches)}, batch_size=[num_epochs, num_batches])

        for i in range(num_epochs):
            # we can safely flatten the data, GAE supports that
            rb.empty()
            with torch.no_grad():
                data_gae = gae(
                    data
                ).to("cpu", non_blocking=True)
            rb.extend(data_gae.reshape(-1))
            if len(rb) != data.numel():
                raise RuntimeError("rb size does not match the data size.")
            for j, batch in enumerate(rb):
                if j >= num_batches:
                    raise RuntimeError('too many batches')
                batch = batch.to(device)
                pbar.update(1)
                loss_vals = loss_fn(batch)
                loss_val = sum(loss_vals.values())
                loss_val.backward()
                pbar.set_description(
                    f"collection {k}, epoch {i}, batch {j}, "
                    f"reward: {data['next', 'reward'].mean(): 4.4f}, "
                    f"loss critic: {loss_vals['loss_critic'].item(): 4.4f}, "
                    f"test reward: {test_rewards[-1]: 4.4f}, "
                    f"test ep reward: {er.item()}"
                )
                td_log[i, j] = loss_vals.detach()
                td_log['grad norm'][i, j] = torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
        datacollector.update_policy_weights_()
        logger.log_scalar("frames", frames, step=frames)
        if ep_reward:
            logger.log_scalar("episode reward", ep_reward[-1], step=frames)
        logger.log_scalar("train_reward", data.get(('next', 'reward')).mean(), step=frames)
        for key, val in td_log.items():
            logger.log_scalar(key, val.mean(), step=frames)

        # testdata = env.rollout(actor)

if __name__ == "__main__":
    main()
