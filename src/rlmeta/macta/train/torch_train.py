import os
import copy
import hydra
import sys
import torch
import tqdm
from omegaconf import OmegaConf
from tensordict import TensorDict

from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType

import torchrl.collectors


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
# from env.cache_guessing_game_env import CacheGuessingGameEnv
# from env.env_pettingzoo import CacheAttackerDefenderEnv
from env.macta_pettingzoo import CacheAttackerDetectorEnv
# from env.cache_attacker_detector_env import CacheAttackerDetectorEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import Compose, TransformedEnv, \
    EnvCreator, ParallelEnv, RewardSum, StepCounter
from torchrl.envs import set_exploration_type, ExplorationType
from torchrl.envs.utils import check_env_specs

import model_utils

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchrl.record.loggers.wandb import WandbLogger
from pettingzoo.utils import agent_selector, wrappers

HERE = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path="../config",
    config_name="new_macta"
)
def main(cfg):
    print(f"workding_dir = {os.getcwd()}")

    # ========= Logger ========= #
    # logger = WandbLogger(
    #     exp_name="-".join(["macta", "exp0"]),
    #     config=cfg
    # )

    # ========= Save config ========= #
    # save the config
    # os.makedirs(f"{HERE}/saved_{logger.exp_name}", exist_ok=True)
    # torch.save(cfg, f"{HERE}/saved_{logger.exp_name}/cfg.pt")

    # ========= Extract config params (for efficiency) ========= #
    # frames_per_batch = 262144
    # total_frames = cfg.collector.total_frames
    # num_epochs = cfg.num_epochs
    # eval_freq = cfg.eval_freq
    # device = cfg.device
    # env_config = cfg.env_config
    # env_config = OmegaConf.to_container(env_config)
    # num_workers = cfg.collector.num_workers
    # envs_per_collector = cfg.collector.envs_per_collector
    # preemptive_threshold = cfg.collector.preemptive_threshold
    # collector_device = cfg.collector.device
    # clip_grad_norm = cfg.loss.clip_grad_norm
    # save_freq = cfg.logger.save_frequency
    prefetch = cfg.prefetch
    batch_size = cfg.batch_size
    # replay_buffer_size = cfg.replay_buffer_size

    # ========= Env factory ========= #
    # We don't want to serialize the env, a constructor is sufficient.
    def make_env():
        env = CacheAttackerDetectorEnv(cfg.env_config)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        env = PettingZooWrapper(
            # env=CacheAttackerDetectorEnv(cfg.env_config),
            env=env,
            use_mask=True, # Must use it since one player plays at a time
            group_map=None # # Use default for AEC (one group per player)
            )
        return env = lambda: TransformedEnv(env, transform.clone())
        # return TransformedEnv(
        #     env, 
        #     RewardSum(in_keys=[env.reward_key[1]], out_keys=[("opponent", "reward")]),
        #     # StepCounter(),
        # )

    # def test_make_env():
    #     return TransformedEnv(
    #         GymWrapper(CacheGuessingGameEnv(cfg.env_config), device=cfg.train_device),
    #         Compose(
    #             RewardSum(),
    #             StepCounter(),
    #         )
    #     )

    env = make_env()
    # print("reward_keys:", type(env.reward_keys))
    # print(env.reset()["episode_reward"])
    # print(env.group_map)
    # print(env.rollout(5))
    # print("observation_spec:", env.observation_spec)
    check_env_specs(env)
    print("++++++++++++++env made")
    dummy_env = make_env()
    dummy_env_d = make_env()


    # ========= Construct model ========= # for attacker
    # attacker
    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env.action_spec['opponent']['action'].space.n#output dim
    ).to(cfg.train_device)
    # detector
    train_model_d = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        dummy_env_d.action_spec['detector']['action'].space.n #output dim
    ).to(cfg.train_device)
    # ========= Replay buffer ========= #
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(262144, device=cfg.train_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=batch_size,
        prefetch=prefetch,
    )
    rb_d = TensorDictReplayBuffer(
        storage=LazyTensorStorage(262144, device=cfg.train_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=batch_size,
        prefetch=prefetch,
    )
    # # ========= Isolate model components ========= #
    actor = train_model.get_actor()
    value_net = train_model.get_value()
    value_head = train_model.get_value_head()

    actor_d = train_model_d.get_actor()
    value_net_d = train_model_d.get_value()
    value_head_d = train_model_d.get_value_head()

    # # ========= Loss module ========= #
    loss_fn = ClipPPOLoss(
        actor,
        value_head,
        entropy_coef=0.02,
        loss_critic_type="smooth_l1",
    )
    gae = GAE(
        value_network=value_net,
        gamma=0.99,
        lmbda=0.95,
        average_gae=True,
        shifted=True
    )

    loss_fn_d = ClipPPOLoss(
        actor_d,
        value_head_d,
        entropy_coef=0.02,
        loss_critic_type="smooth_l1",
    )
    gae = GAE(
        value_network=value_net_d,
        gamma=0.99,
        lmbda=0.95,
        average_gae=True,
        shifted=True
    )

    # # ========= Optimizer ========= #
    optimizer = torch.optim.Adam(loss_fn.parameters(), **cfg.optimizer) #ToDo add optimizer to the cfg
    optimizer_d = torch.optim.Adam(loss_fn_d.parameters(), **cfg.optimizer)

    # ========= Data collector ========= #
    # We customize the data collection depending on the resources available
    num_workers = 1
    frames_per_batch = 8192
    total_frames = 1000000
    # if num_workers > 1 and envs_per_collector:
    #     datacollector = torchrl.collectors.MultiSyncDataCollector(
    #         (num_workers // envs_per_collector) * [
    #             ParallelEnv(envs_per_collector, EnvCreator(make_env))],
    #         policy=actor.eval(),
    #         frames_per_batch=frames_per_batch,
    #         total_frames=total_frames,
    #         device=collector_device,
    #         preemptive_threshold=preemptive_threshold,
    #     )
    # elif num_workers > 1:
    if num_workers > 1:
        datacollector = torchrl.collectors.SyncDataCollector(
            ParallelEnv(num_workers, EnvCreator(make_env)),
            policy=actor.eval(),
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device="cuda",
        )
    else:
        datacollector = torchrl.collectors.SyncDataCollector(
            make_env(),
            policy=actor.eval(),
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device="cuda",
        )

    # print("!!!!!!!!!!", datacollector)
    num_batches = -(frames_per_batch // -batch_size)
    pbar = tqdm.tqdm(total=total_frames)
    frames = 0
    test_rewards = []
    ep_reward = []
'''
    # ========= Training loop ========= #
    for k, data in enumerate(datacollector):
        frames += data.numel()
        pbar.update(data.numel())
        # make sure that the data has the right shape: we reshape the batch
        # dimension to time.
        # We just need to make sure that trajectories are marked as finished
        # when truncated
        data['next', 'done'][..., -1, :] = True
        data = data.reshape(-1)  # [time x others]

        episode_reward = data.get(("next", "episode_reward"))[
            data.get(("next", "done"))]
        if episode_reward.numel():
            ep_reward.append(episode_reward.mean())

        # ========= Evaluation ========= #
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

        td_log = TensorDict(
            {'grad norm': torch.zeros(num_epochs, num_batches)},
            batch_size=[num_epochs, num_batches]
        )

        # ========= Training sub-loop ========= #
        actor.train()
        for i in range(num_epochs):
            rb.empty()
            with torch.no_grad():
                data_gae = gae(
                    data.to(device, non_blocking=True)
                )
            rb.extend(data_gae)
            if len(rb) != data.numel():
                raise RuntimeError("rb size does not match the data size.")
            for j, batch in enumerate(rb):
                if j >= num_batches:
                    raise RuntimeError('too many batches')
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
                td_log['grad norm'][i, j] = torch.nn.utils.clip_grad_norm_(
                    loss_fn.parameters(),
                    clip_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        datacollector.update_policy_weights_()
        actor.eval()

        # ========= Logging ========= #
        logger.log_scalar("frames", frames, step=frames)
        if ep_reward:
            logger.log_scalar("episode reward", ep_reward[-1], step=frames)
        logger.log_scalar(
            "train_reward",
            data.get(('next', 'reward')).mean(),
            step=frames
        )
        for key, val in td_log.items():
            logger.log_scalar(key, val.mean(), step=frames)

        # ========= Saving ========= #
        if k % save_freq == 0:
            # save parameters as a memory-mapped array
            td = TensorDict.from_module(actor)
            for param in actor.parameters():
                param.requires_grad_(False)
            td.memmap_(f"{HERE}/saved_{logger.exp_name}/")
            for param in actor.parameters():
                param.requires_grad_(True)
'''

if __name__ == "__main__":
    main()

