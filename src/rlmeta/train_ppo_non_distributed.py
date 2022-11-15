import copy
import logging
import os
import sys
import time

from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from rich.console import Console
from rich.progress import track

import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.controller import Controller
from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.samplers import UniformSampler
from rlmeta.storage import TensorCircularBuffer
from rlmeta.utils.optimizer_utils import get_optimizer
from rlmeta.utils.stats_dict import StatsDict

import model_utils

from cache_env_wrapper import CacheEnvWrapperFactory
from metric_callbacks import MetricCallbacks


class LoopRunner:
    def __init__(self,
                 env: Env,
                 agent: Agent,
                 replay_buffer: Optional[ReplayBuffer] = None,
                 should_update: bool = True,
                 seed: Optional[int] = None,
                 episode_callbacks: Optional[EpisodeCallbacks] = None) -> None:
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._should_update = should_update

        self._seed = seed
        if seed is not None:
            self._env.reset(seed=seed)

        self._episode_callbacks = episode_callbacks

    def run(self,
            num_episodes: Optional[int] = None,
            num_steps: Optional[int] = None) -> StatsDict:
        assert num_episodes is not None or num_steps is not None

        if self._replay_buffer is not None:
            self._replay_buffer.clear()

        stats = StatsDict()
        if num_episodes is not None:
            for _ in range(num_episodes):
                metrics = self._run_loop()
                stats.extend(metrics)
        else:
            while len(self._replay_buffer) < num_steps:
                metrics = self._run_loop()
                stats.extend(metrics)

        return stats

    def _batch_obs(self, timestep: TimeStep) -> TimeStep:
        obs, reward, terminated, truncated, info = timestep
        return TimeStep(obs.unsqueeze(0), reward, truncated, terminated, info)

    def _unbatch_action(self, action: Action) -> Action:
        act, info = action
        act.squeeze_(0).cpu()
        info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
        return Action(act, info)

    def _run_loop(self) -> None:
        episode_length = 0
        episode_return = 0.0

        if self._episode_callbacks is not None:
            self._episode_callbacks.reset()
            self._episode_callbacks.on_episode_start(index=0)

        timestep = self._env.reset()
        self._agent.observe_init(timestep)
        if self._episode_callbacks is not None:
            self._episode_callbacks.on_episode_init(index=0, timestep=timestep)

        while not (timestep.terminated or timestep.truncated):
            # Model server requires a batch_dim, so unsqueeze here for local
            # runs.
            timestep = self._batch_obs(timestep)
            action = self._agent.act(timestep)
            # Unbatch the action.
            action = self._unbatch_action(action)

            timestep = self._env.step(action)
            self._agent.observe(action, timestep)
            if self._should_update:
                self._agent.update()

            episode_length += 1
            episode_return += timestep.reward
            if self._episode_callbacks is not None:
                self._episode_callbacks.on_episode_step(index=0,
                                                        step=episode_length -
                                                        1,
                                                        action=action,
                                                        timestep=timestep)
        metrics = {
            "episode_length": episode_length,
            "episode_return": episode_return,
        }
        if self._episode_callbacks is not None:
            self._episode_callbacks.on_episode_end(index=0)
            metrics.update(self._episode_callbacks.custom_metrics)

        return metrics


@hydra.main(config_path="./config", config_name="ppo_attack")
def main(cfg):
    print(f"workding_dir = {os.getcwd()}")
    my_callbacks = MetricCallbacks()
    logging.info(hydra_utils.config_to_json(cfg))

    env_fac = CacheEnvWrapperFactory(OmegaConf.to_container(cfg.env_config))
    t_env = env_fac(index=0)
    e_env = env_fac(index=1)

    model = model_utils.get_model(cfg.model_config, cfg.env_config.window_size,
                                  t_env.action_space.n).to(cfg.train_device)
    optimizer = get_optimizer(cfg.optimizer.name, model.parameters(),
                              cfg.optimizer.args)
    replay_buffer = ReplayBuffer(TensorCircularBuffer(cfg.replay_buffer_size),
                                 UniformSampler())
    # A dummy Controller.
    controller = Controller()

    learner = PPOAgent(
        model,
        replay_buffer=replay_buffer,
        controller=controller,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        entropy_coeff=cfg.entropy_coeff,
        model_push_period=sys.maxsize,  # No model push
    )
    t_agent = PPOAgent(model,
                       replay_buffer=replay_buffer,
                       deterministic_policy=False)
    e_agent = PPOAgent(model, deterministic_policy=True)

    t_loop = LoopRunner(t_env,
                        t_agent,
                        replay_buffer,
                        should_update=True,
                        seed=cfg.train_seed,
                        episode_callbacks=my_callbacks)
    e_loop = LoopRunner(e_env,
                        e_agent,
                        replay_buffer=None,
                        should_update=False,
                        seed=cfg.eval_seed,
                        episode_callbacks=my_callbacks)

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        # Train
        stats = StatsDict()
        n_step = cfg.replay_buffer_size // cfg.batch_size
        for _ in track(range(0, cfg.steps_per_epoch, n_step),
                       description="Training..."):
            rollout_stats = t_loop.run(num_steps=cfg.replay_buffer_size)
            stats.update(rollout_stats)

            for _ in range(n_step):
                _, batch, _ = replay_buffer.sample(cfg.batch_size)

                train_stats = learner._train_step(batch)
                stats.extend(train_stats)

        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))

        # Eval
        stats = e_loop.run(num_episodes=cfg.num_eval_episodes)
        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))


if __name__ == "__main__":
    main()
