import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike, ModelVersion
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.rescalers import Rescaler, RMSRescaler
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import Tensor, NestedTensor
from rlmeta.utils.stats_dict import StatsDict
from rlmeta.agents.ppo.ppo_agent import PPOAgent

console = Console()


class PPOAgent(PPOAgent):

    def __init__(self,
                 model: ModelLike,
                 deterministic_policy: bool = False,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 controller: Optional[ControllerLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 512,
                 max_grad_norm: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ratio_clipping_eps: float = 0.2,
                 value_clipping_eps: Optional[float] = 0.2,
                 vf_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 rescale_reward: bool = True,
                 max_abs_reward: float = 10.0,
                 normalize_advantage: bool = True,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 10,
                 local_batch_size: int = 1024) -> None:
        super().__init__(model, deterministic_policy, replay_buffer,
                         controller, optimizer, batch_size, max_grad_norm,
                         gamma, gae_lambda, ratio_clipping_eps, value_clipping_eps, vf_loss_coeff,
                         entropy_coeff, rescale_reward, max_abs_reward, 
                         normalize_advantage,
                         learning_starts, model_push_period, local_batch_size)

    def set_use_history(self, use_history):
        self._model.set_use_history(use_history)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        try:
            self._model.version = await self._model.async_sample_model()
        except:
            pass
        if self._replay_buffer is None:
            return
        obs, _, done, _ = timestep
        if done:
            self._trajectory.clear()
        else:
            self._trajectory = [{"obs": obs, "done": done}]    
    
    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = self._model.act(
            obs, torch.tensor([self._deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = await self._model.async_act(
            obs, torch.tensor([self._deterministic_policy]))
        return Action(action, info={"logpi": logpi, "v": v})
    
    def train(self,
              num_steps: int,
              keep_evaluation_loops: bool = False) -> StatsDict:
        phase = self._controller.phase()
        self._replay_buffer.warm_up(self._learning_starts)
        stats = StatsDict()

        console.log(f"Training for num_steps = {num_steps}")
        for _ in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            _, batch, _ = self._replay_buffer.sample(self._batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(batch)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time/ms": (t1 - t0) * 1000.0,
                "batch_learn_time/ms": (t2 - t1) * 1000.0,
            }
            stats.extend(step_stats)
            stats.extend(time_stats)
            
            self._step_counter += 1
            if self._step_counter % self._model_push_period == 0:
                self._model.push()
        self._model.push() 
        episode_stats = self._controller.stats(phase)
        stats.update(episode_stats)
        self._controller.reset_phase(phase)
        return stats
    
    def eval(self,
             num_episodes: Optional[int] = None,
             keep_training_loops: bool = False,
             non_blocking: bool = False
            ) -> Union[StatsDict, Future]:
        print("Local PPO agent eval")
        phase = self._controller.phase()
        while self._controller.count(phase) < num_episodes:
            print(self._controller.count(phase))
            time.sleep(1)
        stats = self._controller.stats(phase)
        self._controller.reset_phase(phase)
        return stats
