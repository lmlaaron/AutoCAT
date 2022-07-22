import time

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from rich.console import Console
from rich.progress import track

import rlmeta.utils.data_utils as data_utils
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Controller, ControllerLike, Phase
from rlmeta.core.model import ModelLike
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
                 batch_size: int = 128,
                 grad_clip: float = 1.0,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 vf_loss_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 reward_rescaling: bool = True,
                 advantage_normalization: bool = True,
                 value_clip: bool = True,
                 learning_starts: Optional[int] = None,
                 push_every_n_steps: int = 1) -> None:
        super().__init__(model, deterministic_policy, replay_buffer,
                         controller, optimizer, batch_size, grad_clip,
                         gamma, gae_lambda, eps_clip, vf_loss_coeff,
                         entropy_coeff, reward_rescaling, 
                         advantage_normalization, value_clip,
                         learning_starts, push_every_n_steps)

    def set_use_history(self, use_history):
        self.model.set_use_history(use_history)

