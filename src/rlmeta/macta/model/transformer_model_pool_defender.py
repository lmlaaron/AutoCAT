import os
import sys

from typing import Dict, List, Tuple, Optional, Union

import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_model import PPOModel
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.server import Server

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from cache_ppo_transformer_model_defender import CachePPOTransformerModelDefender


class CachePPOTransformerModelPoolDefender(CachePPOTransformerModelDefender):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__(latency_dim, victim_acc_dim, action_dim, step_dim,
                         window_size, action_embed_dim, step_embed_dim,
                         hidden_dim, output_dim, num_layers)
        self.history = []
        self.latest = None
        self.use_history = False

    # @remote.remote_method(batch_size=128)
    # def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor,
    #         reload_model: bool
    #         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if reload_model:
    #         if self.use_history and len(self.history) > 0:
    #             state_dict = random.choice(self.history)
    #             self.load_state_dict(state_dict)
    #         elif self.latest is not None:
    #             self.load_state_dict(self.latest)
    #         #print("reloading model", reload_model)
    #         #print("length of history:", len(self.history), "use history:", self.use_history, "latest:", self.latest if self.latest is None else len(self.latest))

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.use_history and len(self.history) > 0:
            state_dict = random.choice(self.history)
            self.load_state_dict(state_dict)
        elif self.latest is not None:
            self.load_state_dict(self.latest)

        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            try:
                x = obs.to(self._device)
            except:
                print("look at obs --------------------------------------------------> ", obs)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)


            logpi_layer1 = logpi[:, :2]  # Log probabilities for output layer 1
            logpi_layer2 = logpi[:, 2:4]  # Log probabilities for output layer 2
            logpi_layer3 = logpi[:, 4:6]  # Log probabilities for output layer 3
            logpi_layer4 = logpi[:, 6:]  # Log probabilities for output layer 4

            sample_action1 = logpi_layer1.exp().multinomial(1, replacement=True)
            sample_action2 = logpi_layer2.exp().multinomial(1, replacement=True)
            sample_action3 = logpi_layer3.exp().multinomial(1, replacement=True)
            sample_action4 = logpi_layer4.exp().multinomial(1, replacement=True)

            greedy_action1 = logpi_layer1.argmax(-1, keepdim=True)
            greedy_action2 = logpi_layer2.argmax(-1, keepdim=True)
            greedy_action3 = logpi_layer3.argmax(-1, keepdim=True)
            greedy_action4 = logpi_layer4.argmax(-1, keepdim=True)

            action1 = torch.where(d, greedy_action1, sample_action1)
            action2 = torch.where(d, greedy_action2, sample_action2)
            action3 = torch.where(d, greedy_action3, sample_action3)
            action4 = torch.where(d, greedy_action4, sample_action4)

            combined_action = torch.cat((action1, action2, action3, action4), dim=1)
            new_action = torch.sum(combined_action * (2 ** torch.arange(combined_action.size(1)-1, -1, -1).float().to(combined_action.device)), dim=1)
            
            logpi1 = logpi_layer1.gather(dim=-1, index=action1)
            logpi2 = logpi_layer2.gather(dim=-1, index=action2)
            logpi3 = logpi_layer3.gather(dim=-1, index=action3)
            logpi4 = logpi_layer4.gather(dim=-1, index=action4)
            total_logpi = logpi1 + logpi2 + logpi3 + logpi4
            

            '''
            new_action = []
            total_logpi = []
            if len(logpi.shape) == 1:
                max_logpi = 0
                reshaped_tensor = logpi.reshape(-1, 2)
                max_indices = torch.argmax(reshaped_tensor, dim=1)
                max_logpi = torch.max(reshaped_tensor, dim=1)[0].sum().item()
                actual_action = 0
                for i in range(len(max_indices)):
                    actual_action += int(max_indices[i]) * (2**i)
                new_action = torch.tensor(actual_action)
                total_logpi = torch.tensor(max_logpi)
            else:
                for lgp in logpi:
                    max_logpi = 0
                    reshaped_tensor = lgp.reshape(-1, 2)
                    max_indices = torch.argmax(reshaped_tensor, dim=1)
                    max_logpi = torch.max(reshaped_tensor, dim=1)[0].sum().item()
                    total_logpi.append(max_logpi)
                    actual_action = 0
                    for i in range(len(max_indices)):
                        actual_action += int(max_indices[i]) * (2**i)
                    new_action.append(actual_action)
                new_action = torch.tensor(new_action)
                total_logpi = torch.tensor(total_logpi)
            tensor_list = [torch.tensor([value]) for value in new_action]
            new_action = torch.stack(tensor_list)
            tensor_list = [torch.tensor([value]) for value in total_logpi]
            total_logpi = torch.stack(tensor_list)
            '''
            # print("this is in pool defender   ", total_logpi)

            # print(new_action, "  __---__--__--  ", total_logpi)

            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", type(deterministic_policy), "  _______________________", deterministic_policy)
            # print("*******", logpi)
            ##greedy_action = logpi.argmax(-1, keepdim=True)
            ########logpi = logpi.gather(dim=-1, index=greedy_action)
            ########indices = torch.eq(greedy_action, 1)
            ########greedy_action[indices] = 15
            ##sample_action = logpi.exp().multinomial(1, replacement=True)
            ##action = torch.where(d, greedy_action, sample_action)
            # print(action, "  ()()()()() ", type(action))
            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", sample_action, sample_action.shape, type(sample_action), "  _______________________", logpi)
            # print("^^^^^^^^^^^in cache ppo transformer model defender  ", action.shape, type(int(logpi[0][3])), int(logpi[0][3]), sample_action[0][1], "  _______________________", logpi)
            # print("$$$$$$$$", torch.exp(logpi))
            ##logpi = logpi.gather(dim=-1, index=action)
            # print("*********", logpi)
            # print("&&&&&&&&&", total_logpi, type(total_logpi))
            # print("**************************in cache ppo transformer model defender  ", greedy_action, "%%%%%%%%%%%%", sample_action, "  &&&&&&&&&&&&&&&&&&&&&&&&", logpi)
            ####greedy_action = logpi.argmax(-1, keepdim=True)
            ####sample_action = logpi.exp().multinomial(1, replacement=True)
            ####action = torch.where(d, greedy_action, sample_action)
            ####logpi = logpi.gather(dim=-1, index=action)
            ##return action.cpu(), logpi.cpu(), v.cpu()
            return new_action.cpu(), total_logpi.cpu(), v.cpu()
            ########return greedy_action.cpu(), logpi.cpu(), v.cpu()

    @remote.remote_method(batch_size=None)
    def push(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.load_state_dict(state_dict)

    @remote.remote_method(batch_size=None)
    def push_to_history(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Move state_dict to device before loading.
        # https://github.com/pytorch/pytorch/issues/34880
        device = next(self.parameters()).device
        state_dict = nested_utils.map_nested(lambda x: x.to(device),
                                             state_dict)
        self.latest = state_dict
        self.history.append(self.latest)

    @remote.remote_method(batch_size=None)
    def set_use_history(self, use_history: bool) -> None:
        print("set use history", use_history)
        self.use_history = use_history
        print("after setting:", self.use_history)


class DownstreamModelPool(DownstreamModel):
    def __init__(self,
                 model: nn.Module,
                 server_name: str,
                 server_addr: str,
                 name: Optional[str] = None,
                 timeout: float = 60) -> None:
        super().__init__(model, server_name, server_addr, name, timeout)

    def set_use_history(self, use_history):
        self.client.sync(self.server_name,
                         self.remote_method_name("set_use_history"),
                         use_history)

    def push_to_history(self) -> None:
        state_dict = self.wrapped.state_dict()
        state_dict = nested_utils.map_nested(lambda x: x.cpu(), state_dict)
        self.client.sync(self.server_name,
                         self.remote_method_name("push_to_history"),
                         state_dict)


ModelLike = Union[nn.Module, RemotableModel, DownstreamModel, remote.Remote]


def wrap_downstream_model(model: nn.Module,
                          server: Server,
                          name: Optional[str] = None,
                          timeout: float = 60) -> DownstreamModel:
    return DownstreamModelPool(model, server.name, server.addr, name, timeout)
