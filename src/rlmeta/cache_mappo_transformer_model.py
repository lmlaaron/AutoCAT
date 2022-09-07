import os
import sys

from typing import Dict, List, Tuple

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import rlmeta.core.remote as remote
import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_model import PPOModel


class CacheMAPPOTransformerModel(PPOModel):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 victim_addr_dim: int,
                 victim_embed_dim: int,
                 cache_state_size: int,
                 cache_state_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                          self.action_embed_dim + self.step_embed_dim)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.victim_addr_dim = victim_addr_dim
        self.victim_embed_dim = victim_embed_dim
        self.cache_state_size = cache_state_size
        self.cache_state_dim = cache_state_dim
        self.value_net_dim = (self.hidden_dim + self.victim_embed_dim +
                              self.cache_state_size * self.cache_state_dim)

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)

        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim)
        # self.linear_o = nn.Linear(self.hidden_dim * self.window_size,
        #                           self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim,
                                                   nhead=8,
                                                   dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.linear_a = nn.Linear(self.hidden_dim, self.output_dim)

        self.victim_addr_embed = nn.Embedding(self.victim_addr_dim,
                                              self.victim_embed_dim)
        self.linear_v0 = nn.Linear(self.value_net_dim, self.hidden_dim)
        self.linear_v1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_v2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, victim_addr, cache_state = obs
        obs = obs.to(torch.int64)
        assert obs.dim() == 3

        batch_size = obs.size(0)
        l, v, act, stp = torch.unbind(obs, dim=-1)
        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_one_hot(v, self.victim_acc_dim)
        act = self.make_embedding(act, self.action_embed)
        stp = self.make_embedding(stp, self.step_embed)

        x = torch.cat((l, v, act, stp), dim=-1)
        x = self.linear_i(x)
        x = x.transpose(0, 1).contiguous()
        h = self.encoder(x)
        # h = self.linear_o(h.view(batch_size, -1))
        h = h.mean(dim=0)

        p = self.linear_a(h)
        logpi = F.log_softmax(p, dim=-1)

        victim_addr = self.make_embedding(victim_addr, self.victim_addr_embed)
        cache_state = self.make_one_hot(cache_state, self.cache_state_dim)
        value_net_input = torch.cat((h, victim_addr.view(
            batch_size, -1), cache_state.view(batch_size, -1)),
                                    dim=-1)
        # ResNet block
        z = self.linear_v0(value_net_input)
        v = self.linear_v1(F.relu(z))
        v = self.linear_v1(F.relu(v))
        v = self.linear_v(z + v)

        return logpi, v

    @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            # x = obs.to(self._device)
            x = nested_utils.map_nested(lambda x: x.to(self._device), obs)
            d = deterministic_policy.to(self._device)
            logpi, v = self.forward(x)

            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(d, greedy_action, sample_action)
            logpi = logpi.gather(dim=-1, index=action)

            return action.cpu(), logpi.cpu(), v.cpu()
