import torch
import torch.nn as nn

from typing import Optional

import sys

sys.path.append("..")

from models.dnn import DNNEncoder
#from ray.rllib.models import ModelCatalog
#from ray.rllib.models.tf.tf_modelv2 import TFModelV2
#from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class DQNModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_blocks: Optional[int] = 1) -> None:
        super(DQNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_blocks = num_blocks

        self.backbone = DNNEncoder(self.input_dim, self.hidden_dim,
                                   self.hidden_dim, self.num_blocks)
        self.linear_a = nn.Linear(self.hidden_dim, self.action_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        a = self.linear_a(h)
        v = self.linear_v(h)
        return v + a - a.mean(-1, keepdim=True)