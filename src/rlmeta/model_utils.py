from typing import Any, Dict

import torch
import torch.nn as nn

from cache_ppo_lstm_model import CachePPOLstmModel
from cache_ppo_transformer_model import CachePPOTransformerModel


def get_model(cfg: Dict[str, Any]) -> nn.Module:
    model_type = cfg.pop("type")

    model = None
    if model_type == "transformer":
        model = CachePPOTransformerModel(**cfg)
    elif model_type == "lstm":
        model = CachePPOLstmModel(**cfg)

    return model
