import os
import sys
sys.path.append(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from cache_ppo_transformer_model import CachePPOTransformerModel
from .transformer_model_pool import CachePPOTransformerModelPool
