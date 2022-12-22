import logging

from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn

import rlmeta.utils.nested_utils as nested_utils

from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

import model_utils

from cache_env_wrapper import CacheEnvWrapperFactory
from metric_callbacks import MetricCallbacks
from loop_runner import LoopRunner

def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int = -1,
              seed: int = 0,
              reset_cache_state: bool = False) -> StatsDict:
    metric_callbacks = MetricCallbacks()
    loop = LoopRunner(env,
                      agent,
                      replay_buffer=None,
                      should_update=False,
                      seed=seed,
                      episode_callbacks=metric_callbacks)

    metrics = StatsDict()
    if num_episodes == -1:
        start = env.env.victim_address_min
        stop = env.env.victim_address_max + 1 + int(
            env.env.allow_empty_victim_access)
        for victim_addr in range(start, stop):
            cur_metrics = loop.run(num_episodes=1,
                                   victim_address=victim_addr,
                                   reset_cache_state=reset_cache_state)
            # Only one StatsItem, use mean value should be enough.
            cur_metrics = {k: v["mean"] for k, v in cur_metrics.dict().items()}
            metrics.extend(cur_metrics)
    else:
        metrics = loop.run(num_episodes=num_episodes,
                           victim_address=-1,
                           reset_cache_state=reset_cache_state)

    return metrics


@hydra.main(config_path="./config", config_name="sample_attack")
def main(cfg):
    # Create env
    cfg.env_config.verbose = 1
    env_fac = CacheEnvWrapperFactory(OmegaConf.to_container(cfg.env_config))
    env = env_fac(index=0)

    # Load model
    model = model_utils.get_model(cfg.model_config, cfg.env_config.window_size,
                                  env.action_space.n, cfg.checkpoint)
    model.eval()

    # Create agent
    agent = PPOAgent(model, deterministic_policy=cfg.deterministic_policy)

    # Run loops
    metrics = run_loops(env, agent, cfg.num_episodes, cfg.seed,
                        cfg.reset_cache_state)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
