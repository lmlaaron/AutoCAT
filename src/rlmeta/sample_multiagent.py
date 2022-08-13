import logging

from typing import Dict

import hydra
import torch
import torch.nn

import rlmeta.utils.nested_utils as nested_utils

#from rlmeta.agents.ppo.ppo_agent import PPOAgent
from agents.ppo_agent import PPOAgent
from agents.spec_agent import SpecAgent
from agents.prime_probe_agent import PrimeProbeAgent
from agents.evict_reload_agent import EvictReloadAgent
from agents.benign_agent import BenignAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

from cache_env_wrapper import CacheAttackerDetectorEnvFactory
from cache_ppo_model import CachePPOModel
from cache_ppo_transformer_model import CachePPOTransformerModel


def unbatch_action(action: Action) -> Action:
    act, info = action
    act.squeeze_(0)
    info = nested_utils.map_nested(lambda x: x.squeeze(0), info)
    return Action(act, info)


def run_loop(env: Env, agents: PPOAgent, victim_addr=-1) -> Dict[str, float]:
    episode_length = 0
    episode_return = 0.0
    detector_count = 0.0
    detector_acc = 0.0
    
    env.env.opponent_weights = [0,1]
    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)
    print("victim address: ", env.env.victim_address ) 
    for agent_name, agent in agents.items():
        agent.observe_init(timestep[agent_name])
    while not timestep["__all__"].done:
        # Model server requires a batch_dim, so unsqueeze here for local runs.
        actions = {}
        for agent_name, agent in agents.items():
            timestep[agent_name].observation.unsqueeze_(0)
            #print("attacker obs")
            #print(timestep["attacker"].observation)
            action = agent.act(timestep[agent_name])
            # Unbatch the action.
            if isinstance(action, tuple):
                action = Action(action[0], action[1])
            if not isinstance(action.action, int):
                action = unbatch_action(action)
            actions.update({agent_name:action})
        #print(actions)
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], timestep[agent_name])
        
        episode_length += 1
        episode_return += timestep['attacker'].reward
        if timestep["__all__"].done and actions['detector'].action.item()==1:
            detector_count += 1
        detector_accuracy = detector_count

    metrics = {
        "episode_length": episode_length,
        "episode_return": episode_return,
        "detector_accuracy": detector_accuracy,
    }

    return metrics


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int,
              seed: int = 0) -> StatsDict:
    env.seed(seed)
    metrics = StatsDict()
    if env.env._env.allow_empty_victim_access == False:
        end_address = env.env._env.victim_address_max + 1
    else:
        end_address = env.env._env.victim_address_max + 1 + 1
    '''
    for victim_addr in range(env.env._env.victim_address_min, end_address):
        cur_metrics = run_loop(env, agent, victim_addr=victim_addr)
        metrics.extend(cur_metrics)
    '''
    for i in range(num_episodes):
        cur_metrics = run_loop(env, agent, victim_addr=-1)
        metrics.extend(cur_metrics)
    return metrics


@hydra.main(config_path="./config", config_name="sample_multiagent")
def main(cfg):
    # Create env
    cfg.env_config['verbose'] = 1
    env_fac = CacheAttackerDetectorEnvFactory(cfg.env_config)
    env = env_fac(index=0)
    
    # Load model
    '''
    cfg.model_config["output_dim"] = env.action_space.n
    attacker_params = torch.load(cfg.attacker_checkpoint)
    attacker_model = CachePPOTransformerModel(**cfg.model_config)
    attacker_model.load_state_dict(attacker_params)
    attacker_model.eval()
    '''
    cfg.model_config["output_dim"] = 2
    cfg.model_config["step_dim"] += 2
    detector_params = torch.load(cfg.detector_checkpoint, map_location='cuda:1')
    detector_model = CachePPOTransformerModel(**cfg.model_config)
    detector_model.load_state_dict(detector_params)
    detector_model.eval()

    # Create agent
    #attacker_agent = PPOAgent(attacker_model, deterministic_policy=cfg.deterministic_policy)
    attacker_agent = PrimeProbeAgent(cfg.env_config)
    detector_agent = PPOAgent(detector_model, deterministic_policy=cfg.deterministic_policy)
    #spec_trace = '/private/home/jxcui/remix3.txt'
    spec_trace_f = open('/private/home/jxcui/remix3.txt','r')
    spec_trace = spec_trace_f.read().split('\n')#[:100000]
    y = []
    for line in spec_trace:
        line = line.split()
        y.append(line)
    spec_trace = y
    benign_agent = SpecAgent(cfg.env_config, spec_trace)
    agents = {"attacker": attacker_agent, "detector": detector_agent, "benign": benign_agent}
    # Run loops
    metrics = run_loops(env, agents, cfg.num_episodes, cfg.seed)
    logging.info("\n\n" + metrics.table(info="sample") + "\n")


if __name__ == "__main__":
    main()
