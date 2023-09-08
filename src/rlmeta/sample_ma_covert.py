import logging
import os
from typing import Dict

import tqdm
import hydra
import torch
import torch.nn

import rlmeta.utils.nested_utils as nested_utils

#from rlmeta.agents.ppo.ppo_agent import PPOAgent
from agents.ppo_agent import PPOAgent
from agents.spec_agent import SpecAgent
from agents.prime_probe_agent import PrimeProbeAgent
from agents.evict_reload_agent import EvictReloadAgent
from agents.cchunter_agent import CCHunterAgent
from agents.benign_agent import BenignAgent
from agents.random_agent import RandomAgent
from agents.cyclone_agent import CycloneAgent
from rlmeta.core.types import Action
from rlmeta.envs.env import Env
from rlmeta.utils.stats_dict import StatsDict

from cache_env_wrapper import CacheCovertSenderReceiverEnvFactory #CacheAttackerDetectorEnvFactory
from cache_ppo_model import CachePPOModel
from cache_ppo_transformer_model import CachePPOTransformerSenderModel, CachePPOTransformerModel


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
    num_total_guess = 0.0
    num_total_correct_guess = 0.0

    if victim_addr == -1:
        timestep = env.reset()
    else:
        timestep = env.reset(victim_address=victim_addr)
    #print("victim address: ", env.env.victim_address )
    #print("Victim domain id: ", env.env.random_domain)
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
        #if env.env.step_count <= 31:
        #    actions['detector'] = Action(0, actions['detector'].info)
        timestep = env.step(actions)

        for agent_name, agent in agents.items():
            agent.observe(actions[agent_name], timestep[agent_name])
        
        episode_length += 1
        episode_return += timestep['receiver'].reward
        is_guess = timestep['receiver'].info.get("is_guess",0)
        correct_guess = timestep['receiver'].info.get("guess_correct",0)
        num_total_guess += is_guess
        num_total_correct_guess += correct_guess

        try:
            detector_action = actions['sender'].action.item()
        except:
            detector_action = actions['sender'].action
        if timestep["__all__"].done and detector_action ==1:
            detector_count += 1
        detector_accuracy = detector_count

    metrics = {
        "episode_length": env.env.step_count,
        "episode_return": episode_return,
        "num_total_guess": num_total_guess,
        "num_total_correct_guess": num_total_correct_guess,
        "detector_accuracy": detector_accuracy,
    }

    return metrics


def run_loops(env: Env,
              agent: PPOAgent,
              num_episodes: int,
              gen_encode: bool = False,
              seed: int = 0) -> StatsDict:
    env.env._env.set_seed(seed)
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
    if gen_encode == False:
        for i in tqdm.tqdm(range(num_episodes)):
            cur_metrics = run_loop(env, agent, victim_addr=-1)
            metrics.extend(cur_metrics)
    else:
        for i in range(env._env.victim_secret_min, env._env.victim_secret_max + 1):
            print("victim addr is "+str(i))
            cur_metrics = run_loop(env, agent, victim_addr = i)
            metrics.extend(cur_metrics)
    
    return metrics


def tournament(env,
               cfg,
               ):
    
    attacker_list = cfg.attackers
    detector_list = cfg.detectors
    spec_trace_f = open('/data/home/jxcui/remix3.txt','r')
    spec_trace = spec_trace_f.read().split('\n')[1000000:]
    y = []
    for line in spec_trace:
        line = line.split()
        y.append(line)
    spec_trace = y
    benign_agent = SpecAgent(cfg.env_config, spec_trace)

    f = open('tournament.txt','w')
    for detector in detector_list:
        for attacker in attacker_list:
            if detector[1].split('/')[-3:-1] == attacker[1].split('/')[-3:-1]:
                print(detector[1].split('/')[-3:-1], attacker[1].split('/')[-3:-1])
                print("from same training instance, skipping")
                print(detector, attacker)
                continue

    
            # Detector
            if "Cyclone" in detector[0]:
                detector_agent = CycloneAgent(cfg.env_config, svm_model_path=detector[1], mode='active')
            elif "CC-Hunter" in detector[0]:
                detector_agent = CCHunterAgent(cfg.env_config)
            elif "None" in detector[0]:
                detector_agent = RandomAgent(1)
            else:
                cfg.model_config["output_dim"] = 2
                cfg.model_config["step_dim"] = 66
                detector_params = torch.load(detector[1], map_location='cuda:1')
                detector_model = CachePPOTransformerModel(**cfg.model_config)
                detector_model.load_state_dict(detector_params)
                detector_model.eval()
                detector_agent = PPOAgent(detector_model, deterministic_policy=cfg.deterministic_policy)
            
            # Attacker
            if attacker[0]=="PrimeProbe":
                env.env.opponent_weights = [0,1]
                attacker_agent = PrimeProbeAgent(cfg.env_config)
            elif attacker[0]=="Benign":
                env.env.opponent_weights = [1,0]
                attacker_agent = PrimeProbeAgent(cfg.env_config)
            else:
                env.env.opponent_weights = [0,1]
                cfg.model_config["output_dim"] = env.action_space.n
                cfg.model_config["step_dim"] = 64
                attacker_params = torch.load(attacker[1])
                attacker_model = CachePPOTransformerModel(**cfg.model_config)
                attacker_model.load_state_dict(attacker_params)
                attacker_model.eval()           
                attacker_agent = PPOAgent(attacker_model, deterministic_policy=cfg.deterministic_policy)
            
            agents = {"attacker": attacker_agent, "detector": detector_agent, "benign": benign_agent}
            metrics = run_loops(env, agents, cfg.num_episodes, cfg.seed)
            
            print(detector)
            print(attacker)
            print(metrics.table())
            f.write(detector[0]+detector[1]+'\n')
            f.write(attacker[0]+attacker[1]+'\n')
            f.write(metrics.table()+'\n')
    f.close()


@hydra.main(config_path="./config", config_name="sample_multiagent_covert")
def main(cfg):
    # Create env
    cfg.env_config['verbose'] = 1 
    env_fac = CacheCovertSenderReceiverEnvFactory(cfg.env_config)
    env = env_fac(0)
    env.env.opponent_weights = [1,0]
    
    # Load model
    # Attacker / Receiver
    cfg.model_config["output_dim"] = env.action_space.n
    attacker_params = torch.load(cfg.attacker_checkpoint)
    print(cfg.model_config)
    attacker_model = CachePPOTransformerModel(**cfg.model_config)
    attacker_model.load_state_dict(attacker_params)
    attacker_model.eval()
    
    # Detector / Sender
    cfg.sender_model_config["output_dim"] = env._env.sender_action_space.n
    cfg.sender_model_config["input_dim"] = env._env.victim_secret_max - env._env.victim_secret_min + 1 
    cfg.sender_model_config["step_dim"] += 2
    detector_params = torch.load(cfg.detector_checkpoint, map_location='cuda:1')
    #detector_model = CachePPOTransformerModel(**cfg.model_config)
    detector_model = CachePPOTransformerSenderModel(**cfg.sender_model_config)
    detector_model.load_state_dict(detector_params)
    detector_model.eval()

    cfg.deterministic_policy = True

    # Create agent
    attacker_agent = PPOAgent(attacker_model, deterministic_policy=cfg.deterministic_policy)
    #attacker_agent = PrimeProbeAgent(cfg.env_config)

    #detector_agent = RandomAgent(1)
    detector_agent = PPOAgent(detector_model, deterministic_policy=cfg.deterministic_policy)
    #detector_agent = CCHunterAgent(cfg.env_config)
    #detector_agent = CycloneAgent(cfg.env_config, svm_model_path=cfg.cyclone_path, mode='active')

    #spec_trace = '/private/home/jxcui/remix3.txt'
    #spec_trace_f = open('/data/home/jxcui/remix3.txt','r')
    #spec_trace = spec_trace_f.read().split('\n')[1000000:]
    agents = {"receiver": attacker_agent, "sender": detector_agent}
    metrics = run_loops(env, agents, cfg.num_episodes, gen_encode=True, seed = cfg.seed) 
    logging.info("\n\n" + metrics.table(info="sample") + "\n")
    '''
    tournament(env, cfg)

    '''
if __name__ == "__main__":
    main()