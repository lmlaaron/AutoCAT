import os
import sys
import copy
import logging
import time
import hydra
import torch
import torch.multiprocessing as mp
import rlmeta.envs.gym_wrappers as gym_wrappers
import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.remote_utils as remote_utils

from rlmeta.agents.agent import AgentFactory
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.model import wrap_downstream_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CachePPOTransformerModel
# from env.cache_attacker_detector_env_factory import CacheAttackerDetectorEnvFactory
from env.cache_attacker_defender_env_factory import CacheAttackerDefenderEnvFactory

from utils.ma_metric_callbacks import MACallbacks
from utils.wandb_logger import WandbLogger, stats_filter
from utils.controller import Phase, Controller
from utils.maloop import LoopList, MAParallelLoop

from utils.trace_parser import load_trace

from agent import RandomAgent, BenignAgent, SpecAgent, PPOAgent
from agent import SpecAgentFactory


@hydra.main(config_path="../config", config_name="macta_defense")
def main(cfg):
    #wandb_logger = WandbLogger(project="cache_attack_detect", config=cfg)
    wandb_logger = WandbLogger(project="RLdefense", config=cfg)
    print(f"working_dir = {os.getcwd()}")
    my_callbacks = MACallbacks()
    logging.info(hydra_utils.config_to_json(cfg))

    #### Define env factory
    # =========================================================================
    env_fac = CacheAttackerDefenderEnvFactory(cfg.env_config)
    unbalanced_env_config = copy.deepcopy(cfg.env_config)
    unbalanced_env_config["opponent_weights"] = [0, 1]  # 100% attacker
    env_fac_unbalanced = CacheAttackerDefenderEnvFactory(unbalanced_env_config)
    benign_env_config = copy.deepcopy(cfg.env_config)
    benign_env_config["opponent_weights"] = [1, 0]  # 100% benign
    env_fac_benign = CacheAttackerDefenderEnvFactory(benign_env_config)
    #print("\033[31m" + "configs check: " + "\033[0m", cfg.env_config)
    # =========================================================================

    #### Define model
    # =========================================================================
    env = env_fac(0)
    #### attacker
    cfg.model_config["output_dim"] = env.action_space.n
    train_model = CachePPOTransformerModel(**cfg.model_config).to(
        cfg.train_device)
    attacker_checkpoint = cfg.attacker_checkpoint
    if len(attacker_checkpoint) > 0:
        attacker_params = torch.load(cfg.attacker_checkpoint, map_location=cfg.train_device)
        train_model.load_state_dict(attacker_params)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.lr)

    infer_model = copy.deepcopy(train_model).to(cfg.infer_device)
    infer_model.eval()

    ctrl = Controller()
    rb = ReplayBuffer(cfg.replay_buffer_size)
    '''
    #### detector
    cfg.model_config["output_dim"] = 2
    cfg.model_config["step_dim"] += 2
    train_model_d = CachePPOTransformerModel(**cfg.model_config).to(
        cfg.train_device_d)
    optimizer_d = torch.optim.Adam(train_model_d.parameters(), lr=cfg.lr)

    infer_model_d = copy.deepcopy(train_model_d).to(cfg.infer_device_d)
    infer_model_d.eval()

    rb_d = ReplayBuffer(cfg.replay_buffer_size)
    '''
    #### defender
    cfg.model_config["output_dim"] = 16  # output_dim = action space
    cfg.model_config["step_dim"] += 2
    train_model_d = CachePPOTransformerModel(**cfg.model_config).to(
        cfg.train_device_d)
    optimizer_d = torch.optim.Adam(train_model_d.parameters(), lr=cfg.lr)

    infer_model_d = copy.deepcopy(train_model_d).to(cfg.infer_device_d)
    infer_model_d.eval()

    rb_d = ReplayBuffer(cfg.replay_buffer_size)
    print('1' * 20)
    
    # =========================================================================

    #### start server
    # =========================================================================
    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(infer_model)
    r_server.add_service(rb)
    c_server.add_service(ctrl)
    md_server = Server(cfg.md_server_name, cfg.md_server_addr)
    rd_server = Server(cfg.rd_server_name, cfg.rd_server_addr)
    md_server.add_service(infer_model_d)
    rd_server.add_service(rb_d)
    servers = ServerList([m_server, r_server, c_server, md_server, rd_server])
    # =========================================================================

    #### Define remote model and control
    # =========================================================================
    a_model = wrap_downstream_model(train_model, m_server)
    t_model = remote_utils.make_remote(infer_model, m_server)
    ea_model = remote_utils.make_remote(infer_model, m_server)
    ed_model = remote_utils.make_remote(infer_model, m_server)
    td_model = remote_utils.make_remote(infer_model, m_server)
    # ---- control
    a_ctrl = remote_utils.make_remote(ctrl, c_server)
    ta_ctrl = remote_utils.make_remote(ctrl, c_server)
    td_ctrl = remote_utils.make_remote(ctrl, c_server)
    ea_ctrl = remote_utils.make_remote(ctrl, c_server)
    ed_ctrl = remote_utils.make_remote(ctrl, c_server)
    # =========================================================================

    a_rb = make_remote_replay_buffer(rb, r_server, prefetch=cfg.prefetch)
    t_rb = make_remote_replay_buffer(rb, r_server)

    agent = PPOAgent(a_model,
                     replay_buffer=a_rb,
                     controller=a_ctrl,
                     optimizer=optimizer,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     dual_clip=cfg.get("dual_clip", None),
                     push_every_n_steps=cfg.push_every_n_steps)
    ta_agent_fac = AgentFactory(PPOAgent, t_model, replay_buffer=t_rb)
    td_agent_fac = AgentFactory(PPOAgent, td_model, deterministic_policy=True)
    ea_agent_fac = AgentFactory(PPOAgent, ea_model, deterministic_policy=True)
    ed_agent_fac = AgentFactory(PPOAgent, ed_model, deterministic_policy=True)
    #### random detector
    '''
    detector = RandomAgent(2)
    t_d_fac = AgentFactory(RandomAgent, 2)
    e_d_fac = AgentFactory(RandomAgent, 2)
    '''
    '''
    #### random benign agent
    benign = BenignAgent(env.action_space.n)
    t_b_fac = AgentFactory(BenignAgent, env.action_space.n)
    e_b_fac = AgentFactory(BenignAgent, env.action_space.n)
    #### spec benign agent
    
    '''
    # spec_trace_f = open('/data/home/jxcui/remix3.txt','r')
    # spec_trace = spec_trace_f.read().split('\n')[:1000000]#[:100000]
    # y = []
    # for line in spec_trace:
    #     line = line.split()
    #     y.append(line)
    # spec_trace = y
    # benign = SpecAgent(cfg.env_config, spec_trace)
    # t_b_fac = AgentFactory(SpecAgent, cfg.env_config, spec_trace)
    # e_b_fac = AgentFactory(SpecAgent, cfg.env_config, spec_trace)
    # spec_trace = load_trace(cfg.trace_file,
    #                         limit=cfg.trace_limit,
    #                         legacy_trace_format=cfg.legacy_trace_format)
    #
    # benign = SpecAgent(cfg.env_config,
    #                    spec_trace,
    #                    legacy_trace_format=cfg.legacy_trace_format)
    # t_b_fac = AgentFactory(SpecAgent,
    #                        cfg.env_config,
    #                        spec_trace,
    #                        legacy_trace_format=cfg.legacy_trace_format)
    # e_b_fac = AgentFactory(SpecAgent,
    #                        cfg.env_config,
    #                        spec_trace,
    #                        legacy_trace_format=cfg.legacy_trace_format)
    #
    t_b_fac = SpecAgentFactory(cfg.env_config,
                               cfg.trace_files,
                               cfg.trace_limit,
                               legacy_trace_format=cfg.legacy_trace_format)
    e_b_fac = SpecAgentFactory(cfg.env_config,
                               cfg.trace_files,
                               cfg.trace_limit,
                               legacy_trace_format=cfg.legacy_trace_format)

    '''
    #### detector agent
    a_model_d = wrap_downstream_model(train_model_d, md_server)
    t_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ea_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ed_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ta_model_d = remote_utils.make_remote(infer_model_d, md_server)
    a_rb_d = make_remote_replay_buffer(rb_d, rd_server, prefetch=cfg.prefetch)
    t_rb_d = make_remote_replay_buffer(rb_d, rd_server)

    agent_d = PPOAgent(a_model_d,
                     replay_buffer=a_rb_d,
                     controller=a_ctrl,
                     optimizer=optimizer_d,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     dual_clip=cfg.get("dual_clip", None),
                     push_every_n_steps=cfg.push_every_n_steps)
    td_d_fac = AgentFactory(PPOAgent, t_model_d, replay_buffer=t_rb_d)
    ta_d_fac = AgentFactory(PPOAgent, ta_model_d, deterministic_policy=True)
    ea_d_fac = AgentFactory(PPOAgent, ea_model_d, deterministic_policy=True)
    ed_d_fac = AgentFactory(PPOAgent, ed_model_d, deterministic_policy=True)
    '''
    #### defender agent
    a_model_d = wrap_downstream_model(train_model_d, md_server)
    t_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ea_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ed_model_d = remote_utils.make_remote(infer_model_d, md_server)
    ta_model_d = remote_utils.make_remote(infer_model_d, md_server)
    a_rb_d = make_remote_replay_buffer(rb_d, rd_server, prefetch=cfg.prefetch)
    t_rb_d = make_remote_replay_buffer(rb_d, rd_server)

    agent_d = PPOAgent(a_model_d,
                       replay_buffer=a_rb_d,
                       controller=a_ctrl,
                       optimizer=optimizer_d,
                       batch_size=cfg.batch_size,
                       learning_starts=cfg.get("learning_starts", None),
                       entropy_coeff=cfg.get("entropy_coeff", 0.01),
                       dual_clip=cfg.get("dual_clip", None),
                       push_every_n_steps=cfg.push_every_n_steps)
    td_d_fac = AgentFactory(PPOAgent, t_model_d, replay_buffer=t_rb_d)
    ta_d_fac = AgentFactory(PPOAgent, ta_model_d, deterministic_policy=True)
    ea_d_fac = AgentFactory(PPOAgent, ea_model_d, deterministic_policy=True)
    ed_d_fac = AgentFactory(PPOAgent, ed_model_d, deterministic_policy=True)
    print('2' * 20)
    #### create agent list - MACTA
    '''
    ta_ma_fac = {"benign":t_b_fac, "attacker":ta_agent_fac, "detector":ta_d_fac}
    td_ma_fac = {"benign":t_b_fac, "attacker":td_agent_fac, "detector":td_d_fac}
    ea_ma_fac = {"benign":e_b_fac, "attacker":ea_agent_fac, "detector":ea_d_fac}
    ed_ma_fac = {"benign":e_b_fac, "attacker":ed_agent_fac, "detector":ed_d_fac}
    '''
    # create agent list - RLdefense
    ta_ma_fac = {"benign": t_b_fac, "attacker": ta_agent_fac, "defender": ta_d_fac}
    td_ma_fac = {"benign": t_b_fac, "attacker": td_agent_fac, "defender": td_d_fac}
    ea_ma_fac = {"benign": e_b_fac, "attacker": ea_agent_fac, "defender": ea_d_fac}
    ed_ma_fac = {"benign": e_b_fac, "attacker": ed_agent_fac, "defender": ed_d_fac}
    print('3' * 20)

    ta_loop = MAParallelLoop(env_fac_unbalanced,
                          ta_ma_fac,
                          ta_ctrl,
                          running_phase=Phase.TRAIN_ATTACKER,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed,
                          episode_callbacks=my_callbacks)
    td_loop = MAParallelLoop(env_fac,
                          td_ma_fac,
                          td_ctrl,
                          # running_phase=Phase.TRAIN_DETECTOR,
                          running_phase=Phase.TRAIN_DEFENDER,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.train_seed,
                          episode_callbacks=my_callbacks)
    ea_loop = MAParallelLoop(env_fac_unbalanced,
                          ea_ma_fac,
                          ea_ctrl,
                          running_phase=Phase.EVAL_ATTACKER,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed,
                          episode_callbacks=my_callbacks)
    ed_loop = MAParallelLoop(env_fac_benign,
                          ed_ma_fac,
                          ed_ctrl,
                          #running_phase=Phase.EVAL_DETECTOR,
                          running_phase=Phase.EVAL_DEFENDER,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=cfg.eval_seed,
                          episode_callbacks=my_callbacks)

    loops = LoopList([ta_loop, td_loop, ea_loop, ed_loop])

    servers.start()
    loops.start()
    agent.connect()
    agent_d.connect()
    a_ctrl.connect()
    #print('4' * 20)
    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        a_stats, d_stats = None, None
        a_ctrl.set_phase(Phase.TRAIN, reset=True)
        #print('6' * 20)
        if epoch % 100 >= 50:
            '''
            # Train Detector
            agent_d.controller.set_phase(Phase.TRAIN_DETECTOR, reset=True)
            d_stats = agent_d.train(cfg.steps_per_epoch)
            # wandb_logger.save(epoch, train_model_d, prefix="detector-")
            torch.save(train_model_d.state_dict(), f"detector-{epoch}.pth")
            '''
            # Train Defender
            #print('7' * 20)
            agent_d.controller.set_phase(Phase.TRAIN_DEFENDER, reset=True)
            d_stats = agent_d.train(cfg.steps_per_epoch)
            # wandb_logger.save(epoch, train_model_d, prefix="defender-")
            torch.save(train_model_d.state_dict(), f"defender-{epoch}.pth")

        else:
            # Train Attacker
            #print('8' * 20)
            agent.controller.set_phase(Phase.TRAIN_ATTACKER, reset=True)
            #print('9' * 20)
            if epoch >= 50:
                a_stats = agent.train(cfg.steps_per_epoch)
            else:
                #print('10' * 20)
                a_stats = agent.train(0)
                #print('11' * 20)
                #print('train_ibrppo_defense.py, L336: a_stats: ', a_stats)
            # wandb_logger.save(epoch, train_model, prefix="attacker-")
            torch.save(train_model.state_dict(), f"attacker-{epoch}.pth")
        # stats = d_stats
        stats = a_stats or d_stats

        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        if epoch % 100 >= 50:
            # train_stats = {"detector": d_stats}
            train_stats = {'defender': d_stats}
        else:
            train_stats = {"attacker": a_stats}
        time.sleep(1)

        a_ctrl.set_phase(Phase.EVAL, limit=cfg.num_eval_episodes, reset=True)
        agent.controller.set_phase(Phase.EVAL_ATTACKER, limit=cfg.num_eval_episodes, reset=True)
        a_stats = agent.eval(cfg.num_eval_episodes)
        # agent_d.controller.set_phase(Phase.EVAL_DETECTOR, limit=cfg.num_eval_episodes, reset=True)
        agent_d.controller.set_phase(Phase.EVAL_DEFENDER, limit=cfg.num_eval_episodes, reset=True)
        d_stats = agent_d.eval(cfg.num_eval_episodes)
        # stats = d_stats
        stats = a_stats

        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))
        eval_stats = {"attacker": a_stats, "defender": d_stats}
        # eval_stats = {"attacker":a_stats, "detector":d_stats}
        time.sleep(1)

        # wandb_logger.log(train_stats, eval_stats)

    #print('5' * 20)
    loops.terminate()
    servers.terminate()


def add_prefix(input_dict, prefix=''):
    res = {}
    for k, v in input_dict.items():
        res[prefix+str(k)] = v
    return res


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()