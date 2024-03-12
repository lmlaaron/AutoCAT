# a defender who locks all cache lines with no consideration
# is going to be used against the ppo attacker to see how the attacker reacts
# and what is the optimum behaviour of the defender. 
import numpy as np


class RandLockerAgent:

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 4
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 7
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 3
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            # assert(self.num_ways == 1) # currently only support direct-map cache
            assert(flush_inst == False) # do not allow flush instruction
            # assert(attacker_addr_e - attacker_addr_s == victim_addr_e - victim_addr_s ) # address space must be shared
            #must be no shared address space
            assert( ( attacker_addr_e + 1 == victim_addr_s ) or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)

    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        return


    # returns an action
    def act(self, timestep):
        info = {}
        if timestep.observation[0][0][0] == -1:
            self.local_step = 0
        cur_step_obs = timestep.observation[0][0]
        info = timestep.info
        latency = cur_step_obs[0] #if self.keep_latency else -1
        domain_id = cur_step_obs[1]
        locked_lines = cur_step_obs[2]
        way_accessed = cur_step_obs[4]
        set_accessed = cur_step_obs[5]
        action = np.random.randint(low=0, high=15)
        return action

    # is it useful for non-ML agent or not???
    def observe(self, action, timestep):
        return
