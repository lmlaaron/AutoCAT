''' updated to work without rlmeta environment. action offsets modified to universal cache configuration as below:
flush_inst: true
attacker_addr_s: 8
attacker_addr_e: 23
victim_addr_s: 8
victim_addr_e: 15
'''

class FlushReloadAgent():

    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] # n-ways in N
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 0 # 0 or N*M
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 15
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 7
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False   
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
    # initialize the agent with an observation
    def observe_init(self, timestep):
        # initialization doing nothing
        self.local_step = 0
        self.lat = []
        self.no_prime = False
        return

    # returns an action
    def act(self, timestep):
        info = {}
        #if timestep[0][0] == -1:
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            #from IPython import embed; embed()
            self.local_step = 0
            self.lat=[]
            self.no_prime = False

        # flush phase
        if self.local_step < self.cache_size - ( self.cache_size if self.no_prime else 0 ): 
            action = self.local_step + 2 * self.cache_size - (self.cache_size if self.no_prime else 0 ) 
            self.local_step += 1
            return action, info

        # do victim trigger
        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ): 
            action = 4 * self.cache_size 
            self.local_step += 1
            return action, info

        # reload phase
        elif self.local_step < 2 * self.cache_size -(self.cache_size if self.no_prime else 0 ) +1: 
            action = self.local_step - self.cache_size - (self.cache_size if self.no_prime else 0 ) -1  
            self.local_step += 1
            return action, info

        # is_guess = 1
        # victim_addr = action - ( len(self.attacker_address_space) + 1 + 1) --> victim addr = action - 18
        elif self.local_step == 2 * self.cache_size - (self.cache_size if self.no_prime else 0 ) + 1:# - 1 - 1: # do guess and terminate
            action = self.local_step + 2 * self.cache_size + 2 * len(self.lat) +1 # default assume that last is hit
            for addr in range(1, len(self.lat)):
                if self.lat[addr] == 0: # 0 for hit, 1 for miss
                    action =  addr + 4 * self.cache_size -1
                    break
            self.local_step = 0 # reset the attacker 
            self.lat=[] # reset the attacker
            self.no_prime = True # reset the attacker
            if action > self.cache_size:
                action+=1
            return action, info
        else: 
            assert(False)
    
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):#- 1:
            #self.local_step += 1
            #self.lat.append(timestep[0][0])
            self.lat.append(timestep.observation[0][0])
        return
