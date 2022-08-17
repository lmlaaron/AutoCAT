class EvictReloadAgent():

    # the config is the same as the config cor cache_guessing_game_env_impl
    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            #self.logger.info('Load config from JSON')
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] # n-ways in N
            self.cache_size = self.configs['cache_1']['blocks']
            
            # self.n_sets = self.cache_size / self.num_ways # Total number of sets (M) in the cache
        
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 0 # 0 or N*M
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 15
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 7
            flush_inst = env_config["flush_inst"] if "flush_inst" in env_config else False            
            self.allow_empty_victim_access = env_config["allow_empty_victim_access"] if "allow_empty_victim_access" in env_config else False
            
            assert(self.num_ways == 1) # currently only support direct-map cache
            assert(self.cache_size == 8) # assume the cache config is for 8 sets 
            assert(flush_inst == False) # do not allow flush instruction
            #assert((attacker_addr_e - attacker_addr_s ) == (2 * (victim_addr_e - victim_addr_s )+1)) # address space must be shared
            
            assert( attacker_addr_s == victim_addr_s)
            #assert( ( attacker_addr_e + 1 == victim_addr_s )) # or ( victim_addr_e + 1 == attacker_addr_s ) )
            assert(self.allow_empty_victim_access == False)
            #assert(attacker_addr_s == 0)
            #assert(attacker_addr_e == 15)
            #assert(victim_addr_s == 0)
            #assert(victim_addr_e == 7)

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
        if timestep[0][0] == -1:
            #reset the attacker
            #from IPython import embed; embed()
            self.local_step = 0
            self.lat=[]
            self.no_prime = False

        # do evict
        '''
        if self.flush_inst == False:
            if action < len(self.attacker_address_space): ---> action = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
                address = action ---> 
            elif action == len(self.attacker_address_space): --> action = 16
                is_victim = 1
            elif action == len(self.attacker_address_space)+1: --> action = 17
                is_victim_random = 1
            else: 
                is_guess = 1
                victim_addr = action - ( len(self.attacker_address_space) + 1 + 1) --> victim addr = action - 18
        '''
        # evict phase
        if self.local_step < self.cache_size - ( self.cache_size if self.no_prime else 0 ): # action = 8,9,10,11,12,13,14,15
            action = self.local_step + self.cache_size - (self.cache_size if self.no_prime else 0 ) 
            self.local_step += 1
            return action, info

        # do victim trigger
        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ): # action = 16
            action = 2 * self.cache_size # do victim access, because action==len(self.attacker_address_space) then is_victim = 1
            self.local_step += 1
            return action, info

        # reload phase
        elif self.local_step < 2 * self.cache_size -(self.cache_size if self.no_prime else 0 ): # action = 0,1,2,3,4,5,6,7
            action = self.local_step - self.cache_size - (self.cache_size if self.no_prime else 0 ) -1  
            self.local_step += 1
            
            if action > self.cache_size: # why?
                action += 1
            
            return action, info

        # is_guess = 1
        # victim_addr = action - ( len(self.attacker_address_space) + 1 + 1) --> victim addr = action - 18
        elif self.local_step == 2 * self.cache_size - (self.cache_size if self.no_prime else 0 ):# - 1 - 1: # do guess and terminate
            #action = 2 * self.cache_size # default assume that last is hit
            for addr in range(1, len(self.lat)):
                if self.lat[addr].int() == 0: # 0 for hit, 1 for miss
                    action =  addr + self.cache_size 
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
        ##    self.local_step += 1
            self.lat.append(timestep.observation[0][0])
        return
