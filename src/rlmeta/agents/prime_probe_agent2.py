''' action offsets modified for 1set-8way. Prepared for ICLR rebuttal.'''
#   allow_victim_multi_access: true
#   allow_empty_victim_access: true
#   attacker_addr_s: 8
#   attacker_addr_e: 15
#   victim_addr_s: 0
#   victim_addr_e: 0
#   victim_rand_s: 0
#   victim_rand_e: 7
#   cache_configs:
#       blocks: 8
#       associativity: 8
#       rep_policy: "lru"

class PrimeProbeAgent:

    def __init__(self, env_config):
        self.local_step = 0
        self.lat = []
        self.no_prime = False # set to true after first prime
        if "cache_configs" in env_config:
            
            self.configs = env_config["cache_configs"]
            self.num_ways = self.configs['cache_1']['associativity'] 
            self.cache_size = self.configs['cache_1']['blocks']
            attacker_addr_s = env_config["attacker_addr_s"] if "attacker_addr_s" in env_config else 8
            attacker_addr_e = env_config["attacker_addr_e"] if "attacker_addr_e" in env_config else 15
            victim_addr_s = env_config["victim_addr_s"] if "victim_addr_s" in env_config else 0
            victim_addr_e = env_config["victim_addr_e"] if "victim_addr_e" in env_config else 7
            #victim_rand_s = env_config["victim_rand_s"] if "victim_rand_s" in env_config else 0
            #victim_rand_e = env_config["victim_rand_e"] if "victim_rand_e" in env_config else 7
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
        if timestep.observation[0][0][0] == -1:
            #reset the attacker
            #from IPython import embed; embed()
            self.local_step = 0
            self.lat=[]
            self.no_prime = False

        # do prime 
        if self.local_step < self.cache_size -  ( self.cache_size if self.no_prime else 0 ):# +1:
            action = self.local_step #+ self.cache_size - (self.cache_size if self.no_prime else 0) 
            self.local_step += 1
            #print(self.local_step)
            #print(action)
            return action, info

        # do victim trigger
        elif self.local_step == self.cache_size - (self.cache_size if self.no_prime else 0 ):#+1:
            action = self.cache_size  # do victim access
            self.local_step += 1
            #print(self.local_step)
            #print(action)
            return action, info

        # do probe
        elif self.local_step < 2 * self.cache_size - (self.cache_size if self.no_prime else 0 ) +1 :
            action = self.local_step - 9 - (self.cache_size if self.no_prime else 0 ) 
            self.local_step += 1
            if action > self.cache_size:
                action += 1
            #print(self.local_step)
            #print(action)
            return action, info

        # do guess and terminate
        elif self.local_step == 2 * self.cache_size - (self.cache_size if self.no_prime else 0 ) +1:
            '''#action = 2 * self.cache_size   # default assume that last is miss
            #action = 1 * self.cache_size + 4
            #for addr in range(1, len(self.lat)):
                #print(self.lat[addr])
                if self.lat[addr].int() == 1: # miss
                    
                    action = addr + 1 * self.cache_size 
                    break
            self.local_step = 0
            self.lat=[]
            self.no_prime = True
            if action > self.cache_size:
                action+=1
            #print(self.lat)
            #print(action)

            
            '''
            action = 10 +1 #1 * self.cache_size 
            if self.lat[5].int()==1:
                action = 10 #action + 1
            else:
                action = 11 # action + 2
            return action, info
        else:        
            assert(False)
            
    def observe(self, action, timestep):
        if self.local_step < 2 * self.cache_size + 1 + 1 - (self.cache_size if self.no_prime else 0 ) and self.local_step > self.cache_size - (self.cache_size if self.no_prime else 0 ):
        ##    self.local_step += 1
            self.lat.append(timestep[0][0][0])
        return
