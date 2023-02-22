import block
import random
INVALID_TAG = '--------'


# interface for cache replacement policy per set
class rep_policy:
    def __init__(self):
        self.verbose = False

    def touch(self, tag, timestamp):
        pass

    def reset(self, tag, timestamp):
        pass

    def invalidate(self, tag):
        pass

    def find_victim(self, timestamp):
        pass

    def vprint(self, *args):
        if self.verbose == 1:
            print( " "+" ".join(map(str,args))+" ")

# LRU policy
class lru_policy(rep_policy):
    def __init__(self, associativity, block_size, verbose=False):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}
        self.verbose = verbose

    def touch(self, tag, timestamp):
        assert(tag in self.blocks)
        self.blocks[tag].last_accessed = timestamp

    def reset(self, tag, timestamp):
        return self.touch(tag, timestamp)

    def instantiate_entry(self, tag, timestamp):
        assert(tag not in self.blocks)
        self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)

    #def reset(self, tag):
    def invalidate(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag]

    def find_victim(self, timestamp):
        in_cache = list(self.blocks.keys())
        victim_tag = in_cache[0] 
        for b in in_cache:
            self.vprint(b + ' '+ str(self.blocks[b].last_accessed))
            if self.blocks[b].last_accessed < self.blocks[victim_tag].last_accessed:
                victim_tag = b
        return victim_tag 

# random replacement policy
class rand_policy(rep_policy):
    def __init__(self, associativity, block_size, verbose=False):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}
        self.verbose = verbose

    def touch(self, tag, timestamp):
        assert(tag in self.blocks)
        self.blocks[tag].last_accessed = timestamp

    def reset(self, tag, timestamp):
        return self.touch(tag, timestamp)

    def instantiate_entry(self, tag, timestamp):
        assert(tag not in self.blocks)
        self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)

    def invalidate(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag]

    def find_victim(self, timestamp):
        in_cache = list(self.blocks.keys())
        index = random.randint(0,len(in_cache)-1)
        victim_tag = in_cache[index] 
        return victim_tag


# still needs to debug
import math
# based on c implementation of tree_plru
# https://github.com/gem5/gem5/blob/87c121fd954ea5a6e6b0760d693a2e744c2200de/src/mem/cache/replacement_policies/tree_plru_rp.cc
class tree_plru_policy(rep_policy):
    import math
    def __init__(self, associativity, block_size, verbose = False):
        self.associativity = associativity
        self.block_size = block_size
        self.num_leaves = associativity
        self.plrutree = [ False ] * ( self.num_leaves - 1 )
        self.count = 0
        self.candidate_tags = [ INVALID_TAG ] * self.num_leaves
        self.verbose = verbose

        self.vprint(self.plrutree)
        self.vprint(self.candidate_tags)
        #self.tree_instance = # holds the latest temporary tree instance created by 

    def parent_index(self,index):
        return math.floor((index - 1) / 2)

    def left_subtree_index(self,index):
        return 2 * index + 1

    def right_subtree_index(self,index):
        return 2 * index + 2

    def is_right_subtree(self, index):
        return index % 2 == 0

    def touch(self, tag, timestamp):
        # find the index
        tree_index = 0
        self.vprint(tree_index)
        while tree_index < len(self.candidate_tags):
            if self.candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        tree_index += ( self.num_leaves - 1)

        # set the path       
        right = self.is_right_subtree(tree_index)
        tree_index = self.parent_index(tree_index)
        self.plrutree[tree_index] = not right
        while tree_index != 0:
            right = self.is_right_subtree(tree_index)
            tree_index = self.parent_index(tree_index)
            #exit(-1)
            self.plrutree[tree_index] = not right
        self.vprint(self.plrutree)
        self.vprint(self.candidate_tags)

    def reset(self, tag, timestamp):
        self.touch(tag, timestamp)

    #def reset(self, tag):
    def invalidate(self, tag):
        # find index of tag
        self.vprint('invalidate  ' + tag)
        tree_index = 0
        while tree_index < len(self.candidate_tags):
            if self.candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        #print(tree_index)
        
        self.candidate_tags[tree_index] = INVALID_TAG
        tree_index += (self.num_leaves - 1 )
        
        # invalidate the path
        right = self.is_right_subtree(tree_index)
        tree_index = self.parent_index(tree_index)
        self.plrutree[tree_index] = right
        while tree_index != 0:
            right = self.is_right_subtree(tree_index)
            tree_index = self.parent_index(tree_index)
            self.plrutree[tree_index] = right

        self.vprint(self.plrutree)
        self.vprint(self.candidate_tags)

    def find_victim(self, timestamp):
        tree_index = 0
        while tree_index < len(self.plrutree): 
            if self.plrutree[tree_index] == 1:
                tree_index = self.right_subtree_index(tree_index)
            else:
                tree_index = self.left_subtree_index(tree_index)
            
        victim_tag = self.candidate_tags[tree_index - (self.num_leaves - 1) ]
        return victim_tag 

    # notice the usage of instantiate_entry() here is 
    # different from instantiateEntry() in gem5
    # in gem5 the function is only called during cache initialization
    # while here instantiate_entry is used when a line is evicted and new line is installed
    def instantiate_entry(self, tag, timestamp):
        # find a tag that can be invalidated

        index = 0
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == INVALID_TAG:
                break
            index += 1
        assert(self.candidate_tags[index] == INVALID_TAG) # this does not always hold for tree-plru
        self.candidate_tags[index] = tag

        # touch the entry
        self.touch(tag, timestamp)

class bit_plru(rep_policy):
    def __init__(self, associativity, block_size, verbose = False):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}
        self.verbose = verbose

    def touch(self, tag, timestamp):
        assert(tag in self.blocks)
        self.blocks[tag].last_accessed = 1

    def reset(self, tag, timestamp):
        return self.touch(tag, timestamp)

    def instantiate_entry(self, tag, timestamp):
        assert(tag not in self.blocks)
        timestamp = 1
        self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)

    #def reset(self, tag):
    def invalidate(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag]

    def find_victim(self, timestamp):
        in_cache = list(self.blocks.keys())
        victim_tag = in_cache[0] 
        found = False
        for b in in_cache:
            self.vprint(b + ' '+ str(self.blocks[b].last_accessed))
            # find the smallest last_accessed address 
            if self.blocks[b].last_accessed == 0:
                victim_tag = b
                found = True
                break
        
        if found == True:
            return victim_tag
        else:
            # reset all last_accessed to 0
            for b in in_cache:
                self.blocks[b].last_accessed = 0
            # find the leftmost tag
            for b in in_cache:            
                if self.blocks[b].last_accessed == 0:
                    victim_tag = b
                    break
            return victim_tag         
                

#pl cache option
PL_NOTSET = 0
PL_LOCK = 1
PL_UNLOCK = 2
class plru_pl_policy(rep_policy):
    def __init__(self, associativity, block_size, verbose = False):
        self.associativity = associativity
        self.block_size = block_size
        self.num_leaves = associativity
        self.plrutree = [ False ] * ( self.num_leaves - 1 )
        self.count = 0
        self.candidate_tags = [ INVALID_TAG ] * self.num_leaves
        self.lockarray = [ PL_UNLOCK ] * self.num_leaves
        self.verbose = verbose

        self.vprint(self.plrutree)
        self.vprint(self.lockarray)
        self.vprint(self.candidate_tags)
        #self.tree_instance = # holds the latest temporary tree instance created by 

    def parent_index(self,index):
        return math.floor((index - 1) / 2)

    def left_subtree_index(self,index):
        return 2 * index + 1

    def right_subtree_index(self,index):
        return 2 * index + 2

    def is_right_subtree(self, index):
        return index % 2 == 0

    def touch(self, tag, timestamp):
        # find the index
        tree_index = 0
        self.vprint(tree_index)
        while tree_index < len(self.candidate_tags):
            if self.candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        tree_index += ( self.num_leaves - 1)

        # set the path       
        right = self.is_right_subtree(tree_index)
        tree_index = self.parent_index(tree_index)
        self.plrutree[tree_index] = not right
        while tree_index != 0:
            right = self.is_right_subtree(tree_index)
            tree_index = self.parent_index(tree_index)
            #exit(-1)
            self.plrutree[tree_index] = not right
        self.vprint(self.plrutree)
        self.vprint(self.lockarray)
        self.vprint(self.candidate_tags)

    def reset(self, tag, timestamp):
        self.touch(tag, timestamp)

    #def reset(self, tag):
    def invalidate(self, tag):
        # find index of tag
        self.vprint('invalidate  ' + tag)
        tree_index = 0
        while tree_index < len(self.candidate_tags):
            if self.candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        #print(tree_index)
        self.candidate_tags[tree_index] = INVALID_TAG
        tree_index += (self.num_leaves - 1 )
        
        # invalidate the path
        right = self.is_right_subtree(tree_index)
        tree_index = self.parent_index(tree_index)
        self.plrutree[tree_index] = right
        while tree_index != 0:
            right = self.is_right_subtree(tree_index)
            tree_index = self.parent_index(tree_index)
            self.plrutree[tree_index] = right

        self.vprint(self.plrutree)
        self.vprint(self.lockarray) 
        self.vprint(self.candidate_tags)

    def find_victim(self, timestamp):
        tree_index = 0
        while tree_index < len(self.plrutree): 
            if self.plrutree[tree_index] == 1:
                tree_index = self.right_subtree_index(tree_index)
            else:
                tree_index = self.left_subtree_index(tree_index)
        index = tree_index - (self.num_leaves - 1) 
        
        # pl cache 
        if self.lockarray[index] == PL_UNLOCK:
            victim_tag = self.candidate_tags[index]
            return victim_tag 
        else:
            return INVALID_TAG

    # notice the usage of instantiate_entry() here is 
    # different from instantiateEntry() in gem5
    # in gem5 the function is only called during cache initialization
    # while here instantiate_entry is used when a line is evicted and new line is installed
    def instantiate_entry(self, tag, timestamp):
        # find a tag that can be invalidated
        index = 0
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == INVALID_TAG:
                break
            index += 1

        assert(self.candidate_tags[index] == INVALID_TAG)
        self.candidate_tags[index] = tag
        ###while index < self.num_leaves:
        ###    if self.candidate_tags[index] == INVALID:
        ###        self.candidate_tags[index] = tag  
        ###        break 
        ###    else:
        ###        index += 1     
        # touch the entry
        self.touch(tag, timestamp)

    # pl cache set lock scenario
    def setlock(self, tag, lock):
        self.vprint("setlock "+ tag + ' ' + str(lock))
        # find the index
        index = 0
        self.vprint(index)
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == tag:
                break
            else:
                index += 1
        # set / unset lock
        self.lockarray[index] = lock 

#implementation based on https://github.com/gem5/gem5/blob/87c121fd954ea5a6e6b0760d693a2e744c2200de/src/mem/cache/replacement_policies/brrip_rp.cc
# testcase based on https://dl.acm.org/doi/pdf/10.1145/1816038.1815971
class brrip_policy(rep_policy):
    def __init__(self, associativity, block_size, verbose = False):
        self.associativity = associativity
        self.block_size = block_size
        self.count = 0
        self.candidate_tags = [ INVALID_TAG ] * self.associativity
        self.verbose = verbose
        self.num_rrpv_bits = 2
        self.rrpv_max = int(math.pow(2, self.num_rrpv_bits)) - 1
        
        self.rrpv = [ self.rrpv_max ] * associativity
        self.hit_priority = False
        self.btp = 100

        self.vprint(self.candidate_tags)
        self.vprint(self.rrpv)
        #self.tree_instance = # holds the latest temporary tree instance created by 

    def instantiate_entry(self, tag, timestamp):
        # find a tag that can be invalidated
        index = 0
        while index < len(self.candidate_tags): 
            if self.candidate_tags[index] == INVALID_TAG:
                self.candidate_tags[index] = tag
                self.rrpv[index] = self.rrpv_max
                break
            index += 1
        # touch the entry
        self.touch(tag, timestamp, hit = False)

    def touch(self, tag, timestamp, hit = True):
        # find the index
        index = 0
        self.vprint(index)
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == tag:
                break
            else:
                index += 1
        if self.hit_priority == True:
            self.rrpv[index] = 0
        else:
            if self.rrpv[index] > 0:
                if hit == True:
                    self.rrpv[index] = 0
                else:
                    self.rrpv[index] -= 1
        self.vprint(self.candidate_tags)
        self.vprint(self.rrpv)

    def reset(self, tag, timestamp):
        index = 0
        self.vprint(index)
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == tag:
                break
            else:
                index += 1
        if random.randint(1,100) <= self.btp:
            if self.rrpv[index] > 0:
                self.rrpv[index] -= 1
        self.vprint(self.candidate_tags)
        self.vprint(self.rrpv)

    #def reset(self, tag):
    def invalidate(self, tag):
        # find index of tag
        self.vprint('invalidate  ' + tag)
        index = 0
        while index < len(self.candidate_tags):
            if self.candidate_tags[index] == tag:
                break
            else:
                index += 1
        #print(tree_index)        
        self.candidate_tags[index] = INVALID_TAG
        self.rrpv[index] = self.rrpv_max
        self.vprint(self.candidate_tags)
        self.vprint(self.rrpv)

    def find_victim(self, timestamp):
        max_index = 0
        index = 0
        while index < len(self.candidate_tags):
            if self.rrpv[index] > self.rrpv[max_index]:
                max_index = index
            index += 1
        # invalidate the path
        diff = self.rrpv_max - self.rrpv[max_index] 
        self.rrpv[max_index] = self.rrpv_max
        if diff > 0:
            index = 0
            while index < len(self.candidate_tags):
                self.rrpv[index] += diff
                index += 1
        #self.vprint(self.plrutree)
        #self.vprint(self.candidate_tags)
        self.vprint(self.candidate_tags)
        self.vprint(self.rrpv)

        return self.candidate_tags[max_index] 
    
LOCK = 1
UNLOCK = 0
class lru_lock_policy(rep_policy):
    
    def __init__(self, associativity, block_size, verbose=False):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}
        self.verbose = verbose
        self.lock_vector_array = [ UNLOCK ] * self.associativity # [0, 0, 0, 0]
        self.vprint(self.lock_vector_array)
        
    def touch(self, tag, timestamp): 
        assert(tag in self.blocks)
        self.blocks[tag].last_accessed = timestamp
        
    def reset(self, tag, timestamp): # touch 
        return self.touch(tag, timestamp)
    
    def instantiate_entry(self, tag, timestamp):# instantiate a new address in the cache
        #assert(tag == INVALID_TAG or tag not in self.blocks)
        #assert(len(self.blocks) < self.associativity)
        self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)
            
    def instantiate_entry_test(self, tag, timestamp, cache_set_data):
        pass
        # this function will inovoked only when len(in_cache) < self.associativity in Cache
        # 1. get the values of lock_vector_array
        # 2. search for the blocks with lock_vector is 0. 
        # 3. if all blocks are w/ lock_vector = 1, then cache miss.    
        # 4. if block(s) exist, instantiate a tag into 1st block
        # 5. this function should limit len(in_cache) less than self.associativity  

    
        '''
        #print(cache_set_data)
        locked_tag = tag
        print(locked_tag)
        i = 0
        for i in range(len(cache_set_data)):
            print(cache_set_data[i])
            print(self.lock_vector_array[i])
        # (tag, block.Block(self.block_size, current_step, False, address))
        #tag = INVALID_TAG
        #candidate_tags = []

        if self.lock_vector_array[i] == UNLOCK:
            self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)
            
        else:
            #TODO: what happens cache is locked and empty? if cache is empty and try to 
            #read it, it will get cache misss and write the address in there. if locked, 
            #even if empty the address can not be written there. 
            #so it should be regarded as read from memory
            
            #del self.blocks[tag]
            locked_tag = INVALID_TAG
            return locked_tag
            return self.reset(locked_tag, timestamp)
            #print(tag)
            #assert(False)
            #return self.find_victim(tag, timestamp, cache_set_data)
        #else:
        #    pass
        #    else:
        #        pass
                #return self.invalidate_unsafe(tag)
                #return INVALID_TAG
                #tag = INVALID_TAG
                #break
                #self.blocks[tag] = INVALID_TAG
        ''' 
    def invalidate(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag] # delete the oldest entry in the cache = delete the evicted entry
        
    def invalidate_unsafe(self, tag):
        if tag in self.blocks:
            del self.blocks[tag]
            
    def find_victim(self, tag, current_step, cache_set_data): 
        Is_evict = False
        victim_tag = INVALID_TAG
        candidate_tags = []
        for i in range(self.associativity):
            if self.lock_vector_array[i] == UNLOCK:
                candidate_tags.append(cache_set_data[i])

        if len(candidate_tags) == 0:
            Is_evict = False
            victim_tag = INVALID_TAG
                    
        else:
                    
            min_timestamp = min(self.blocks[item[0]].last_accessed for item in candidate_tags)
            victim_tag = [item[0] for item in candidate_tags if item[1].last_accessed == min_timestamp][0]
            Is_evict = True

        #print('victim_tag: ', victim_tag)
        return Is_evict, victim_tag
    
    # gathers lock vectors per line into the array
    def set_lock_vector(self, lock_vector_array): 
        
        for i in range(0, len(self.lock_vector_array)):
            self.lock_vector_array[i] = lock_vector_array[i]
            i = +i
            
        return self.lock_vector_array