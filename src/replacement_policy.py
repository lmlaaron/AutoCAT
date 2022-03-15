import block
import random

# interface for cache replacement policy per set
class rep_policy:
    def touch(self, tag, timestamp):
        pass

    def reset(self, tag):
        pass

    def find_victim(self, timestamp):
        pass

# LRU policy
class lru_policy(rep_policy):
    def __init__(self, associativity, block_size):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}

    def touch(self, tag, timestamp):
        if tag in self.blocks:
            self.blocks[tag].last_accessesd = timestamp
        else:
            self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)

    def reset(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag]

    def find_victim(self, timestamp):
        in_cache = list(self.blocks.keys())
        victim_tag = in_cache[0] 
        for b in in_cache:
            if self.blocks[b].last_accessed < self.blocks[victim_tag].last_accessed:
                victim_tag = b
        return victim_tag

# random replacement policy
class rand_policy(rep_policy):
    def __init__(self, associativity, block_size):
        self.associativity = associativity
        self.block_size = block_size
        self.blocks = {}

    def touch(self, tag, timestamp):
        if tag not in self.blocks:
            self.blocks[tag] = block.Block(self.block_size, timestamp, False, 0)

    def reset(self, tag):
        assert(tag in self.blocks)
        del self.blocks[tag]

    def find_victim(self, timestamp):
        in_cache = list(self.blocks.keys())
        index = random.randint(0,len(in_cache)-1)
        victim_tag = in_cache[index] 
        return victim_tag