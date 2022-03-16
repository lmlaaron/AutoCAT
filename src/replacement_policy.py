import block
import random

# interface for cache replacement policy per set
class rep_policy:
    def touch(self, tag, timestamp):
        pass

    def reset(self, tag, timestamp):
        pass

    def invalidate(self, tag):
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
            print(b + ' '+ str(self.blocks[b].last_accessed))
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
'''
INVALID = '--------'
# based on c implementation of tree_plru
# https://github.com/gem5/gem5/blob/87c121fd954ea5a6e6b0760d693a2e744c2200de/src/mem/cache/replacement_policies/tree_plru_rp.cc
class tree_plru(rep_policy):
    import math
    def __init__(self, associativity, block_size):
        self.associativity = associativity
        self.block_size = block_size
        self.num_leaves = associativity
        self.plrutree = [ False ] * ( self.num_leaves - 1 )
        self.count = 0
        self.candidate_tags = [ INVALID ] * self.num_leaves
        #self.tree_instance = # holds the latest temporary tree instance created by 

    @staticmethod
    def parent_index(index):
        return math.floor((index - 1) / 2)

    @staticmethod
    def left_subtree_index(index):
        return 2 * index + 1

    @staticmethod
    def right_subtree_index(index):
        return 2 * index + 2

    @staticmethod
    def is_right_subtree(index):
        return index % 2 == 0

    def touch(self, tag, timestamp):
        # find the index
        tree_index = 0
        while tree_index < len(condidate_tags):
            if candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        # set the path        
        right = is_right_subtree(tree_index)
        tree_index = parent_index(tree_index)
        tree[tree_index] = not right
        while tree_index != 0:
            right = is_right_subtree(tree_index)
            tree_index = parent_index(tree_index)
            tree[tree_index] = not right

    def reset(self, tag, timestamp):
        self.touch(tag, timestamp)

    #def reset(self, tag):
    def invalidate(self, tag):
        # find index of tag
        tree_index = 0
        while tree_index < len(condidate_tags):
            if candidate_tags[tree_index] == tag:
                break
            else:
                tree_index += 1
        candidate_tags[tree_index] = INVALID
        # invalidate the path
        right = is_right_subtree(tree_index)
        tree_index = parent_index(tree_index)
        tree[tree_index] = right
        while tree_index != 0:
            right = is_right_subtree(tree_index)
            tree_index = parent_index(tree_index)
            tree[tree_index] = right

    def find_victim(self, timestamp):
        tree_index = 0
        while tree_index < len(plrutree): 
            if plru_tree[tree_index] == 1:
                tree_index = right_subtree_index(tree_index)
            else:
                tree_index = left_subtree_index(tree_index)
            
        victim_tag = candidate_tags[tree_index - (self.num_leaves - 1) ]
        return victim_tag 

    def instantiate_entry(self, tag, timestamp):
        # find a tag that can be invalidated

        # create replacement data using current tree instance
        self.candidate_tags[count ]

        # update instance counter
        self.count += 1


class brrip_policy(rep_policy):
class bit_plru(rep_policy):
class plru_rp_cache_policy(rep_policy):
'''