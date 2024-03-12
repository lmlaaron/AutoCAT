#!/usr/bin/env python
"""above line ensures the used interpreter is the first one on your environment's $PATH
If there are several versions of Python installed"""

import yaml
import cache
import argparse
import logging
import pprint
from terminaltables.other_tables import UnixTable
from replacement_policy import *


def main():
    # Set up our arguments
    parser = argparse.ArgumentParser(description='Simulate a cache')
    parser.add_argument('-c', '--config-file', help='Configuration file for the memory hierarchy', required=True)
    parser.add_argument('-t', '--trace-file', help='Tracefile containing instructions', required=True)
    parser.add_argument('-l', '--log-file', help='Log file name', required=False)
    parser.add_argument('-p', '--pretty', help='Use pretty colors', required=False, action='store_true')
    parser.add_argument('-d', '--draw-cache', help='Draw cache layouts', required=False, action='store_true')
    parser.add_argument('-f', '--result-file', help='Result trace', required=False)
    arguments = vars(parser.parse_args())

    if arguments['pretty']:
        import colorer

    log_filename = 'cache_simulator.log'
    if arguments['log_file']:
        log_filename = arguments['log_file']

    result_file = 'result.txt'
    if arguments['result_file']:
        result_file = arguments['result_file']

    with open(result_file, 'w'):
        pass

    # Clear the log file if it exists
    with open(log_filename, 'w'):
        pass

    logger = logging.getLogger()
    fh = logging.FileHandler(log_filename)
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    fh_format = logging.Formatter('%(message)s')
    fh.setFormatter(fh_format)
    sh.setFormatter(fh_format)
    logger.setLevel(logging.INFO)
    logger.info('Loading config...')
    config_file = open(arguments['config_file'])
    configs = yaml.full_load(config_file)
    hierarchy = build_hierarchy(configs, logger)
    logger.info('Memory hierarchy built.')
    logger.info('Loading tracefile...')
    trace_file = open(arguments['trace_file'])
    trace = trace_file.read().splitlines()
    trace = [item for item in trace if not item.startswith('#')]
    logger.info('Loaded tracefile ' + arguments['trace_file'])
    logger.info('Begin simulation!')
    simulate(hierarchy, trace, logger, result_file=result_file)
    if arguments['draw_cache']:
        for cache in hierarchy:
            if hierarchy[cache].next_level:
                print_cache(hierarchy[cache])


def print_cache(cache):
    # Print the contents of a cache as a table
    # If the table is too long, it will print the first few sets,
    # break, and then print the last set
    table_size = 5
    ways = [""]
    sets = []
    set_indexes = sorted(cache.data.keys())
    
    if len(cache.data.keys()) > 0:
        way_no = 0

        # Label the columns
        for way in range(cache.associativity):
            ways.append("Way " + str(way_no))
            way_no += 1

        # Print either all the sets if the cache is small, or just a few
        # sets and then the last set
        sets.append(ways)
        if len(set_indexes) > table_size + 4 - 1:
            for s in range(min(table_size, len(set_indexes) - 4)):
                set_ways = cache.data[set_indexes[s]].keys()
                temp_way = ["Set " + str(s)]
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[s]][w].address)
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

            for i in range(3):
                temp_way = ['.']
                for w in range(cache.associativity):
                    temp_way.append('')
                sets.append(temp_way)

            set_ways = cache.data[set_indexes[len(set_indexes) - 1]].keys()
            temp_way = ['Set ' + str(len(set_indexes) - 1)]
            for w in range(0, cache.associativity):
                temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w][1].address)
                for w in set_ways:
                    temp_way.append(cache.data[set_indexes[len(set_indexes) - 1]][w].address)
            sets.append(temp_way)
            
        else:
            for s in range(len(set_indexes)):
                temp_way = ["Set " + str(s)]
                for w in range(0, cache.associativity):
                    temp_way.append(cache.data[set_indexes[s]][w][1].address)
                sets.append(temp_way)

                # add additional rows only if the replacement policy = lru_lock_policy
                if cache.rep_policy == lru_lock_policy:
                    lock_info = ["Lock bit"]

                    lock_vector_array = cache.set_rep_policy[set_indexes[s]].lock_vector_array

                    for w in range(0, len(lock_vector_array)):
                        lock_info.append(lock_vector_array[w])
                    sets.append(lock_info)

                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                    sets.append(timestamp)
                elif cache.rep_policy == new_plru_pl_policy: # add a new row to the table to show the lock bit in the plru_pl_policy cache
                    lock_info = ["Lock bit"]

                    lockarray = cache.set_rep_policy[set_indexes[s]].lockarray

                    for w in range(0, len(lockarray)):
                        if lockarray[w] == 2:
                            lock_info.append("unlocked")
                        elif lockarray[w] == 1:
                            lock_info.append("locked")
                        elif lockarray[w] == 0:
                            lock_info.append("unknown")
                        else:
                            lock_info.append(lockarray[w])
                    sets.append(lock_info)
                elif cache.rep_policy == lru_policy:  # or cache.rep_policy == lru_lock_policy:
                    timestamp = ["Timestamp"]
                    for w in range(0, cache.associativity):
                        if cache.data[set_indexes[s]][w][0] != INVALID_TAG:
                            timestamp.append(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                            print(cache.set_rep_policy[set_indexes[s]].blocks[cache.data[set_indexes[s]][w][0]].last_accessed)
                        else:
                            timestamp.append(0)
                            
                    sets.append(timestamp)
                    # print(timestamp)

        table = UnixTable(sets)
        table.title = cache.name
        table.inner_row_border = True
        print(table.table)
        return set_indexes


def simulate(hierarchy, trace, logger, result_file=''):
    # Loop through the instructions in the tracefile and use
    # the given memory hierarchy to find AMAT
    r = None
    responses = []
    if result_file != '':
        f = open(result_file, 'w')

    # We only interface directly with L1. Reads and writes will automatically
    # interact with lower levels of the hierarchy
    l1 = hierarchy['cache_1']

    if 'cache_1_core_2' in hierarchy:
        l1_c2 = hierarchy['cache_1_core_2']

    for current_step in range(len(trace)):
        instruction = trace[current_step]
        n_sets = 0
        inst2 = instruction.split()
        if len(inst2) == 2:
            address = inst2[0]
            op = inst2[1]

        else:
            n_sets = inst2[0]
            op = inst2[1]
            lock_bits = inst2[2:]

            # Call read for this address on our memory hierarchy
        if op == 'R' or op == 'R2':
            logger.info(str(current_step) + ':\tReading ' + address + ' op: ' + op)
            if op == 'R2':
                l = l1_c2
            else:
                l = l1
            r, _, moz = l.read(address, current_step)
            # print("check the returns of the read function:  ", r, "  ", moz)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)

        elif op == 'RL' or op == 'RL2':  # pl cache locks cache line, multicore not implemented
            assert (l1.rep_policy == plru_pl_policy)
            logger.info(str(current_step) + ':\tReading ' + address + ' ' + op)
            r, _ = l1.read(address, current_step, pl_opt=PL_LOCK)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)

        elif op == 'RU' or op == 'RU2':  # pl cache unlocks cache line
            assert (l1.rep_policy == plru_pl_policy)
            logger.info(str(current_step) + ':\tReading ' + address + ' ' + op)
            r, _ = l1.read(address, current_step, pl_opt=PL_UNLOCK)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)

        # Call write
        elif op == 'W' or op == 'W2':
            assert (op == 'W')
            logger.info(str(current_step) + ':\tWriting ' + address + ' ' + op)
            r, _ = l1.write(address, True, current_step)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)

        # Call cflush
        elif op == 'F' or op == 'F2':
            logger.info(str(current_step) + ':\tFlushing ' + address + ' ' + op)
            r, _, _, _ = l1.cflush(address, current_step)
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')

        # Call lock
        elif op == 'D':
            assert (l1.rep_policy == lru_lock_policy)
            
            for set_index in range(0, int(n_sets)):
                logger.info('current step: ' + str(current_step) + ' set ' + str(set_index) + ':\tLock_bit ' + str(
                    lock_bits[set_index]) + ' ' + 'op:' + op)
                r, _ = l1.lock(set_index, lock_bits[set_index])
            logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            responses.append(r)

        #insert detector into the pl cache code
        elif op == 'DD':
            assert (l1.rep_policy == new_plru_pl_policy)
            
            for set_index in range(0, int(n_sets)):
                # logger.info('current step: ' + str(current_step) + ' set ' + str(set_index) + ':\tLock_bit ' + str(
                #     lock_bits[set_index]) + ' ' + 'op:' + op)
                # r, _ = l1.lock(set_index, lock_bits[set_index])
                print("number of sets: ", int(n_sets), "  set index : ", set_index, "  lock bit : ", lock_bits[0][3])
            l1.detector_func(lock_bits[0])
            # logger.warning('\thit_list: ' + pprint.pformat(r.hit_list) + '\ttime: ' + str(r.time) + '\n')
            # responses.append(r)

        elif op == 'CH':
            assert (l1.rep_policy == new_plru_pl_policy)
            l1.check_func()

        else:
            raise InvalidOpError

        if op == 'D':
            print(str(r.time), file=f)
        
        elif op == 'DD':
            for cache in hierarchy:
                if hierarchy[cache].next_level:
                    print_cache(hierarchy[cache])
            continue
        
        else:
            print(address + ' ' + str(r.time), file=f)

        for cache in hierarchy:
            if hierarchy[cache].next_level:
                print_cache(hierarchy[cache])

    logger.info('Simulation complete')
    analyze_results(hierarchy, responses, logger)


def analyze_results(hierarchy, responses, logger):
    # Parse all the responses from the simulation
    n_instructions = len(responses)

    total_time = 0
    for r in responses:
        total_time += r.time
    logger.info('\nNumber of instructions: ' + str(n_instructions))
    logger.info('\nTotal cycles taken: ' + str(total_time) + '\n')

    amat = compute_amat(hierarchy['cache_1'], responses, logger)
    logger.info('\nAMATs:\n' + pprint.pformat(amat))


def compute_amat(level, responses, logger, results=None):
    # Check if this is the memory
    # Main memory has a non-variable hit time
    if results is None:
        results = {}
    if not level.next_level:
        results[level.name] = level.hit_time
    else:
        # Find out how many times this level of cache was accessed
        # And how many of those accesses were misses
        n_miss = 0
        n_access = 0
        for r in responses:
            if level.name in r.hit_list.keys():
                n_access += 1
                if not r.hit_list[level.name]:
                    n_miss += 1

        if n_access > 0:
            miss_rate = float(n_miss) / n_access
            # Recursively compute the AMAT of this level of cache by computing
            # the AMAT of lower levels
            results[level.name] = level.hit_time + miss_rate * compute_amat(level.next_level, responses, logger)[
                level.next_level.name]
        else:
            results[level.name] = 0 * compute_amat(level.next_level, responses, logger)[level.next_level.name]

        logger.info(level.name)
        logger.info('\tNumber of accesses: ' + str(n_access))
        logger.info('\tNumber of hits: ' + str(n_access - n_miss))
        logger.info('\tNumber of misses: ' + str(n_miss))
    return results


def build_hierarchy(configs, logger):
    hierarchy = {}
    # Main memory is required
    main_memory = build_cache(configs, 'mem', None, logger)
    prev_level = main_memory
    hierarchy['mem'] = main_memory
    if 'cache_3' in configs.keys():
        cache_3 = build_cache(configs, 'cache_3', prev_level, logger)
        prev_level = cache_3
        hierarchy['cache_3'] = cache_3
    if 'cache_2' in configs.keys():
        cache_2 = build_cache(configs, 'cache_2', prev_level, logger)
        prev_level = cache_2
        hierarchy['cache_2'] = cache_2
    if 'cache_1_core_2' in configs.keys():
        cache_1_core_2 = build_cache(configs, 'cache_1_core_2', prev_level, logger)
        prev_level = cache_2
        hierarchy['cache_1_core_2'] = cache_1_core_2
    # Cache_1 is required
    cache_1 = build_cache(configs, 'cache_1', prev_level, logger)
    if 'cache_1_core_2' in configs.keys():
        cache_1.add_same_level_cache(cache_1_core_2)
        cache_1_core_2.add_same_level_cache(cache_1)
    hierarchy['cache_1'] = cache_1

    return hierarchy


def build_cache(configs, name, next_level_cache, logger):
    return cache.Cache(name, configs['architecture']['word_size'],
                       configs['architecture']['block_size'],
                       configs[name]['blocks'] if (name != 'mem') else -1,
                       configs[name]['associativity'] if (name != 'mem') else -1,
                       configs[name]['hit_time'],
                       configs[name]['hit_time'],
                       configs['architecture']['write_back'],
                       logger,
                       next_level_cache,
                       rep_policy=configs[name]['rep_policy'] if 'rep_policy' in configs[name] else '',
                       prefetcher=configs[name]['prefetcher'] if 'prefetcher' in configs[name] else "none",
                       verbose=configs['verbose'] if 'verbose' in configs else 'False')


if __name__ == '__main__':
    main()
    