#!/usr/bin/env python

import yaml, cache, argparse, logging, pprint
from terminaltables.other_tables import UnixTable
from cache_simulator import *

def this_main():
    #Set up our arguments
    parser = argparse.ArgumentParser(description='Simulate a cache')
    parser.add_argument('-c','--config-file', help='Configuration file for the memory heirarchy', required=True)
    parser.add_argument('-t', '--trace-file', help='Tracefile containing instructions', required=True)
    parser.add_argument('-l', '--log-file', help='Log file name', required=False)
    parser.add_argument('-p', '--pretty', help='Use pretty colors', required=False, action='store_true')
    parser.add_argument('-d', '--draw-cache', help='Draw cache layouts', required=False, action='store_true')
    arguments = vars(parser.parse_args())
    
    if arguments['pretty']:
        import colorer

    log_filename = 'cache_simulator.log'
    if arguments['log_file']:
        log_filename = arguments['log_file']

    #Clear the log file if it exists
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
    configs = yaml.load(config_file)
    hierarchy = build_hierarchy(configs, logger)
    logger.info('Memory hierarchy built.')

    logger.info('Loading tracefile...')
    trace_file = open(arguments['trace_file'])
    trace = trace_file.read().splitlines()
    trace = [item for item in trace if not item.startswith('#')]
    logger.info('Loaded tracefile ' + arguments['trace_file'])
    logger.info('Begin simulation!')
    
    #simulate(hierarchy, trace, logger)
    interactive_simulate(hierarchy, logger)


if __name__ == '__main__':
    this_main()

