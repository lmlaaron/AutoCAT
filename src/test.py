
trace_file = open('/home/geunbae/CacheSimulator/env_test/test1.txt')
trace = trace_file.read().splitlines()
nested_list = [list(map(int, x.split())) for x in trace]
actions = [{'attacker': values[0], 'benign': values[1], 'defender': values[2]} for values in nested_list]

print(actions)