# Author: Mulong Luo
# Date: 2021.9.21
# given the cache trace, find the histgoram of all the different 
# patterns of length to up to max_l
# input: filename of the cache tracefile such as trace1_1, trace2_1, trace2_2
# output: histogram
import numpy
import math
def motif_hist(filename, max_l, cache_size):
    data = numpy.loadtxt(filename)
    hist={}
    for i in range(max_l, len(data)):
      for window_size in range(1,max_l+1):
        temp = data[i-window_size:i].transpose()[0]
        key =""
        for c in temp:
            key += ' '
            key += str(int(c))
        print(key)
        # address >= cache_size means a receiver access
        # address < cache_size means a sender access
        if data[i][0] >= cache_size and data[i][1] == 1001: #receiver cache hiss
            temp= hist.get(key)
            if temp == None:
               hist[key] = (1,0)
            else:
               hist[key] = (temp[0]+1, temp[1])
        elif data[i][0] >= cache_size and data[i][1] == 1: #receiver cache hit 
            temp= hist.get(key)
            if temp == None:
               hist[key] = (0,1)
            else:
               hist[key] = (temp[0], temp[1]+1)
    return hist

# input: histogram
# output: sorted histogram by confidence
# the confidence is calculated using Normal approximation interval
# see this: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
# n = n_s + n_f
# score = n_s / n - 1.96 / (n \sqrt(n)) \sqrt(n_s * n_f)
def sort_hist(hist):
    sorted_hist = sorted(hist.items(), key = lambda x: (-( max(x[1][0], x[1][1]) * 1.0 / (x[1][0] + x[1][1]) - 1.96 / ( (x[1][0] + x[1][1])* math.sqrt(x[1][0] + x[1][1]))* math.sqrt(x[1][0]*x[1][1])), -x[1][0] - x[1][1] ))
    return sorted_hist