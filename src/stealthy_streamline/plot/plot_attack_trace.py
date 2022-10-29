#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# check that correct version is used for Python
assert sys.version_info[:2] == (2,7)


fontaxes = {
    'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 6,
}
fontaxes_title = {
    'family': 'Arial',
        'color':  'black',
        'weight': 'bold',
        'size': 9,
}

"""
    plot_attack_trace.py [inputFile] [Save?1:0] [if save, output file name]
    
    example:
    python ../../plot/plot_attack_trace.py sample_010101.txt 1 sample_010101

"""
inputFile_name_base=str(sys.argv[1])
save_figure=int(sys.argv[2])
if save_figure:
    outputFile_name_base=str(sys.argv[3])
    
for i in range(0,1):
    inputFile_name=inputFile_name_base
    if save_figure:
        outputFile_name=outputFile_name_base
    inputFile=open(inputFile_name,'r')

    t = []
    delay = []

    cnt=0
    offset=0
    for line in inputFile:
        cnt+=1
        if cnt>100:#skip first 100 lines
                t_tmp=int(line.split()[0])
                delay_tmp=int(line.split()[1])
                if t_tmp >3000+offset:
                    break;
                if t_tmp >=1000+offset:
                    t.append(t_tmp-1000-offset);
                    delay.append(delay_tmp);
                    

    plt.figure(num=None, figsize=(3.5, 1.5), dpi=300, facecolor='w')
    plt.subplots_adjust(right = 0.98, top = 0.98, bottom=0.35,left=0.32,wspace=0, hspace=0.2)  

    ax = plt.subplot(111)
    #fig, ax = plt.subplots()
    index = np.arange(200)
    bar_width = 0.35
    opacity = 0.8

    lsmarkersize = 2.5
    lslinewidth = 0.6

    Threshold=43

    plt.plot(t,delay,'.', linewidth=1, markersize=lsmarkersize, markeredgewidth=0)
    plt.plot([0,200], [Threshold,Threshold], linewidth=lslinewidth, color='r', linestyle=':');
    ax.set_xlim([0,200])
    ax.set_ylim([0,100])
    plt.xlabel("Receiver's Observation Sequence",fontdict = fontaxes)
    plt.ylabel('Latency (cycles)',fontdict = fontaxes)

    plt.tick_params(labelsize=6)

    plt.tight_layout()

    if save_figure:
        plt.savefig(outputFile_name+".pdf")
    else:
        plt.show()
        
