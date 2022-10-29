#!/usr/bin/env python
import matplotlib.pyplot as plt


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


lsmarkersize = 2.5
lslinewidth = 0.6

Error_rate_stram=[[0.2177733333, 0.04370133333, 0.01709, 0.007975, 0.005696666667], [0.227539, 0.046631,0.022217,0.009277,0.006592],[0.210693, 0.041016,0.013916,0.007324,0.00415]]       
Error_rate_LRU=[[0.1423338333,0.02587883333,0.003662,0.004801666667,0.0013835],[0.583496, 0.054199, 0.006836, 0.008789, 0.005371],[0.01416, 0.009766, 0.001465, 0.001465,0]]

for i in range(3):
    for j in range(5):
        Error_rate_stram[i][j] = Error_rate_stram[i][j]*100
        Error_rate_LRU[i][j] = Error_rate_LRU[i][j]*100

bit_rate_stream=[113.78, 56.89, 37.92666667, 28.445, 22.756]
bit_rate_LRU=[68.267,34.1335,22.75566667,17.06675,13.6534]

plt.figure(num=None, figsize=(3.5, 1.5), dpi=300, facecolor='w')
plt.subplots_adjust(right = 0.98, top =0.97, bottom=0.21,left=0.12,wspace=0, hspace=0.2)  
    
    
plt.plot(Error_rate_stram[0], bit_rate_stream,'b.-', linewidth=1, markersize=lsmarkersize, markeredgewidth=0, label="Stealthy Streamline")

#error bar
bar_len_y=2
for i in range(5):
    plt.plot([Error_rate_stram[2][i],Error_rate_stram[1][i]],[bit_rate_stream[i], bit_rate_stream[i]], "b-", linewidth=0.5)
    plt.plot([Error_rate_stram[2][i],Error_rate_stram[2][i]],[bit_rate_stream[i]-bar_len_y, bit_rate_stream[i]+bar_len_y], "b-", linewidth=0.5)
    plt.plot([Error_rate_stram[1][i],Error_rate_stram[1][i]],[bit_rate_stream[i]-bar_len_y, bit_rate_stream[i]+bar_len_y], "b-", linewidth=0.5)
    
    
plt.plot(Error_rate_LRU[0], bit_rate_LRU,'go-', linewidth=1, markersize=lsmarkersize, markeredgewidth=0, label="LRU addr_based")

for i in range(5):
    plt.plot([Error_rate_LRU[2][i],Error_rate_LRU[1][i]],[bit_rate_LRU[i], bit_rate_LRU[i]], "g-", linewidth=0.5 )
    plt.plot([Error_rate_LRU[2][i],Error_rate_LRU[2][i]],[bit_rate_LRU[i]-bar_len_y, bit_rate_LRU[i]+bar_len_y], "g-", linewidth=0.5)
    plt.plot([Error_rate_LRU[1][i],Error_rate_LRU[1][i]],[bit_rate_LRU[i]-bar_len_y, bit_rate_LRU[i]+bar_len_y], "g-", linewidth=0.5)
    
   
#plt.title('Hor. symmetric')

ax = plt.subplot(111)
ax.set_xlim([0,25])
ax.set_ylim([0,120])
plt.xlabel("Error rate (%)",fontdict = fontaxes)
plt.ylabel('Bit Rate (Mbps)',fontdict = fontaxes)

plt.tick_params(labelsize=6)

#plt.tight_layout()
ax.legend(prop={'size': 6})
#plt.show()
plt.savefig('stealthy_streamline_error.pdf')  