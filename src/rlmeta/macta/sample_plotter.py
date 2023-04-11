import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D


def process_file(input_file):
    data = []

    with open(input_file, 'r') as infile:
        # Skip the first 21 lines from yaml 
        for _ in range(21):
            next(infile)
            
        for line in infile:
            if "victim address" in line:
                continue
            
            elif "Step" in line:
                line = line.replace("Step", "").strip()
                step = int(line) 
                line = str(step) + ' ' + next(infile).strip()
                line = line.replace("victim access", "v")
                line = line.replace("access", "a")
                row = line.strip().split(' ')
                # Check if the row has enough elements before processing it
                if len(row) >= 3:
                    # separate traces for different domain_id's
                    if row[1] == 'a':
                        data.append(row[0:3])
                    elif row[1] == 'v':
                        data.append(row[0:3])
                else:
                    print("this line is not used: ", row)

    # Remove the last row from data
    if data:
        data.pop()
    print('input data: ', data[0:5])
    return data


def step_offset(input_file):
    if input_file.endswith('MD.txt'):
        return 2000000
    elif input_file.endswith('RR.txt'):
        return 3600000
    elif input_file.endswith('FR.txt'):
        return 0
    else:
        return 0


def update_data(data, offset):
    """in numpy format [step, domain_id, set_no]
    as we're running benign traces, latency is not required"""
    conversion_dict = {'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15}

    for i in range(len(data)):
        data[i][0] = int(data[i][0]) + offset # step no w/ offset
        
        if data[i][1] == "a":
            data[i][1] = 1 
            
        elif data[i][1] == "v":
            data[i][1] = 0          
        
        # Update address to matching set_index 0 to 7 (8sets)
        if data[i][2] in conversion_dict:
            data[i][2] = conversion_dict[data[i][2]] % 8
        else:
            data[i][2] = int(data[i][2]) % 8
    print('converted data: ', data[0:5])
    return data


def get_first_n_rows(data, n):
    return data[:n] if len(data) >= n else data + [data[-1]] * (n - len(data))


def calculate_correlation(attacker_data, victim_data):
    attacker_set_indices = attacker_data[:, 2]
    victim_set_indices = victim_data[:, 2]

    # Truncate the longer array to match the length of the shorter array
    if len(attacker_set_indices) > len(victim_set_indices):
        attacker_set_indices = attacker_set_indices[:len(victim_set_indices)]
    elif len(attacker_set_indices) < len(victim_set_indices):
        victim_set_indices = victim_set_indices[:len(attacker_set_indices)]

    corr_coef = np.corrcoef(attacker_set_indices, victim_set_indices)[0, 1]
    return corr_coef



def draw_heatmap(data_np, title='Memory access patterns', output_file='graph.png'):
    fig, ax = plt.subplots(figsize=(16, 24))  # (width, height)

    attacker_data = data_np[data_np[:, 1] == 1]
    victim_data = data_np[data_np[:, 1] == 0]

    bins_x = np.linspace(min(data_np[:, 0]), max(data_np[:, 0]), 201)
    #bins_y = np.linspace(min(data_np[:, 2]), max(data_np[:, 2]), 5)  # default is 8 
    bins_y = np.linspace(0, 8, 9)  # Create 8 bins spanning from 0 to 8

    hist_atk, x_edges_atk, y_edges_atk = np.histogram2d(attacker_data[:, 0], 
                                                        attacker_data[:, 2], 
                                                        bins=[bins_x, bins_y],
                                                    )
    hist_vic, x_edges_vic, y_edges_vic = np.histogram2d(victim_data[:, 0], 
                                                        victim_data[:, 2], 
                                                        bins=[bins_x, bins_y],
                                                    )
    
    hist_combined = hist_vic + hist_atk

    color_map = plt.cm.get_cmap('Blues')
    color_map.set_under(color='none')
    im_vic = ax.imshow(hist_vic.T, origin='lower', cmap=color_map, 
                       extent=[x_edges_vic[0], x_edges_vic[-1], y_edges_vic[0], y_edges_vic[-1]], 
                       vmin=0.1)
    
    color_map = plt.cm.get_cmap('Reds')
    color_map.set_under(color='none')
    im_atk = ax.imshow(hist_atk.T, origin='lower', 
                       cmap=color_map, extent=[x_edges_atk[0], x_edges_atk[-1], y_edges_atk[0], y_edges_atk[-1]], 
                       vmin=0.1)
    
    corr_coef = calculate_correlation(attacker_data, victim_data)

    legend_elements = [Line2D([0], [0], marker='s', color='w', label='Domain A',
                              markerfacecolor='darkred', markersize=6, alpha=0.9),
                       Line2D([0], [0], marker='s', color='w', label='Domain B',
                              markerfacecolor='darkblue', markersize=6, alpha=0.9),
                       Line2D([], [], color='none', label=f'Corr. Coef.: {corr_coef:.3f}')]

    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Set_index')
    ax.set_title(title)
    ax.set_ylim(0, 8)  # matches to 8-set
    ax.set_aspect(4)

    def format_func(value, tick_number):
        return f'{int(value / 1)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.tight_layout()
    plt.savefig(output_file, dpi=400, format='png', facecolor='white', bbox_inches='tight')  # white # none
    plt.show()

    

def main(input_file, max_rows=None, data=None):
    data = process_file(input_file)
    offset = step_offset(input_file)
    updated_data = update_data(data, offset)
    data_limited = get_first_n_rows(updated_data, max_rows)

    # Convert data_limited list to a NumPy array
    data_limited_np = np.array(data_limited, dtype=int)

    title = f"Memory access patterns ({input_file})"
    output_file = f"colormap_{input_file.split('.')[0]}.png"
    draw_heatmap(data_limited_np, title, output_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        max_rows = 400  
        main(input_file, int(max_rows / 2)) # use when specify max_rows
        #main(input_file, max_rows)
    else:
        print("Please provide an input file name as an argument.")

        
