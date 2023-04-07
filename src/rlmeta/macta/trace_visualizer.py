import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from matplotlib.ticker import FuncFormatter


def process_file(input_file):
    atk_data = []
    vic_data = []

    with open(input_file, 'r') as infile:

        for line_num, line in enumerate(infile):
            if line_num < 11:  # Skip the first 10 rows
                continue

            if "victim address" in line:
                continue

            elif "Step" in line:
                line = line.replace("Step", "").strip()
                step = int(line)
                
                try:
                    line = str(step) + ' ' + next(infile).strip()
                except StopIteration:
                    break
                
                line = line.replace("victim access", "v")
                line = line.replace("access", "a")
                row = line.strip().split(' ')

                if len(row) < 3:  # Skip lines that don't have enough elements
                    print(f"Skipping line {line_num + 1}: {line}")
                    continue

                # separate traces for different domain_id's
                if row[1] == 'a':
                    atk_data.append(row[0:3])
                elif row[1] == 'v':
                    vic_data.append(row[0:3])

    if atk_data:
        atk_data.pop()
    if vic_data:
        vic_data.pop()
    return atk_data, vic_data


def step_offset(input_file):
    if input_file.endswith('MD.txt'):
        return 2000000
    elif input_file.endswith('RR.txt'):
        return 3600000
    elif input_file.endswith('FR.txt'):
        return 0
    else:
        return 0


def update_data(atk_data, vic_data, offset):
    """in numpy format [step, domain_id, set_no]
    as we're running benign traces, latency is not required"""
    conversion_dict = {'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15}

    for i in range(len(atk_data)):

        # Update to int form
        atk_data[i][0] = int(atk_data[i][0]) + offset # step no w/ offset

        # Update the domain id to binary int form
        atk_data[i][1] = 1  # if attacker access: change to 1

        # Update address to matching set_index 0 to 7 (8sets)
        if atk_data[i][2] in conversion_dict:
            atk_data[i][2] = conversion_dict[atk_data[i][2]] % 8
        else:
            atk_data[i][2] = int(atk_data[i][2]) % 8

    for i in range(len(vic_data)):

        vic_data[i][0] = int(vic_data[i][0]) + offset 
        vic_data[i][1] = 0  # if victim access: change to 0
        vic_data[i][2] = int(vic_data[i][2]) % 8

    return atk_data, vic_data


def get_first_n_rows(data, n):
    return data[:n] if len(data) >= n else data + [data[-1]] * (n - len(data))

    
def draw_heatmap(data, title='Memory access patterns', output_file='graph.png'):
    x_coords = []
    y_coords = []

    for row in data:
        x_coord = row[0]
        y_coord = row[2]
        x_coords.append(x_coord)
        y_coords.append(y_coord)

    plt.figure(figsize=(9, 4))  # (width, height) (16, 4) for 10K steps

    #hist, x_edges, y_edges, image = plt.hist2d(x_coords, y_coords, bins=[400, 8], cmap='Reds')  # for 10K steps
    hist, x_edges, y_edges, image = plt.hist2d(x_coords, y_coords, bins=[100, 8], cmap='Reds')  # for 100 steps 
    #plt.colorbar(label='No of cache accesses')
    cbar = plt.colorbar(label='No of cache accesses')
    # Update colorbar tick labels to integers
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
    
    plt.xlabel('Steps')
    plt.ylabel('Set_index')
    plt.title(title)
    plt.ylim(0, 7)
    
    # Define custom x-axis formatting
    def format_func(value, tick_number):
        #return f'{int(value / 1000)}K' # for 10K steps
        return f'{int(value / 1)}'  # for 100 steps
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(output_file, dpi=400, format='png', facecolor='none')
    

def main(input_file, max_rows=None, atk_data=None, vic_data=None):
    atk_data, vic_data = process_file(input_file)
    offset = step_offset(input_file)
    atk_data, vic_data = update_data(atk_data, vic_data, offset)
    
    atk_data_limited = get_first_n_rows(atk_data, max_rows)
    vic_data_limited = get_first_n_rows(vic_data, max_rows)

    atk_data_np = np.array(atk_data_limited, dtype=int)
    vic_data_np = np.array(vic_data_limited, dtype=int)
    
    print('atk_data_np')
    print(atk_data_np)
    print('vic_data_np')
    print(vic_data_np)

    atk_title = f"Benign A Memory access patterns ({input_file})"
    atk_output_file = f"atk_heatmap_{input_file.split('.')[0]}.png"

    vic_title = f"Benign B Memory access patterns ({input_file})"
    vic_output_file = f"vic_heatmap_{input_file.split('.')[0]}.png"

    draw_heatmap(atk_data_np, atk_title, atk_output_file)
    draw_heatmap(vic_data_np, vic_title, vic_output_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        max_rows = 100  # None  # reads to the end  # Set the maximum number of rows in the output. 
        main(input_file, int(max_rows / 2))
        #main(input_file, max_rows)
    else:
        print("Please provide an input file name as an argument.")