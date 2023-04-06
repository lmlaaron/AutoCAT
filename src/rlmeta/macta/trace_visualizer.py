import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def process_file(input_file):
    """read from saved output running sample_mutiagent_check.py
    i.e. python sample_multiagent_check.py > 631s4.txt """
    atk_data = []
    vic_data = []

    with open(input_file, 'r') as infile:

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

                # separate traces for different domain_id's
                if row[1] == 'a':
                    atk_data.append(row[0:3])
                elif row[1] == 'v':
                    vic_data.append(row[0:3])
            
    # Remove the last row from atk_data and vic_data
    if atk_data:
        atk_data.pop()
    if vic_data:
        vic_data.pop()
    return atk_data, vic_data


def step_offset(input_file):
    if input_file.endswith("2.txt"):
        return 1000000
    elif input_file.endswith("3.txt"):
        return 2000000
    elif input_file.endswith("4.txt"):
        return 3000000
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

        vic_data[i][0] = int(vic_data[i][0]) + offset # step no w/ offset
        vic_data[i][1] = 0  # if victim access: change to 0
        vic_data[i][2] = int(vic_data[i][2]) % 8

    return atk_data, vic_data
    
    
def draw_heatmap(data, title='Memory access patterns', output_file='graph.png'):
    x_coords = []
    y_coords = []

    for row in data:
        x_coord = row[0]
        y_coord = row[2]
        x_coords.append(x_coord)
        y_coords.append(y_coord)

    plt.figure(figsize=(9, 4))  # (width, height) (16, 4) for 10K steps

    hist, x_edges, y_edges, image = plt.hist2d(x_coords, y_coords, bins=[20, 8], cmap='Reds')
    plt.colorbar(label='No of cache accesses')
    plt.xlabel('Steps')
    plt.ylabel('Set_index')
    plt.title(title)
    plt.ylim(0, 7)
    
    # Define custom x-axis formatting
    def format_func(value, tick_number):
        #return f'{int(value / 1000)}K'
        return f'{int(value / 1)}'
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(output_file, dpi=400, format='png')
    

def main(input_file, atk_data=None, vic_data=None):
    atk_data, vic_data = process_file(input_file)
    offset = step_offset(input_file)
    atk_data, vic_data = update_data(atk_data, vic_data, offset)
    atk_data_np = np.array(atk_data, dtype=int)
    vic_data_np = np.array(vic_data, dtype=int)
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
        main(input_file)
    else:
        print("Please provide an input file name as an argument.")