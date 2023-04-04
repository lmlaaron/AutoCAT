import matplotlib.pyplot as plt
import numpy as np

# Set up the x-axis and y-axis values
x_values = np.arange(0, 1000)
y_values = np.arange(0, 7)

# Initialize the figure and axis for the plot
fig, ax = plt.subplots()

# Randomly place one dot per step according to the x-axis value
for step in y_values:
    random_x = np.random.choice(x_values)
    ax.plot(random_x, step, 'bo')

# Set the axis labels
ax.set_xlabel('Steps')
ax.set_ylabel('Set index (0 to 7)') 

# Set the axis limits
ax.set_xlim(-1, 1000)
ax.set_ylim(-1, 7)

# Save the graph as a PNG file
plt.savefig('output_graph.png', format='png')

# Display the graph
plt.show()
