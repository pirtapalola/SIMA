"""

Produce a scatterplot to visualize the results of the inference.
STEP 1. Prepare the data.
STEP 2. Make the plot.

"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np


"""STEP 1. Prepare the data."""

true_values = [0.130, 0.400, 0.300, 0.500]  # Add your true values here
means = [0.001, 0.410, 0.320, 0.490]  # Add your mean values here
lower_bounds = [0.000, 0.300, 0.200, 0.450]  # Add your 2.5 percentile values here
upper_bounds = [0.011, 0.520, 0.440, 0.530]  # Add your 97.5 percentile values here

# Define the number of points
x = range(len(true_values))

# Calculate the lower and upper error values
lower_errors = np.array([means[i] - lower_bounds[i] for i in x])
upper_errors = np.array([upper_bounds[i] - means[i] for i in x])

"""STEP 2. Make the plot."""

# Create the figure
plt.figure(figsize=(10, 6))

# Plot the mean with error bars
plt.errorbar(x, means, yerr=[lower_errors, upper_errors],
             fmt='o', capsize=5, label='Mean and 95% confidence interval')

# Plot the true values
plt.scatter(x, true_values, color='red', label='True Values')

# Define the ticks and labels of the x-axis
x_ticks = np.arange(0, 4, 1)
x_labels = [f'Site {x}' for x in x_ticks]
plt.xticks(x_ticks, labels=x_labels)

# Add labels and title
plt.xlabel('Sampling sites')
plt.ylabel('Phytoplankton concentration (mg/$\mathregular{m^3}$)')
plt.legend()

# Show the plot
plt.show()
