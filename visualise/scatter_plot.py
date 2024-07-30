"""

Produce a scatterplot to visualize the results of the inference.
STEP 1. Prepare the data.
STEP 2. Make the plot.

"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""STEP 1. Prepare the data."""

# Read the csv file containing the data into a pandas dataframe
results_df = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results6/'
                         'Wind_results_hyper1000SNR.csv')
true_values = results_df["True_value"]
means = results_df["Mean"]
lower_bounds = results_df["Lower_bound"]
upper_bounds = results_df["Upper_bound"]

# Define the number of points
x = range(len(means))

# Calculate the lower and upper error values
lower_errors = np.array([means[i] - lower_bounds[i] for i in x])
upper_errors = np.array([upper_bounds[i] - means[i] for i in x])

"""STEP 2. Make the plot."""

# Create the figure
plt.figure(figsize=(6, 4))

# Plot the mean with error bars
plt.errorbar(x, means, yerr=[lower_errors, upper_errors],
             fmt='o', capsize=5, label='Mean and 95% confidence interval')

# Plot the true values
plt.scatter(x, true_values, color='red', label='True values')

# Define the ticks and labels of the x-axis
y_ticks = np.arange(0, 25, 5)
x_ticks = np.arange(0, 4, 1)
x_labels = [f'Site {x}' for x in x_ticks]
plt.xticks(x_ticks, labels=x_labels)
plt.yticks(y_ticks)

# Add labels and title
plt.xlabel('Sampling sites')
# plt.ylabel('Phytoplankton concentration (mg/$\mathregular{m^3}$)')
plt.ylabel('Wind speed (m/s)')
# plt.ylabel('Depth (m)')
plt.legend()

# Show the plot
plt.show()
