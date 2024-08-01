"""

Produce a range plot to visualize the results of the inference.
STEP 1. Prepare the data.
STEP 2. Make the plot.

Last updated on 1 August 2024 by Pirta Palola

"""


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""STEP 1. Prepare the data."""

# Read the csv file containing the data into a pandas dataframe
results_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/Summary_Hyper/"
results_SNR = "50SNR/"
results_param = "Min"
results_df = pd.read_csv(results_path + results_SNR + results_param + ".csv")

means = results_df["Mean"]
lower_bounds = results_df["Lower_bound"]
upper_bounds = results_df["Upper_bound"]

# Define the number of points
x = range(len(means))

# Calculate the lower and upper error values
lower_errors = np.array([means[i] - lower_bounds[i] for i in x])
upper_errors = np.array([upper_bounds[i] - means[i] for i in x])

spm_lower = results_df["True_value_min"]
spm_upper = results_df["True_value_max"]

"""STEP 2. Make the plot."""

# Create the figure
plt.figure(figsize=(6, 4))

# Plot the mean with error bars
plt.errorbar(x, means, yerr=[lower_errors, upper_errors],
             fmt='o', capsize=5, label='Mean and 95% confidence interval', color='blue')

# Plot the true value range
plt.fill_between(x, spm_lower, spm_upper, color='red', alpha=0.3, label='True value range')

# Define the ticks and labels of the x-axis
y_ticks = np.arange(0, 35, 5)
x_ticks = np.arange(0, len(means), 1)
x_labels = [f'Site {x}' for x in x_ticks]
plt.xticks(x_ticks, labels=x_labels)
plt.yticks(y_ticks)

# Add labels and title
plt.xlabel('Sampling sites')
plt.ylabel('Mineral particle concentration (g/$\mathregular{m^3}$)')
plt.legend()

# Save the plot
# plt.savefig(results_path + results_SNR + results_param + '.tiff')  # Save the figure as a tiff file
# plt.show()  # Show the plot


def calculate_distance(lower_bound1, upper_bound1, lower_bound2, upper_bound2):
    # Check if the ranges overlap
    if max(lower_bound1, lower_bound2) <= min(upper_bound1, upper_bound2):
        return 0  # The ranges overlap

    # Calculate the distance if they do not overlap
    if upper_bound1 < lower_bound2:
        distance = lower_bound2 - upper_bound1
    else:  # upper_bound2 < lower_bound1
        distance = lower_bound1 - upper_bound2

    return distance


# Loop through all the datapoints
distance_list = []
for lower, upper, lower_spm, upper_spm in lower_bounds, upper_bounds, spm_lower, spm_upper:
    dist = calculate_distance(lower, upper, lower_spm, upper_spm)
    distance_list.append(dist)
distances_df = pd.DataFrame(distance_list, columns=[results_param])  # Save into a dataframe
distances_df.to_csv(results_path + results_SNR + results_param + '_CI_distance.csv', index=False)  # Save as a csv file
