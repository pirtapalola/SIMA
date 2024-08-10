"""

Produce a range plot to visualize the results of the inference.
STEP 1. Prepare the data.
STEP 2. Make the plot.

Last updated on 10 August 2024 by Pirta Palola

"""


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""STEP 1. Prepare the data."""

# Read the csv file containing the data into a pandas dataframe
results_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/"
multi_results_SNR = "Summary_Multi/Multi_100SNR/"
hyper_results_SNR = "Summary_Hyper/100SNR/"
results_param = "Min"
multi_results_df = pd.read_csv(results_path + multi_results_SNR + results_param + ".csv")
hyper_results_df = pd.read_csv(results_path + hyper_results_SNR + results_param + ".csv")

multi_means = multi_results_df["Mean"]
multi_lower_bounds = multi_results_df["Lower_bound"]
multi_upper_bounds = multi_results_df["Upper_bound"]

hyper_means = hyper_results_df["Mean"]
hyper_lower_bounds = hyper_results_df["Lower_bound"]
hyper_upper_bounds = hyper_results_df["Upper_bound"]

# Define the number of points
x = range(len(multi_means))

# Calculate the lower and upper error values
multi_lower_errors = np.array([multi_means[i] - multi_lower_bounds[i] for i in x])
multi_upper_errors = np.array([multi_upper_bounds[i] - multi_means[i] for i in x])
hyper_lower_errors = np.array([hyper_means[i] - hyper_lower_bounds[i] for i in x])
hyper_upper_errors = np.array([hyper_upper_bounds[i] - hyper_means[i] for i in x])

spm_lower = multi_results_df["True_value_min"]
spm_upper = multi_results_df["True_value_max"]

# Define an offset to avoid overlapping of the data points
offset = 0.2

"""STEP 2. Make the plot."""

# Create the figure
plt.figure(figsize=(8, 6))

# Set font size
font = {'size': 12}
plt.rc('font', **font)

# Plot the results of the multispectral application with error bars
plt.errorbar(x - offset, multi_means, yerr=[multi_lower_errors, multi_upper_errors],
             fmt='o', capsize=5, label='Multispectral (mean and 95% confidence interval)',
             color='dodgerblue', alpha=0.8)

# Plot the true value range
plt.fill_between(x, spm_lower, spm_upper, color='red', alpha=0.3)

# Plot the results of the hyperspectral application with error bars
plt.errorbar(x + offset, hyper_means, yerr=[hyper_lower_errors, hyper_upper_errors],
             fmt='o', capsize=5, label='Hyperspectral (mean and 95% confidence interval)',
             color='#184e77')

# Define the ticks and labels of the x-axis
y_ticks = np.arange(0, 35, 5)
x_ticks = np.arange(0, len(multi_means), 1)
x_labels = [f'Site {x}' for x in x_ticks]
plt.xticks(x_ticks, labels=x_labels)
plt.yticks(y_ticks)

# Add labels and title
plt.xlabel('Sampling sites')
plt.ylabel('Mineral particle concentration (g/$\mathregular{m^3}$)')
# plt.legend()

# Save the plot
# plt.savefig(results_path + results_SNR + results_param + '.tiff')  # Save the figure as a tiff file
plt.show()  # Show the plot
