"""

PLOTTING II: Producing a scatter plot to visualise the results of the inference
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

STEP 1. Prepare the data.
STEP 2. Make the plot.

Last updated on 27 August 2024

"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""STEP 1. Prepare the data."""

# Specify folder paths
base_path = "data/results/"
multi_results_SNR = "summary_multi/100SNR/"
hyper_results_SNR = "summary_hyper/100SNR/"

# Specify which parameter to access
results_param = "Depth"

# Read the csv files
multi_results_df = pd.read_csv(base_path + multi_results_SNR + results_param + ".csv")
hyper_results_df = pd.read_csv(base_path + hyper_results_SNR + results_param + ".csv")

# Save the results of the multispectral application as lists
multi_true_values = multi_results_df["True_value"]
multi_means = multi_results_df["Mean"]
multi_lower_bounds = multi_results_df["Lower_bound"]
multi_upper_bounds = multi_results_df["Upper_bound"]

# Save the results of the hyperspectral application as lists
hyper_true_values = hyper_results_df["True_value"]
hyper_means = hyper_results_df["Mean"]
hyper_lower_bounds = hyper_results_df["Lower_bound"]
hyper_upper_bounds = hyper_results_df["Upper_bound"]

# Define the number of points
x = np.arange(len(multi_means))

# Calculate the lower and upper error values
multi_lower_errors = np.array([multi_means[i] - multi_lower_bounds[i] for i in x])
multi_upper_errors = np.array([multi_upper_bounds[i] - multi_means[i] for i in x])
hyper_lower_errors = np.array([hyper_means[i] - hyper_lower_bounds[i] for i in x])
hyper_upper_errors = np.array([hyper_upper_bounds[i] - hyper_means[i] for i in x])

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

# Plot the true (field-measured) values for the multispectral application
plt.scatter(x - offset, multi_true_values, color='red', marker='x')

# Plot the results of the hyperspectral application with error bars
plt.errorbar(x + offset, hyper_means, yerr=[hyper_lower_errors, hyper_upper_errors],
             fmt='o', capsize=5, label='Hyperspectral (mean and 95% confidence interval)',
             color='#184e77')

# Plot the true (field-measured) values for the hyperspectral application
plt.scatter(x + offset, hyper_true_values, color='red', marker='x', label='Field-measured values')

# Define the ticks and labels of the x-axis
y_ticks = np.arange(0, 25, 5)
x_ticks = np.arange(0, 4, 1)
x_labels = [f'Site {x}' for x in x_ticks]
plt.xticks(x_ticks, labels=x_labels)
plt.yticks(y_ticks)

# Add labels and title
plt.xlabel('Sampling sites')
# plt.ylabel('Phytoplankton concentration (mg/$\mathregular{m^3}$)')
# plt.ylabel('Wind speed (m/s)')
plt.ylabel('Depth (m)')
# plt.legend(loc="upper left")

# Save the plot
# plt.savefig(results_path + results_SNR + results_param + '.tiff')  # Save the figure as a tiff file
plt.show()  # Show the plot
