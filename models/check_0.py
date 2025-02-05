"""

MODELS: Applying the inference scheme to simulated data.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Conduct inference on simulated data.
STEP 1. Read the simulated dataset.
STEP 2. Conduct inference on simulated data.

Last updated on 5 February 2025

"""

# Import libraries
import pandas as pd
import torch
import pickle
import numpy as np
# from sbi.analysis import pairplot
# import matplotlib.pyplot as plt
# import seaborn as sns

"""STEP 1. Read the simulated dataset."""

specs = "_500SNR_multi"

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('data/x_data/multi_500SNR_evaluate.csv')

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('data/simulation_setup/Ecolight_parameter_combinations_evaluate.csv')
ecolight_input = ecolight_input.drop(columns=["unique_ID", "water"])  # Remove the "water" column.

"""STEP 2. Conduct inference on simulated data."""

# Load the posterior
with open("data/loaded_posteriors/loaded_posterior" + specs + ".pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

results_path = "data/results/check0/"


def infer_from_simulated_spectra(x_sim):  # Add x_sim_parameters as an argument if plotting
    loaded_posterior.set_default_x(x_sim)
    posterior_samples = loaded_posterior.sample((10000,), x=x_sim)  # Sample from the posterior p(θ|x)

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame()  # Create a dataframe
    theta_means_df["Mean"] = theta_means  # Save the calculated values
    # theta_means_df.to_csv(results_path + str(spectrum_id) + '_theta_means.csv', index=False)

    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(posterior_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    interval1 = theta_intervals_df.iloc[0]
    interval2 = theta_intervals_df.iloc[1]
    interval_width = interval2-interval1  # Calculate the width of the 95% confidence interval
    # results_df = pd.DataFrame()  # Create an empty dataframe
    # results_df["interval_width"] = interval_width  # Save the calculated value in the dataframe
    # results_df.to_csv(results_path + str(spectrum_id) + specs + 'CI_width.csv', index=False)

    """
    # Plot a figure
    _ = pairplot(
        samples=posterior_samples,
        points=x_sim_parameters,
        limits=[[0, 10], [0, 2.5], [0, 10], [0, 20], [0, 15]],
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankton (mg/$\mathregular{m^3}$)",
                "CDOM ($\mathregular{m^-1}$ at 440 nm)",
                "Mineral particles (g/$\mathregular{m^3}$)",
                "Wind (m/s)",
                "Depth (m)"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.tight_layout(pad=1.0)  # Adjust layout to make more space
    # plt.savefig(results_path + str(spectrum_id) + '.png')
    plt.show()"""

    return interval_width


"""
# Test on one spectrum
spectrum_id = 10
x_array = simulated_reflectance.iloc[spectrum_id]  # X contains the simulated spectra
x_tensor = torch.tensor(x_array, dtype=torch.float32)  # Convert to tensor
theta_example = ecolight_input.iloc[spectrum_id]  # Theta contains the five input variables
theta_tensor = torch.tensor(theta_example, dtype=torch.float32)
infer_from_simulated_spectra(x_tensor, theta_tensor)  # Convert to tensor
"""

# Create an empty list to store the results
interval_widths = []

# Loop through the entire dataset
for spectrum_id in range(0, 1000):
    x_array = simulated_reflectance.iloc[spectrum_id]  # X contains the simulated spectra
    x_tensor = torch.tensor(x_array, dtype=torch.float32)  # Convert to tensor
    # print("Shape of the theta tensor: ", theta_tensor.shape)
    # print("Shape of the x tensor: ", x_tensor.shape)
    interval_i = infer_from_simulated_spectra(x_tensor)
    interval_widths.append(interval_i)  # Append the result of each iteration

# Convert list to numpy array for easy computation
interval_widths = np.array(interval_widths)

# Compute mean interval width for each parameter separately
mean_interval_widths = np.mean(interval_widths, axis=0)
print("Mean Interval Widths for each parameter:", mean_interval_widths)
