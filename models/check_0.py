"""

MODELS: Applying the inference scheme to simulated data.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Conduct inference on simulated data.
STEP 1. Read the simulated dataset.
STEP 2. Conduct inference on simulated data.

Last updated on 27 August 2024

"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

"""STEP 1. Read the simulated dataset."""

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('data/x_data/simulated_reflectance_100SNR.csv')

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('data/simulation_setup/Ecolight_parameter_combinations.csv')
ecolight_input = ecolight_input.drop(columns=["water"])  # Remove the "water" column.

# Define theta and x.
spectrum_id = 0
theta_example = ecolight_input.iloc[spectrum_id]  # Theta contains the five input variables
print(theta_example)
x_array = simulated_reflectance.iloc[spectrum_id]  # X contains the simulated spectra
print(x_array)

# Convert to tensors
theta_tensor = torch.tensor(theta_example, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

print("Shape of the theta tensor: ", theta_tensor.shape)
print("Shape of the x tensor: ", x_tensor.shape)

"""STEP 2. Conduct inference on simulated data."""

# Load the posterior
with open("data/loaded_posteriors/loaded_posterior_100SNR_hyper.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

results_path = "data/results/check0/100SNR/"


def infer_from_simulated_spectra(x_sim, x_sim_parameters):
    loaded_posterior.set_default_x(x_sim)
    posterior_samples = loaded_posterior.sample((10000,), x=x_sim)  # Sample from the posterior p(θ|x)

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame()  # Create a dataframe
    theta_means_df["Mean"] = theta_means  # Save the calculated values
    # theta_means_df.to_csv(results_path + str(spectrum_id) + '_theta_means.csv', index=False)

    # Plot a figure
    _ = pairplot(
        samples=posterior_samples,
        points=x_sim_parameters,
        limits=[[0, 0.2], [0, 1], [0, 0.4], [0, 20], [0, 15]],
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
    plt.show()


infer_from_simulated_spectra(x_tensor, theta_tensor)
