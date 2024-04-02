"""
Conduct inference on simulated data.

Last updated on 29 March 2024 by Pirta Palola
"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

"""STEP 1."""

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/' 
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                                    'simulated_reflectance_with_noise_5percent.csv')
print(simulated_reflectance)

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                             'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                             'Ecolight_parameter_combinations.csv')

print(ecolight_input)

# Define theta and x.
spectrum_id = 50
theta_example = ecolight_input.iloc[spectrum_id]  # Theta contains the five input variables

print(theta_example)
constant = 1.0  # Add a constant to avoid issues with the log-transformation of small values
theta_example[:3] += constant  # Only add the constant to the first 3 theta parameters
for x in range(4):  # Apply the log-transformation to the first 4 theta parameters
    theta_example[x] = np.log(theta_example[x])

x_array = simulated_reflectance.iloc[spectrum_id]  # X contains the simulated spectra
print(x_array)

# Convert to tensors
theta_tensor = torch.tensor(theta_example, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

"""STEP 2. Conduct inference on simulated data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/noise_5percent/loaded_posteriors/loaded_posterior1.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

results_path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Jan2024_lognormal_priors/" \
               "noise_5percent/check0/"


def infer_from_simulated_spectra(x_sim, x_sim_parameters):
    posterior_samples = loaded_posterior.sample((10000,), x=x_sim)  # Sample from the posterior p(θ|x)

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame(theta_means)  # Convert to a dataframe
    theta_means_df.to_csv(results_path + str(spectrum_id) + '_theta_means.csv')

    # Plot a figure
    _ = pairplot(
        samples=posterior_samples,
        points=x_sim_parameters,
        limits=[[0, 5], [0, 5], [0, 10], [0, 20], [0, 20]],
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.show()


infer_from_simulated_spectra(x_tensor, theta_tensor)
