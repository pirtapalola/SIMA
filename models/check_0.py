"""
Conduct inference on simulated data.

Last updated on 4 April 2024 by Pirta Palola
"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

"""STEP 1."""

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/' 
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                                    'simulated_reflectance_1000SNR_noise.csv')
print(simulated_reflectance)

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                             'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                             'Ecolight_parameter_combinations.csv')
ecolight_input = ecolight_input.drop(columns=["water"])  # Remove the "water" column.
print(ecolight_input)

# Define theta and x.
spectrum_id = 22
theta_example = ecolight_input.iloc[spectrum_id]  # Theta contains the five input variables

print(theta_example)
constant = 1.0  # Add a constant to avoid issues with the log-transformation of small values
theta_example[:3] += constant  # Only add the constant to the first 3 theta parameters
for x in range(4):  # Apply the log-transformation to the first 4 theta parameters
    theta_example[x] = np.log(theta_example[x])

x_array = simulated_reflectance.iloc[spectrum_id]  # X contains the simulated spectra

# Convert to tensors
theta_tensor = torch.tensor(theta_example, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

print("Shape of the theta tensor: ", theta_tensor.shape)
print("Shape of the x tensor: ", x_tensor.shape)

"""STEP 2. Conduct inference on simulated data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/Noise_1000SNR/loaded_posteriors/loaded_posterior11.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

results_path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Jan2024_lognormal_priors/" \
               "Noise_1000SNR/check0_model11/"


def infer_from_simulated_spectra(x_sim, x_sim_parameters):
    loaded_posterior.set_default_x(x_sim)
    posterior_samples = loaded_posterior.sample((10000,), x=x_sim)  # Sample from the posterior p(θ|x)

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_exp = theta_means
    for i in range(4):  # Apply an exponential transformation to the first 4 theta parameters
        theta_exp[i] = np.exp(theta_means[i])
    for i in range(3):  # Remove the constant from the first 3 theta parameters
        theta_exp[i] = theta_exp[i] - constant
    theta_means_df = pd.DataFrame()  # Create a dataframe
    theta_means_df["Mean"] = theta_exp  # Save the calculated values
    theta_means_df.to_csv(results_path + str(spectrum_id) + '_theta_means.csv', index=False)

    # Plot a figure
    _ = pairplot(
        samples=posterior_samples,
        points=x_sim_parameters,
        limits=[[0, 0.25], [0, 0.5], [0, 0.5], [0, 3], [0, 20]],
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.savefig(results_path + str(spectrum_id) + '.png')


infer_from_simulated_spectra(x_tensor, theta_tensor)
