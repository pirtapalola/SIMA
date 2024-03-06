"""

Conduct inference on simulated data.

Last updated on 5 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
import torch
import matplotlib.pyplot as plt
import pickle

"""STEP 1."""

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/check0/check0_x.csv')

# Read the csv file containing the inputs of each of the HydroLight simulation runs
ecolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                             'Methods/Methods_Ecolight/Jan2024_lognormal_priors/check0/check0_theta.csv')

# Define theta and x.
theta_dataframe = ecolight_input.iloc[20]  # Theta contains the five input variables.
x_dataframe = simulated_reflectance.iloc[20]  # X contains the simulated spectra.

# Convert the pandas DataFrames to numpy arrays
theta_array = theta_dataframe.to_numpy()
x_array = x_dataframe.to_numpy()
print(theta_dataframe)
print(x_dataframe)

# Convert the numpy arrays to PyTorch tensors
theta_tensor = torch.tensor(theta_dataframe, dtype=torch.float32)
x_tensor = torch.tensor(x_dataframe, dtype=torch.float32)
print(theta_tensor)
print(x_tensor)

"""STEP 2. Conduct inference on simulated data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/loaded_posterior0.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)


results_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
               'Jan2024_lognormal_priors/check0/'


def infer_from_simulated_spectra(x_sim, x_sim_parameters):
    posterior_samples = loaded_posterior.sample((1000,), x=x_sim)  # Sample from the posterior p(θ|x)
    # Create the figure
    _ = pairplot(
        samples=posterior_samples,
        points=x_sim_parameters,
        limits=[[0, 7], [0, 2.5], [0, 30], [0, 20], [0, 20]],
        points_colors=["red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.show()


infer_from_simulated_spectra(x_tensor, theta_tensor)
