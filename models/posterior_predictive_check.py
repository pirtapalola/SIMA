"""

Posterior predictive check
STEP 1. Draw samples from the posterior.
STEP 2. Save the samples into a csv file.

Last updated on 8 May 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import pickle
import torch
from sbi.analysis import pairplot

"""STEP 1. Draw samples from the posterior."""

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Noise_1000SNR/Noise_1000SNR/"
          "loaded_posteriors/loaded_posterior17.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)


# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/'
                                    'simulated_reflectance_1000SNR_noise_sbc.csv')
x_o = simulated_reflectance.iloc[0]
loaded_posterior.set_default_x(x_o)

# Draw theta samples from the posterior
posterior_samples = loaded_posterior.sample((5000,))

# Print the posterior samples
print(posterior_samples)

# Plot the posterior samples
_ = pairplot(
    samples=posterior_samples,
    limits=[[0, 7], [0, 2.5], [0, 30], [0, 20], [0, 20]],
    points_colors=["red", "red", "red", "red", "red"],
    figsize=(8, 8),
    labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
    offdiag="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=20)
)

posterior_samples = pd.DataFrame(posterior_samples)

"""STEP 2. Save the samples into a csv file."""

posterior_samples.to_csv("C:/Users/kell5379/Documents/Chapter2_May2024/PPC/posterior_samples.csv")
