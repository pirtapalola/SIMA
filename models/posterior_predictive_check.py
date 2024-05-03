"""

Posterior predictive check
STEP 1. Draw samples from the posterior.
STEP 2. Save the samples into a csv file.

Last updated on 1 May 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import pickle

"""STEP 1. Draw samples from the posterior."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/Noise_1000SNR/loaded_posteriors/loaded_posterior1.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Draw theta samples from the posterior
posterior_samples = loaded_posterior.sample((5000,))

print(posterior_samples)

"""STEP 2. Save the samples into a csv file."""

