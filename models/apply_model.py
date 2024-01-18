"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the reflectance data.
STEP 2. Define an observation.
STEP 3. Infer the parameters corresponding to the observation.

Last updated on 18 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from sbi import analysis as analysis
import matplotlib.pyplot as plt
import numpy as np
import pickle

"""STEP 1. Load the posterior and the reflectance data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Dec2023_lognormal_priors/loaded_posterior.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Dec2023_lognormal_priors/simulated_rrs_dec23_lognorm.csv')
x_dataframe = simulated_reflectance.drop(columns={simulated_reflectance.columns[-1]})

"""STEP 2. Define an observation."""

# Define an observation x
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/RIM03_2022_surface_reflectance_interpolated_400_700nm.csv'
obs_df = pd.read_csv(observation_path)
# x_o = obs_df['reflectance']

# Test simulation run no. 2, correct input parameters: [0.28, 0.11, 1.18, 4.69, 6.23]
x_o = x_dataframe.iloc[1]
print(x_o)

"""STEP 3. Infer the parameters corresponding to the observation."""

# Given this observation, sample from the posterior p(θ|x), or plot it.
posterior_samples = loaded_posterior.sample((1000,), x=x_o)

# Evaluate the log-probability of the posterior samples
log_probability = loaded_posterior.log_prob(posterior_samples, x=x_o)
log_prob_np = log_probability.numpy()  # convert to Numpy array
log_prob_df = pd.DataFrame(log_prob_np)  # convert to a dataframe
log_prob_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'
                   'Methods_Ecolight/Dec2023_lognormal_priors/log_probability_dec23_lognormal.csv')

# Plot posterior samples
_ = analysis.pairplot(posterior_samples, limits=[[0, 10], [0, 5], [0, 30], [0, 10], [0, 20]], figsize=(6, 6))
plt.show()

# Print the posterior to know how it was trained
print(loaded_posterior)

theta_samples = posterior_samples.numpy()  # Convert to NumPy array

# Mean estimates for each parameter
theta_means = torch.mean(posterior_samples, dim=0)
print(theta_means)

# Credible intervals (e.g., 95% interval) for each parameter using NumPy
theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
print(theta_intervals)
