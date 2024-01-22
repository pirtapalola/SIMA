"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the simulated reflectance data.
STEP 2. Load the observation data.
STEP 3. Infer the parameters corresponding to the observation data.

Last updated on 22 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from sbi import analysis as analysis
import numpy as np
import pickle
from models.tools import min_max_normalisation

"""STEP 1. Load the posterior and the simulated reflectance data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Dec2023_lognormal_priors/loaded_posterior.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Dec2023_lognormal_priors/simulated_rrs_dec23_lognorm.csv')
x_dataframe = simulated_reflectance.drop(columns={simulated_reflectance.columns[-1]})

"""STEP 2. Load the observation data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/Methods_Ecolight/In_water_calibration_2022/smooth_surface_reflectance_2022.csv'
obs_df = pd.read_csv(observation_path)

# Create a list of sample IDs
sample_IDs = list(obs_df.columns)
print(sample_IDs)

# Test simulation run no. 2, correct input parameters: [0.28, 0.11, 1.18, 4.69, 6.23]
# x_o = x_dataframe.iloc[1]

"""STEP 3. Infer the parameters corresponding to the observation data."""


def infer_from_observation(sample_id):
    x_obs = obs_df[sample_id]
    # x_o = min_max_normalisation(x_obs)  # Apply max-min normalisation
    posterior_samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    # Evaluate the log-probability of the posterior samples
    log_probability = loaded_posterior.log_prob(posterior_samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to Numpy array
    log_prob_df = pd.DataFrame(log_prob_np)  # Convert to a dataframe
    log_prob_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'
                       'Methods_Ecolight/Dec2023_lognormal_priors/results/log_probability/'
                       + sample_id + '_log_probability.csv')
    theta_samples = posterior_samples.numpy()  # Convert to NumPy array
    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame(theta_means)  # Convert to a dataframe
    theta_means_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'
                          'Methods_Ecolight/Dec2023_lognormal_priors/results/theta_means/'
                          + sample_id + '_theta_means.csv')
    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    theta_intervals_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'
                              'Methods_Ecolight/Dec2023_lognormal_priors/results/theta_intervals/'
                              + sample_id + '_theta_intervals.csv')


# Apply the function
for item in sample_IDs:
    infer_from_observation(item)
