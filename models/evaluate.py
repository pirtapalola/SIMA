"""

Assess the performance of the inference scheme.
STEP 1. Sample from the posterior.
STEP 2. Read the ground-truth data.
STEP 3. Define functions to assess inference performance.
STEP 4. Apply the functions.

Last updated on 1 June 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import pickle
import torch

"""STEP 1. Sample from the posterior."""

# Define sample IDs
sample_id_list = ['ONE05', 'RIM03', 'RIM04', 'RIM05']
sample_id_index = 1

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/1000SNR/"
          "Loaded_posteriors/loaded_posterior29.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the observation data
observation_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
obs_file = 'hp_field_1000SNR.csv'
obs_df = pd.read_csv(observation_path + obs_file)


# Define a function to sample from the posterior
def posterior_sampling(sample_id, observation_dataframe):
    x_obs = observation_dataframe[sample_id].to_list()
    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    modified_data = torch.cat((samples[:, :1], samples[:, 2:]), dim=1)
    posterior_samples_array = modified_data.numpy()  # Convert to NumPy array
    return posterior_samples_array


# Create an empty list
posterior_list = []

# Add the posterior samples of each field site as elements into the list
for item in sample_id_list:
    samples_i = posterior_sampling(item, obs_df)
    posterior_list.append(samples_i)

"""STEP 2. Read the ground-truth data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
param_file = 'parameters_TET22.csv'
obs_parameters = pd.read_csv(observation_path + param_file)
unique_ids = obs_parameters["unique_ID"]

# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in obs_parameters["chl"]]
samples_nap = [i+constant for i in obs_parameters["spm"]]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_phy = [round(item, 3) for item in samples_phy]
samples_nap = np.log(samples_nap)
samples_nap = [round(item, 3) for item in samples_nap]
samples_wind = np.log(samples_wind)
samples_wind = [round(item, 3) for item in samples_wind]

# Save the transformed data in a dataframe
transformed_dictionary = {"unique_ID": unique_ids,
                          "phy": samples_phy, "spm": samples_nap,
                          "wind": samples_wind, "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)


# Function to define ground-truth parameters
def groundtruth(sample_id, theta_dataframe):
    theta_obs = theta_dataframe.loc[theta_dataframe['unique_ID'] == sample_id]
    theta_obs = theta_obs.drop(columns="unique_ID")
    theta_obs = theta_obs.iloc[0].to_list()
    theta_obs_array = np.array(theta_obs)  # Convert to NumPy array
    return theta_obs_array


# Create an empty list
gt_list = []

# Add the posterior samples of each field site as elements into the list
for item in sample_id_list:
    gt_i = groundtruth(item, transformed_theta)
    gt_list.append(gt_i)

"""STEP 3. Define functions to assess inference performance."""


# Measure how often the true parameter value falls within the credible intervals of the posterior distributions
def coverage_probability(post_samples_list, true_values):
    coverage_counts = 0
    for post_samples, true_value in zip(post_samples_list, true_values):
        theta_intervals = np.percentile(post_samples, [2.5, 97.5], axis=0)
        lower_bound = theta_intervals[0]
        upper_bound = theta_intervals[1]
        if true_value >= lower_bound and true_value <= upper_bound:
            coverage_counts += 1
    return coverage_counts / len(true_values)


# Measure the distance between the estimated posterior distributions and the true parameter value,
# considering the entire shape of the distributions
def wasserstein(post_samples, true_value):
    return wasserstein_distance(post_samples, [true_value])


# The average squared difference between the true parameter value and samples from the posterior distribution
def pmse(post_samples, true_value):
    return np.mean((post_samples - true_value) ** 2)


# Evaluate the posterior distributions based on the log probability of the true parameter values
# Higher log scores correspond to better performance
def log_score(post_samples, true_values):
    log_probs = []
    for i in range(len(true_values)):
        kde = gaussian_kde(post_samples[:, i])
        log_probs.append(kde.logpdf(true_values[i]))
    return np.mean(log_probs)


"""STEP 4. Apply the functions."""

# Access the data corresponding to one sampling site
posterior_samples = posterior_list[sample_id_index]
gt_array = gt_list[sample_id_index]

# From the ground-truth array, access each parameter
phy_gt = gt_array[0]
spm_gt = gt_array[1]
wind_gt = gt_array[2]
depth_gt = gt_array[3]

# Extracting elements
phy_posterior = [row[0] for row in posterior_samples]
spm_posterior = [row[1] for row in posterior_samples]
wind_posterior = [row[2] for row in posterior_samples]
depth_posterior = [row[3] for row in posterior_samples]

coverage = coverage_probability(wind_posterior, wind_gt)
#wasserstein_dist = wasserstein(first_elements, phy_gt)
#pmse_value = pmse(first_elements, phy_gt)

print(f"Coverage Probability: {coverage}")
#print(f"Wasserstein Distance: {wasserstein_dist}")
#print(f"PMSE: {pmse_value}")
