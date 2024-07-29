"""

Conduct inference.
STEP 1. Load the posterior.
STEP 2. Load the observation data.
STEP 3. Infer the theta parameters.

Last updated on 29 July 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from sbi import analysis as analysis
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""STEP 1. Load the posterior."""

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/10SNR/"
          "Loaded_posteriors/loaded_posterior1_hyper.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 2. Load the observation data."""

# Define whether the data is hyperspectral (hyper) or multispectral (multi) and what the signal-to-noise ratio (SNR) is
model_spec = '_hyper_10SNR_'

# Read the csv file containing the observation data
observation_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
obs_file = 'hyper_field_10SNR.csv'  # This file contains the measured reflectance spectra
param_file = 'parameters_TET22.csv'  # This file contains the measured theta parameters

# Read the file containing the reflectance spectra
obs_df = pd.read_csv(observation_path + obs_file)

# Read the file containing the theta parameters
obs_parameters = pd.read_csv(observation_path + param_file)
unique_ids = obs_parameters["unique_ID"]

samples_phy = obs_parameters["chl"]
samples_cdom = obs_parameters["cdom"]
samples_nap = obs_parameters["spm"]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Save the theta values into a dataframe
theta_dictionary = {"unique_ID": unique_ids,
                    "phy": samples_phy,
                    "cdom": samples_cdom,
                    "spm": samples_nap,
                    "wind": samples_wind,
                    "depth": samples_depth}

theta_dataframe = pd.DataFrame(data=theta_dictionary)

# Create a list of sample IDs
sample_IDs = obs_df.columns.tolist()
print(sample_IDs)

"""STEP 3. Infer the theta parameters."""

# Define the path to the folder in which to save the results
results_path = ('C:/Users/kell5379/Documents/Chapter2_May2024/Final/'
                'Results/Hyper_10SNR/1' + model_spec)

# Define a function to conduct inference


def infer_from_observation(sample_id):

    # Create an empty dataframe
    results_df = pd.DataFrame()

    # Define x
    x_obs = obs_df[sample_id].to_list()
    # print('Reflectance data: ', x_obs)
    x_obs = torch.tensor(x_obs, dtype=torch.float32)  # Convert to a tensor

    # Sample from the posterior p(θ|x)
    posterior_samples = loaded_posterior.sample((1000,), x=x_obs)  # Take 1000 samples
    theta_samples = posterior_samples.numpy()  # Convert to a NumPy array
    theta_df = pd.DataFrame(theta_samples)  # Convert to a dataframe

    # Define theta
    theta_obs = theta_dataframe.loc[theta_dataframe['unique_ID'] == sample_id]
    # print(theta_obs)
    theta_obs = theta_obs.drop(columns="unique_ID")  # Drop the unique_ID column
    theta_obs = theta_obs.iloc[0].to_list()  # Convert to a list
    print("Theta: ", theta_obs)
    theta_obs = torch.tensor(theta_obs, dtype=torch.float32)  # Convert to a tensor

    # Evaluate the log-probability of the posterior samples
    log_probability = loaded_posterior.log_prob(posterior_samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to a Numpy array
    log_prob_df = pd.DataFrame(log_prob_np)  # Convert to a dataframe

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    print("Mean values: ", theta_means)
    results_df["Mean"] = theta_means  # Save the calculated values in a column

    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    interval1 = theta_intervals_df.iloc[0]
    # print("Interval 1 values: ", interval1)
    interval2 = theta_intervals_df.iloc[1]
    # print("Interval 2 values: ", interval2)
    results_df["2.5percent"] = interval1  # Save the calculated values in a column
    results_df["97.5percent"] = interval2  # Save the calculated values in a column

    # Save the dataframes
    # This contains the 1000 posterior samples
    theta_df.to_csv(results_path + sample_id + '_posterior_samples.csv', index=False)
    # This contains the log-probability of the posterior samples
    log_prob_df.to_csv(results_path + sample_id + '_log_probability.csv', index=False)
    # This contains the mean and credible intervals for each parameter
    results_df.to_csv(results_path + sample_id + '_results.csv', index=False)

    # Create a figure
    _ = analysis.pairplot(
        samples=posterior_samples,  # The posterior samples
        points=theta_obs,  # The observed theta
        limits=[[0, 7], [0, 2], [0, 30], [0, 20], [0, 20]],  # Define the limits of the x-axes
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "Mineral particles", "Wind", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20),
        diag="hist"
    )
    plt.savefig(results_path + sample_id + '.png')  # Save the figure as a png file


# Apply the function to field observations
for i in ["ONE05", "RIM03", "RIM04", "RIM05"]:  # , "GID_2505", "GID_2506"
    infer_from_observation(i)
