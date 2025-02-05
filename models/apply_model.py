"""

MODELS: Applying the inference scheme to field data
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Conduct inference.
STEP 1. Load the posterior.
STEP 2. Load the observation data.
STEP 3. Infer the theta parameters.

Last updated on 3 February 2025

"""

# Import libraries
import pandas as pd
import torch
from sbi import analysis as analysis
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time
import os

# Get process info
process = psutil.Process(os.getpid())

# Get number of logical CPU cores
num_cores = psutil.cpu_count(logical=True)
ram_gb = psutil.virtual_memory().total / (1024 ** 3)  # Total RAM in GB

# Record start time and initial CPU usage
start_time = time.time()
cpu_start = process.cpu_times().user  # User-mode CPU time

"""STEP 1. Load the posterior."""

# Load the posterior
with open("data/loaded_posteriors/loaded_posterior_100SNR_multi.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 2. Load the observation data."""

# Define whether the data is hyperspectral (hyper) or multispectral (multi) and what the signal-to-noise ratio (SNR) is
model_spec = 'multi_100SNR_'

# Read the csv file containing the observation data
observation_path = 'data/field_data/'
obs_file = 'multi_field_100SNR.csv'  # This file contains the measured reflectance spectra
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
results_path = ('data/results2/' + model_spec)


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
    results_df["Mean"] = theta_means  # Save the calculated values in a column
    print("Mean values: ", theta_means)

    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    interval1 = theta_intervals_df.iloc[0]
    interval2 = theta_intervals_df.iloc[1]
    results_df["2.5percent"] = interval1  # Save the calculated values in a column
    results_df["97.5percent"] = interval2  # Save the calculated values in a column

    # Save the dataframes
    # This contains the 1000 posterior samples
    theta_df.to_csv(results_path + sample_id + '_posterior_samples.csv', index=False)
    # This contains the log-probability of the posterior samples
    log_prob_df.to_csv(results_path + sample_id + '_log_probability.csv', index=False)
    # This contains the mean and credible intervals for each parameter
    results_df.to_csv(results_path + sample_id + '_results2.csv', index=False)

    # Create a figure
    sns.set_style("whitegrid")
    _ = analysis.pairplot(
        samples=posterior_samples,  # The posterior samples
        points=theta_obs,  # The observed theta
        limits=[[0, 2], [0, 1], [0, 20], [0, 20], [0, 20]],  # Define the limits of the x-axes
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "Mineral particles", "Wind", "Depth"],
        offdiag="hist",
        cmap="viridis",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20),
        diag="hist"
    )

    # plt.savefig(results_path + sample_id + '.tiff')  # Save the figure as a tiff file
    # plt.show()


# Apply the function to field observations
for i in ["ONE05"]:
    infer_from_observation(i)

# Record end time and final CPU usage
end_time = time.time()
cpu_end = process.cpu_times().user  # User-mode CPU time

# Calculate execution time and CPU usage
execution_time = end_time - start_time
cpu_usage = cpu_end - cpu_start  # Total CPU time used by script

# Compute CPU Utilization per core
cpu_utilization = (cpu_usage / execution_time) * 100 if execution_time > 0 else 0
cpu_utilization_per_core = cpu_utilization / num_cores if num_cores > 0 else 0

# Calculate relative performance
relative_performance = execution_time * (num_cores * ram_gb)

# Report results
print(f"Execution Time: {execution_time:.6f} seconds")
print(f"CPU Time Used: {cpu_usage:.6f} seconds")
print(f"Overall CPU Utilization: {(cpu_usage / execution_time) * 100:.2f}%")
print(f"Relative Performance: {relative_performance:.6f} (Execution Time × (Cores × RAM in GB))")
print(f"Per-Core CPU Utilization: {cpu_utilization_per_core:.2f}% (across {num_cores} cores)")
print(ram_gb)
