"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the simulated reflectance data.
STEP 2. Load the observation data.
STEP 3. Infer the parameters corresponding to the observation data.

Last updated on 26 July 2024 by Pirta Palola

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
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/1000SNR/"
          "Loaded_posteriors/loaded_posterior1_hp.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 2. Load the observation data."""

model_spec = '_hp_1000SNR_'

# Read the csv file containing the observation data
observation_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
obs_file = 'hp_field_1000SNR.csv'
param_file = 'parameters_TET22.csv'

obs_df = pd.read_csv(observation_path + obs_file)
# obs_df = obs_df.drop(columns=["unique_ID"])

# Read the file containing the corresponding parameters
obs_parameters = pd.read_csv(observation_path + param_file)
# print(obs_parameters)
unique_ids = obs_parameters["unique_ID"]
# print(unique_ids)

# Define
samples_phy = obs_parameters["chl"]
samples_cdom = obs_parameters["cdom"]
samples_nap = obs_parameters["spm"]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Save the data into a dataframe
theta_dictionary = {"unique_ID": unique_ids,
                    "phy": samples_phy,
                    "cdom": samples_cdom,
                    "spm": samples_nap,
                    "wind": samples_wind,
                    "depth": samples_depth}  #

theta_dataframe = pd.DataFrame(data=theta_dictionary)

# Create a list of sample IDs
sample_IDs = obs_df.columns.tolist()
# sample_IDs = list(obs_parameters["unique_ID"])
print(sample_IDs)

"""STEP 3. Infer the parameters corresponding to the observation data."""

# Define the path to the folder in which to save the results
results_path = ('C:/Users/kell5379/Documents/Chapter2_May2024/Final/'
                'Results/HP_1000SNR/1' + model_spec)


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


# Apply the function to real observations
for i in ["ONE05", "RIM03", "RIM04", "RIM05"]:
    infer_from_observation(i)
