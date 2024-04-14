"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the simulated reflectance data.
STEP 2. Load the observation data.
STEP 3. Infer the parameters corresponding to the observation data.

Last updated on 5 April 2024 by Pirta Palola

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
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/Noise_1000SNR/loaded_posteriors/loaded_posterior11.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 2. Load the observation data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                   'Jan2024_lognormal_priors/field_data/'
obs_df = pd.read_csv(observation_path + 'field_surface_reflectance_1000SNR_noise.csv')
print(obs_df)

# Read the file containing the corresponding parameters
obs_parameters = pd.read_csv(observation_path + 'parameters.csv')

# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in obs_parameters["chl"]]
samples_cdom = [i+constant for i in obs_parameters["cdom"]]
samples_nap = [i+constant for i in obs_parameters["spm"]]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_phy = [round(item, 3) for item in samples_phy]
samples_cdom = np.log(samples_cdom)
samples_cdom = [round(item, 3) for item in samples_cdom]
samples_nap = np.log(samples_nap)
samples_nap = [round(item, 3) for item in samples_nap]
samples_wind = np.log(samples_wind)
samples_wind = [round(item, 3) for item in samples_wind]

# Save the transformed data in a dataframe
transformed_dictionary = {"unique_ID": obs_parameters["unique_ID"],
                          "phy": samples_phy, "cdom": samples_cdom, "spm": samples_nap,
                          "wind": samples_wind, "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)
print("Transformed theta: ", transformed_theta)

# Create a list of sample IDs
sample_IDs = obs_parameters["unique_ID"]
print(sample_IDs)

"""STEP 3. Infer the parameters corresponding to the observation data."""

results_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
               'Jan2024_lognormal_priors/Noise_1000SNR/results_model11/model11_'


def infer_from_observation(sample_id):

    # Create an empty dataframe
    results_df = pd.DataFrame()

    # Define x
    x_obs = obs_df[sample_id].to_list()
    x_obs = torch.tensor(x_obs, dtype=torch.float32)

    # Sample from the posterior p(θ|x)
    posterior_samples = loaded_posterior.sample((10000,), x=x_obs)
    theta_samples = posterior_samples.numpy()  # Convert to NumPy array

    # Define theta
    theta_obs = transformed_theta.loc[transformed_theta['unique_ID'] == sample_id]
    print(theta_obs)
    theta_obs = theta_obs.drop(columns="unique_ID")
    theta_obs = theta_obs.iloc[0].to_list()
    print("Theta obs list: ", theta_obs)
    theta_obs = torch.tensor(theta_obs, dtype=torch.float32)

    # Evaluate the log-probability of the posterior samples
    log_probability = loaded_posterior.log_prob(posterior_samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to Numpy array
    log_prob_df = pd.DataFrame(log_prob_np)  # Convert to a dataframe

    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_exp = theta_means
    for i in range(4):  # Apply an exponential transformation to the first 4 theta parameters
        theta_exp[i] = np.exp(theta_means[i])
    for i in range(3):  # Remove the constant from the first 3 theta parameters
        theta_exp[i] = theta_exp[i] - constant
    results_df["Mean"] = theta_exp  # Save the calculated values in a column

    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe

    interval1 = theta_intervals_df.iloc[0]
    interval1_exp = interval1
    for i in range(4):  # Apply an exponential transformation to the first 4 theta parameters
        interval1_exp[i] = np.exp(interval1[i])
    for i in range(3):  # Remove the constant from the first 3 theta parameters
        interval1_exp[i] = interval1[i] - constant

    interval2 = theta_intervals_df.iloc[1]
    interval2_exp = interval2
    for i in range(4):  # Apply an exponential transformation to the first 4 theta parameters
        interval2_exp[i] = np.exp(interval2[i])
    for i in range(3):  # Remove the constant from the first 3 theta parameters
        interval2_exp[i] = interval2[i] - constant

    results_df["2.5percent"] = interval1_exp
    results_df["97.5percent"] = interval2_exp

    # Save the dataframes
    log_prob_df.to_csv(results_path + sample_id + '_log_probability.csv')
    results_df.to_csv(results_path + sample_id + '_results.csv')
    theta_intervals_df.to_csv(results_path + sample_id + '_theta_intervals.csv')

    # Create the figure
    _ = analysis.pairplot(
        samples=posterior_samples,
        points=theta_obs,
        limits=[[0, 1], [0, 1], [0, 10], [0, 10], [0, 2]],
        points_colors=["red", "red", "red", "red", "red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.savefig(results_path + sample_id + '.png')


# Apply the function to real observations
for i in ["RIM03", "RIM04", "RIM05", "ONE05", "ONE06"]:
    infer_from_observation(i)
