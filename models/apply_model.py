"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the simulated reflectance data.
STEP 2. Load the observation data.
STEP 3. Infer the parameters corresponding to the observation data.

Last updated on 6 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from sbi import analysis as analysis
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""STEP 1. Load the posterior and the simulated reflectance data."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/loaded_posterior9.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/simulated_reflectance.csv')
x_dataframe = simulated_reflectance.drop(columns={simulated_reflectance.columns[-1]})

"""STEP 2. Load the observation data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                   'Jan2024_lognormal_priors/field_data/'
obs_df = pd.read_csv(observation_path + 'field_surface_reflectance_brown_coral.csv')

# Read the file containing the corresponding parameters
obs_parameters = pd.read_csv(observation_path + 'parameters_brown_coral.csv')

# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in obs_parameters["chl"]]
samples_cdom = [i+constant for i in obs_parameters["cdom"]]
samples_nap = [i+constant for i in obs_parameters["spm"]]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_cdom = np.log(samples_cdom)
samples_nap = np.log(samples_nap)
samples_wind = np.log(samples_wind)

# Save the transformed data in a dataframe
transformed_dictionary = {"unique_ID": obs_parameters["unique_ID"],
                          "phy": samples_phy, "cdom": samples_cdom, "spm": samples_nap,
                          "wind": samples_wind, "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)

# Create a list of sample IDs
sample_IDs = obs_parameters["unique_ID"]
print(sample_IDs)

"""STEP 3. Infer the parameters corresponding to the observation data."""

results_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
               'Jan2024_lognormal_priors/results/'


def infer_from_observation(sample_id):
    x_obs = obs_df[sample_id]
    x_obs_parameters = transformed_theta.loc[transformed_theta['unique_ID'] == sample_id]
    # x_o = min_max_normalisation(x_obs)  # Apply max-min normalisation
    posterior_samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    # Evaluate the log-probability of the posterior samples
    log_probability = loaded_posterior.log_prob(posterior_samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to Numpy array
    log_prob_df = pd.DataFrame(log_prob_np)  # Convert to a dataframe
    log_prob_df.to_csv(results_path + sample_id + '_log_probability.csv')
    theta_samples = posterior_samples.numpy()  # Convert to NumPy array
    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame(theta_means)  # Convert to a dataframe
    theta_means_df.to_csv(results_path + sample_id + '_theta_means.csv')
    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    theta_intervals_df.to_csv(results_path + sample_id + '_theta_intervals.csv')
    # Create the figure
    _ = analysis.pairplot(
        samples=posterior_samples,
        points=x_obs_parameters,
        limits=[[0, 1], [0, 1], [0, 10], [0, 10], [0, 2]],
        points_colors=["red"],
        figsize=(8, 8),
        labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.savefig(results_path + sample_id + '.png')


# Apply the function to real observations
for item in sample_IDs:
    infer_from_observation(item)
