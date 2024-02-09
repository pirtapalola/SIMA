"""

Apply the posterior estimated in "modelSBI"
STEP 1. Load the posterior and the simulated reflectance data.
STEP 2. Load the observation data.
STEP 3. Infer the parameters corresponding to the observation data.

Last updated on 5 February 2024 by Pirta Palola

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
          "Dec2023_lognormal_priors/loaded_posterior.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Dec2023_lognormal_priors/simulated_rrs_dec23_lognorm.csv')
x_dataframe = simulated_reflectance.drop(columns={simulated_reflectance.columns[-1]})

"""STEP 2. Load the observation data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/'
obs_df = pd.read_csv(observation_path + 'just_above_surface_reflectance_tetiaroa_2022_coralbrown.csv')

# Read the file containing the corresponding parameters
obs_parameters = pd.read_csv(observation_path + 'parameters_tetiaroa_2022.csv')

# Create a list of sample IDs
sample_IDs = list(obs_df.columns)
print(sample_IDs)

# Test simulation run no. 2, correct input parameters: [0.28, 0.11, 1.18, 4.69, 6.23]
x_o_test = x_dataframe.iloc[10]
posterior_samples_test = loaded_posterior.sample((1000,), x=x_o_test)  # Sample from the posterior p(θ|x)
print(x_o_test)

"""STEP 3. Infer the parameters corresponding to the observation data."""
results_path = 'C:/Users/pirtapalola/Documents/Methodology/Inference/test_results/'


def infer_from_observation(sample_id):
    x_obs = obs_df[sample_id]
    x_obs_parameters = obs_parameters[sample_id]
    # x_o = min_max_normalisation(x_obs)  # Apply max-min normalisation
    posterior_samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    # Evaluate the log-probability of the posterior samples
    log_probability = loaded_posterior.log_prob(posterior_samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to Numpy array
    log_prob_df = pd.DataFrame(log_prob_np)  # Convert to a dataframe
    log_prob_df.to_csv(results_path + 'log_probability/' + sample_id + '_log_probability.csv')
    theta_samples = posterior_samples.numpy()  # Convert to NumPy array
    # Mean estimates for each parameter
    theta_means = torch.mean(posterior_samples, dim=0)
    theta_means_df = pd.DataFrame(theta_means)  # Convert to a dataframe
    theta_means_df.to_csv(results_path + 'theta_means/' + sample_id + '_theta_means.csv')
    # Credible intervals (e.g., 95% interval) for each parameter using NumPy
    theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
    theta_intervals_df = pd.DataFrame(theta_intervals)  # Convert to a dataframe
    theta_intervals_df.to_csv(results_path + 'theta_intervals/' + sample_id + '_theta_intervals.csv')
    # Create the figure
    # _ = analysis.pairplot(posterior_samples, limits=[[0, 10], [0, 5], [0, 30], [0, 10], [0, 20]], figsize=(6, 6))
    _ = analysis.pairplot(
        samples=posterior_samples,
        points=x_obs_parameters,
        limits=[[0, 10], [0, 5], [0, 30], [0, 10], [0, 20]],
        points_colors=["red", "red", "red"],
        figsize=(8, 8),
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.savefig(results_path + 'figures/' + sample_id + '.png')


# Apply the function
for item in sample_IDs:
    infer_from_observation(x_o_test)
