"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Define the prior.
STEP 3. Instantiate the inference object and pass the simulated data to the inference object.
STEP 4. Train the neural density estimator and build the posterior.

Last updated on 19 December 2023 by Pirta Palola

"""

# Import libraries

import pandas as pd
import torch
import os
from torch.distributions import Uniform, LogNormal
from sbi.inference import SNPE
from sbi import analysis as analysis
from torch import tensor
from models.tools import MultipleIndependent, create_input_dataframe, minimum_maximum, TruncatedLogNormal
import matplotlib.pyplot as plt
import numpy as np


"""
STEP 1. Prepare the simulated data
    -The simulated data is split into input_parameters (5 parameters) and output_values (150 values).
    -After converting them to PyTorch tensors, they are concatenated into combined_input.

"""

# Print the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Create a list of all the filenames
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/test_setup2'  # Define the file location
files = [f for f in os.listdir(path) if f.endswith('.txt')]  # Create a list of all the files in the folder

# Define the simulated dataset
num_simulation_runs = len(files)  # Number of reflectances simulated in HydroLight
num_parameters = 5  # Chl-a, SPM, CDOM, wind speed, and depth
num_output_values = 150  # Hyperspectral reflectance between 400nm and 700nm at 2nm spectral resolution

# Read the csv file containing the simulated Rrs data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/test_setup2.csv')
simulated_reflectance.iloc[:, 0] = files  # Replace the first column repeating "Rrs" with the corresponding file names
simulated_reflectance.rename(columns={simulated_reflectance.columns[0]: "File_ID"}, inplace=True)  # Rename the column

# Apply the function to create a dataframe containing the inputs of each of the HydroLight simulation runs
hydrolight_input = create_input_dataframe(files)
print(hydrolight_input)

# Print the minimum and maximum values of each column in the dataframe
# These should correspond to the empirically realistic range of values.
minimum_maximum(hydrolight_input, ["phy", "cdom", "spm", "wind", "depth"])

# Define theta and x
theta_dataframe = hydrolight_input  # Theta contains the five input variables.
x_dataframe = simulated_reflectance.drop(columns="File_ID")  # The output values are stored in x. Drop the File_ID.

# Convert the pandas DataFrames to numpy arrays
theta_array = theta_dataframe.to_numpy()
x_array = x_dataframe.to_numpy()

# Convert the numpy arrays to PyTorch tensors
theta_tensor = torch.tensor(theta_array, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

"""

STEP 2. Define the prior.
    -Each parameter is associated with its own distribution and name.

"""

# Define individual prior distributions
prior_dist_phy = LogNormal(tensor([0.1]), tensor([1.7]), tensor([0.001]), tensor([10]))
prior_dist_cdom = LogNormal(tensor([0.05]), tensor([1.7]), tensor([0.001]), tensor([5]))
prior_dist_spm = LogNormal(tensor([0.4]), tensor([1.1]), tensor([0.001]), tensor([50]))
prior_dist_wind = LogNormal(tensor([1.85]), tensor([0.33]))
prior_dist_depth = Uniform(tensor([0.10]), tensor([20.0]))

# Create a list of prior distributions
prior_distributions = [
    prior_dist_phy,
    prior_dist_cdom,
    prior_dist_spm,
    prior_dist_wind,
    prior_dist_depth,
]

# Create the combined distribution using MultipleIndependent
prior = MultipleIndependent(prior_distributions)
print(prior)


# Combine input parameters and corresponding output values
# combined_train_data = torch.cat([train_input_tensor, train_output_tensor], dim=1)
# print(train_input_tensor)
# print(train_output_tensor)

"""

STEP 3. Instantiate the inference object and pass the simulated data to the inference object.

"""

# Instantiate the SNPE inference method
inference = SNPE(prior=prior)

# Append the combined data to the inference object
inference.append_simulations(theta_tensor, x_tensor)

"""

STEP 4. Train the neural density estimator and build the posterior.

"""

# Train the neural density estimator
density_estimator = inference.train()

# Use the trained neural density estimator to build the posterior
posterior = inference.build_posterior(density_estimator)

# Save the trained density estimator
torch.save(density_estimator.state_dict(), 'density_estimator.pth')

# Define an observation x
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/RIM03_2022_surface_reflectance_interpolated_400_700nm.csv'
obs_df = pd.read_csv(observation_path)
x_o = obs_df['reflectance']

# Given this observation, sample from the posterior p(θ|x), or plot it.
posterior_samples = posterior.sample((1000,), x=x_o)

# Evaluate the log-probability of the posterior samples
log_probability = posterior.log_prob(posterior_samples, x=x_o)
log_prob_np = log_probability.numpy()  # convert to Numpy array
log_prob_df = pd.DataFrame(log_prob_np)  # convert to a dataframe
log_prob_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/log_probability_test_sample.csv')

# Plot posterior samples
_ = analysis.pairplot(posterior_samples, limits=[[0, 10], [0, 5], [0, 30], [0, 10], [0, 20]], figsize=(6, 6))
plt.show()

# Print the posterior to know how it was trained
print(posterior)

theta_samples = posterior_samples.numpy()  # Convert to NumPy array

# Mean estimates for each parameter
theta_means = torch.mean(posterior_samples, dim=0)
print(theta_means)

# Credible intervals (e.g., 95% interval) for each parameter using NumPy
theta_intervals = np.percentile(theta_samples, [2.5, 97.5], axis=0)
print(theta_intervals)
