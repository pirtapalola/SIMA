"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Define the prior.
STEP 3. Instantiate the inference object and pass the simulated data to the inference object.
STEP 4. Train the neural density estimator and build the posterior.

Last updated on 04 September 2023 by Pirta Palola

"""

# Import libraries

import pandas as pd
import torch
import os
from torch.distributions import Uniform, Gamma
from sbi.inference import SNPE
from sbi import analysis as analysis
from torch import tensor
from models.tools import MultipleIndependent, create_input_dataframe, minimum_maximum
import matplotlib.pyplot as plt

"""
STEP 1. Prepare the simulated data
    -The simulated data is split into input_parameters (5 parameters) and output_values (150 values).
    -After converting them to PyTorch tensors, they are concatenated into combined_input.

"""

# Print the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Create a list of all the filenames
path = 'data/simulated_data/'  # Define the file location
files = [f for f in os.listdir(path) if f.endswith('.txt')]  # Create a list of all the files in the folder

# Define the simulated dataset
num_simulation_runs = len(files)  # Number of reflectances simulated in HydroLight
num_parameters = 5  # Chl-a, SPM, CDOM, wind speed, and depth
num_output_values = 150  # Hyperspectral reflectance between 400nm and 700nm at 2nm spectral resolution

# Read the csv file containing the simulated Rrs data into a pandas dataframe
simulated_reflectance = pd.read_csv('data/HL_output_combined_dataframe.csv')
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
prior_dist_phy = Gamma(tensor([1.1]), tensor([1.1]))
prior_dist_cdom = Gamma(tensor([1.2]), tensor([3.0]))
prior_dist_spm = Gamma(tensor([3.0]), tensor([0.6]))
prior_dist_wind = Uniform(tensor([0.0]), tensor([10.0]))
prior_dist_depth = Uniform(tensor([0.0]), tensor([20.0]))

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

# Define an observation x
x_o = x_tensor[1]

# Given this observation, sample from the posterior p(θ|x), or plot it.
posterior_samples = posterior.sample((10000,), x=x_o)

# Evaluate the log-probability of the posterior samples
log_probability = posterior.log_prob(posterior_samples, x=x_o)

# Plot posterior samples
_ = analysis.pairplot(posterior_samples, limits=[[0, 10], [0, 5], [0, 30], [0, 10], [0, 20]], figsize=(6, 6))
plt.show()

# Print the posterior to know how it was trained
print(posterior)
