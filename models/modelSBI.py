"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Split the data into training and validation sets.
STEP 3. Define the amortized neural network architecture.
STEP 4. Instantiate the amortized neural network.
STEP 5. Train and validate the neural network.
STEP 6. Evaluate the neural network with field-collected data.

Last updated on 22 August 2023 by Pirta Palola

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

# Apply the function
hydrolight_input = create_input_dataframe(files)
print(hydrolight_input)


# Print the minimum and maximum values of each column in the dataframe
minimum_maximum(hydrolight_input, ["phy", "cdom", "spm", "wind", "depth"])

# Define the input and output values for the neural network
input_parameters = simulated_reflectance.drop(columns="File_ID")  # Drop the File_ID column
output_values = hydrolight_input


"""STEP 2. Split the data into training and validation datasets."""

train_size = int(0.8 * num_simulation_runs)  # 80% for training
train_input = input_parameters[:train_size]
train_output = output_values[:train_size]

val_input = input_parameters[train_size:]
val_output = output_values[train_size:]

# Convert the pandas DataFrame to a numpy array
train_input_array = train_input.to_numpy()
train_output_array = train_output.to_numpy()
val_input_array = val_input.to_numpy()
val_output_array = val_output.to_numpy()

# Convert input parameters and training output to PyTorch tensors
train_input_tensor = torch.tensor(train_input_array, dtype=torch.float32)
train_output_tensor = torch.tensor(train_output_array, dtype=torch.float32)


"""

STEP 6. Define the prior distributions.

Each parameter is associated with its own distribution and name.

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


# Combine input parameters and corresponding output values
# combined_train_data = torch.cat([train_input_tensor, train_output_tensor], dim=1)
# print(train_input_tensor)
# print(train_output_tensor)
"""
Arguments for append_simulations():

1) A tensor containing the simulated data, structured as follows:
        - Each row represents a single simulation run.
        - The columns should contain both the input parameters and the corresponding observed (output) data.
        - The order of the columns matters. The input parameters should come before the observed data.
2) prior_log_probs: This argument is optional.
        - If you provide prior log probabilities for your simulated data, 
          you can pass them as a tensor of shape (num_simulations,).
        - The log probabilities should correspond to the provided simulations.   
"""

inference = SNPE(prior=prior)
# Append the combined data to the inference method
inference.append_simulations(train_input_tensor, train_output_tensor)

# Train the inference method
density_estimator = inference.train()
posterior = inference.build_posterior(density_estimator)

x_o = train_output_tensor[1]
posterior_samples = posterior.sample((10000,), x=x_o)

# plot posterior samples
_ = analysis.pairplot(posterior_samples)
plt.show()
