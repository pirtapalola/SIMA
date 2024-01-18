"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Define the prior.
STEP 3. Instantiate the inference object and pass the simulated data to the inference object.
STEP 4. Train the neural density estimator and build the posterior.

Last updated on 18 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
import os
from torch.distributions import Uniform, LogNormal
from sbi.inference import SNPE
from torch import tensor
from models.tools import MultipleIndependent, minimum_maximum
import pickle

"""
STEP 1. Prepare the simulated data
    -The simulated data is split into input_parameters (5 parameters) and output_values (150 values).
    -After converting them to PyTorch tensors, they are concatenated into combined_input.
"""

# Print the current working directory
current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Create a list of all the filenames
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/' \
       'Methods/Methods_Ecolight/Dec2023_lognormal_priors/EL_test_2_dec2023/EL_test_2_dec2023'
files = [f for f in os.listdir(path) if f.endswith('.txt')]  # Create a list of all the files in the folder

# Define the simulated dataset
num_simulation_runs = len(files)  # Number of reflectances simulated in HydroLight
print("Number of simulations: ", num_simulation_runs)
num_parameters = 5  # Chl-a, SPM, CDOM, wind speed, and depth
num_output_values = 150  # Hyperspectral reflectance between 400nm and 700nm at 2nm spectral resolution

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Dec2023_lognormal_priors/simulated_rrs_dec23_lognorm.csv')
simulated_reflectance.iloc[:, -1:] = files  # Replace the first column repeating "Rrs" with the corresponding file names
simulated_reflectance.rename(columns={simulated_reflectance.columns[-1]: "File_ID"}, inplace=True)  # Rename the column
print(simulated_reflectance)

# Read the csv file containing the inputs of each of the HydroLight simulation runs
hydrolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                               'Dec2023_lognormal_priors/Ecolight_parameter_combinations.csv')
hydrolight_input = hydrolight_input.drop(columns="water")  # Remove the "water" column.
print(hydrolight_input)  # Check that the dataframe contains the correct information.

# Print the minimum and maximum values of each column in the dataframe.
# These should correspond to the empirically realistic range of values.
minimum_maximum(hydrolight_input, ["phy", "cdom", "spm", "wind", "depth"])

# Define theta and x.
theta_dataframe = hydrolight_input  # Theta contains the five input variables.
x_dataframe = simulated_reflectance.drop(columns="File_ID")  # The output values are stored in x. Drop the File_ID.

# Convert the pandas DataFrames to numpy arrays
theta_array = theta_dataframe.to_numpy()
x_array = x_dataframe.to_numpy()

print("Length of theta: ", len(theta_array[0]))
print("Length of x: ", len(x_array[0]))
print(x_array[0])

# Convert the numpy arrays to PyTorch tensors
theta_tensor = torch.tensor(theta_array, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

"""

STEP 2. Define the prior.
    -Each parameter is associated with its own distribution and name.

"""

# Define individual prior distributions
prior_dist_phy = LogNormal(tensor([0.1]), tensor([1.7]))
prior_dist_cdom = LogNormal(tensor([0.05]), tensor([1.7]))
prior_dist_spm = LogNormal(tensor([0.4]), tensor([1.1]))
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

# Save the posterior in binary write mode ("wb")
# The "with" statement ensures that the file is closed
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Dec2023_lognormal_priors/loaded_posterior.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
