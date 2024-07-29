"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Instantiate the inference object and pass the simulated data to the inference object.
STEP 3. Train the neural density estimator and build the posterior.

Last updated on 29 July 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from sbi.inference import SNPE
from sbi.neural_nets.embedding_nets import CNNEmbedding, FCEmbedding
import pickle
from sbi import utils
import matplotlib.pyplot as plt
import numpy as np

"""

STEP 1. Prepare the simulated data.

"""

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Ecolight_x/'
                                    'simulated_reflectance_10SNR.csv')

# Read the csv file containing the inputs of each of the EcoLight simulation runs
simulator_input = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/No_noise/'
                              'Ecolight_parameter_combinations_train_no_noise.csv')
simulator_input = simulator_input.drop(columns=["water"])  # Remove the "water" column.

samples_phy = simulator_input["phy"]
samples_cdom = simulator_input["cdom"]
samples_nap = simulator_input["spm"]
samples_wind = simulator_input["wind"]
samples_depth = simulator_input["depth"]

# Save the data in a dataframe
theta_dictionary = {"phy": samples_phy,
                    "cdom": samples_cdom,
                    "spm": samples_nap,
                    "wind": samples_wind,
                    "depth": samples_depth}

theta_dataframe = pd.DataFrame(data=theta_dictionary)
print("Theta: ", theta_dataframe)  # Check that the dataframe contains the correct information.

# Define x.
x_dataframe = simulated_reflectance  # X contains the reflectance spectra.

# Convert the pandas DataFrames to numpy arrays
theta_array = theta_dataframe.to_numpy()
x_array = x_dataframe.to_numpy()

# Check the datasets
print("No. of parameter sets", len(theta_array))
print("No. of simulation outputs", len(x_array))
print("Length of theta: ", len(theta_array[0]))
print("Length of x: ", len(x_array[0]))

# Convert the numpy arrays to PyTorch tensors
theta_tensor = torch.tensor(theta_array, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

# Check the tensors
print("Shape of the theta tensor: ", theta_tensor.shape)
print("Shape of the x tensor: ", x_tensor.shape)

""" 

STEP 2. Instantiate the inference object and pass the simulated data to the inference object.

"""

# Define the embedding net (optional)
# embedding_net = CNNEmbedding(input_shape=(61,))
embedding_net = FCEmbedding(input_dim=61)

# Instantiate the neural density estimator
neural_posterior = utils.posterior_nn(
    model="mdn", hidden_features=90, num_components=6)
# num_transforms=3, z_score_theta="independent", embedding_net=embedding_net,

# Instantiate the SNPE inference method
inference = SNPE(prior=None, density_estimator=neural_posterior)

# Append the combined data to the inference object
inference.append_simulations(theta_tensor, x_tensor)

"""

STEP 3. Train the neural density estimator and build the posterior.

"""

# Train the neural density estimator
density_estimator = inference.train(training_batch_size=400, stop_after_epochs=50, max_num_epochs=10000)

# Plot the training and validation curves
plt.figure(1, figsize=(4, 3), dpi=200)
plt.plot(-np.array(inference.summary["training_log_probs"]), label="training")
plt.plot(
    -np.array(inference.summary["validation_log_probs"]), label="validation", alpha=1
)
plt.xlabel("epoch")
plt.ylabel("-log(p)")
plt.legend()
plt.show()

# Use the trained neural density estimator to build the posterior
posterior = inference.build_posterior(density_estimator)

# Save the posterior in binary write mode ("wb")
# The "with" statement ensures that the file is closed
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/"
          "10SNR/Loaded_posteriors/"
          "loaded_posterior1_hyper.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
