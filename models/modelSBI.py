"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Define the prior.
STEP 3. Instantiate the inference object and pass the simulated data to the inference object.
STEP 4. Train the neural density estimator and build the posterior.

Last updated on 29 April 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import torch
from torch.distributions import Uniform
from sbi.inference import SNPE
from sbi.neural_nets.embedding_nets import CNNEmbedding
from torch import tensor
from models.tools import MultipleIndependent
import pickle
from sbi import utils
import matplotlib.pyplot as plt
import numpy as np

"""

STEP 1. Prepare the simulated data.

"""

# Read the csv file containing the simulated reflectance data into a pandas dataframe
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                                    'simulated_reflectance_1000SNR_noise.csv')

# Read the csv file containing the inputs of each of the EcoLight simulation runs
simulator_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                              'Jan2024_lognormal_priors/Ecolight_parameter_combinations.csv')
simulator_input = simulator_input.drop(columns=["water", "cdom", "wind"])  # Remove the "water" column.

# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in simulator_input["phy"]]
# samples_cdom = [i+constant for i in simulator_input["cdom"]]
samples_nap = [i+constant for i in simulator_input["spm"]]
# samples_wind = simulator_input["wind"]
samples_depth = simulator_input["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_phy = [round(item, 3) for item in samples_phy]
# samples_cdom = np.log(samples_cdom)
# samples_cdom = [round(item, 3) for item in samples_cdom]
samples_nap = np.log(samples_nap)
samples_nap = [round(item, 3) for item in samples_nap]
# samples_wind = np.log(samples_wind)
# samples_wind = [round(item, 3) for item in samples_wind]

# Save the transformed data in a dataframe "cdom": samples_cdom, "wind": samples_wind,
transformed_dictionary = {"phy": samples_phy, "spm": samples_nap,
                          "depth": samples_depth}
transformed_theta = pd.DataFrame(data=transformed_dictionary)
print("Untransformed theta: ", simulator_input)
print("Transformed theta: ", transformed_theta)  # Check that the dataframe contains the correct information.


# Define theta and x.
theta_dataframe = transformed_theta  # Theta contains the five input variables.
x_dataframe = simulated_reflectance  # X contains the reflectance spectra.

# Convert the pandas DataFrames to numpy arrays
theta_array = theta_dataframe.to_numpy()
x_array = x_dataframe.to_numpy()

print("No. of parameter sets", len(theta_array))
print("No. of simulation outputs", len(x_array))
print("Length of theta: ", len(theta_array[0]))
print("Length of x: ", len(x_array[0]))

# Convert the numpy arrays to PyTorch tensors
theta_tensor = torch.tensor(theta_array, dtype=torch.float32)
x_tensor = torch.tensor(x_array, dtype=torch.float32)

print("Shape of the theta tensor: ", theta_tensor.shape)
print("Shape of the x tensor: ", x_tensor.shape)

"""

STEP 2. Define the prior.

"""

# Define individual prior distributions
prior_dist_phy = Uniform(tensor([0.]), tensor([100.]))
prior_dist_cdom = Uniform(tensor([0.]), tensor([100.]))
prior_dist_spm = Uniform(tensor([0.]), tensor([100.]))
prior_dist_wind = Uniform(tensor([0.]), tensor([100.]))
prior_dist_depth = Uniform(tensor([0.]), tensor([100.]))

# Create a list of prior distributions
prior_distributions = [
    prior_dist_phy,
    # prior_dist_cdom,
    prior_dist_spm,
    # prior_dist_wind,
    prior_dist_depth
]

# Create the combined distribution using MultipleIndependent
prior = MultipleIndependent(prior_distributions)
print(prior)

""" 

STEP 3. Instantiate the inference object and pass the simulated data to the inference object.

"""

# Define the embedding net
# embedding_net = CNNEmbedding(input_shape=(61,))
# embedding_net = FCEmbedding(input_dim=61)

# Instantiate the neural density estimator
neural_posterior = utils.posterior_nn(
    model="mdn", hidden_features=40, num_transforms=3)
# num_transforms=3, z_score_theta="independent", embedding_net=embedding_net, num_components=3

# Instantiate the SNPE inference method
inference = SNPE(prior=prior, density_estimator=neural_posterior)

# Append the combined data to the inference object
inference.append_simulations(theta_tensor, x_tensor)

"""

STEP 4. Train the neural density estimator and build the posterior.

"""

# Train the neural density estimator
density_estimator = inference.train()

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
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/Noise_1000SNR/loaded_posteriors/loaded_posterior17.pkl", "wb") as handle:
    pickle.dump(posterior, handle)
