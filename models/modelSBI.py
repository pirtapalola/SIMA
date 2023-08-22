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
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.distributions as dist
import sbi.inference as inference
import torch.distributions.transforms as transforms

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


# Create a pandas dataframe containing the input parameters (each row corresponds to a single simulation run)
def create_input_dataframe(list_of_strings):
    split_df = pd.DataFrame(columns=["data", "empty", "water", "phy1", "cdom1", "spm1", "wind1", "depth1"])
    phy_list = []
    cdom_list = []
    spm_list = []
    wind_list = []
    depth_list = []
    depth_list0 = []

    for i in list_of_strings:
        split_string = i.split("_")  # Split the string at the locations marked by underscores
        split_df.loc[len(split_df)] = split_string  # Add the split string as a row in the dataframe

    for n in split_df["phy1"]:  # Create a list where the decimal dots are added
        phy_list.append(float(n[:1] + '.' + n[1:]))
    split_df["phy"] = phy_list  # Create a new column that contains the values with decimal dots

    for n in split_df["cdom1"]:  # Create a list where the decimal dots are added
        cdom_list.append(float(n[:1] + '.' + n[1:]))
    split_df["cdom"] = cdom_list  # Create a new column that contains the values with decimal dots

    for n in split_df["spm1"]:  # Create a list where the decimal dots are added
        spm_list.append(float(n[:1] + '.' + n[1:]))
    split_df["spm"] = spm_list  # Create a new column that contains the values with decimal dots

    for n in split_df["wind1"]:  # Create a list where the decimal dots are added
        wind_list.append(float(n[:1] + '.' + n[1:]))
    split_df["wind"] = wind_list  # Create a new column that contains the values with decimal dots

    for n in split_df["depth1"]:
        sep = '.'
        depth_list0.append(n.split(sep, 1)[0])  # Remove ".txt" from the string based on the separator "."

    for x in depth_list0:  # Create a list where the decimal dots are added
        depth_list.append(float(x[:1] + '.' + x[1:]))
    split_df["depth"] = depth_list  # Create a new column that contains the values with decimal dots

    # Drop the columns that do not contain the values to be inferred
    split_df = split_df.drop(columns=["data", "empty", "water", "phy1", "cdom1", "spm1", "wind1", "depth1"])
    return split_df


# Apply the function
hydrolight_input = create_input_dataframe(files)
print(hydrolight_input)


# Print the minimum and maximum values of each column in the dataframe
def minimum_maximum(dataframe, column_names):
    for i in column_names:
        print(i + " min: " + dataframe[i].min())
        print(i + " max: " + dataframe[i].max())


# minimum_maximum(hydrolight_input, ["phy", "cdom", "spm", "wind", "depth"])


# Define the input and output values for the neural network
input_parameters = simulated_reflectance.drop(columns="File_ID")  # Drop the File_ID column
output_values = hydrolight_input


"""STEP 2. Split the data into training and validation datasets."""

train_size = int(0.8 * num_simulation_runs)  # 80% for training
train_input = input_parameters[:train_size]
train_output = output_values[:train_size]

val_input = input_parameters[train_size:]
val_output = output_values[train_size:]


"""STEP 3. Define the amortized neural network architecture."""


class AmortizedPosterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AmortizedPosterior, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2)  # Two parameters per inferred parameter
        )

    def forward(self, input_data):
        posterior_params = self.network_layers(input_data)
        return posterior_params


"""STEP 4. Instantiate the amortized neural network."""

input_dim = num_parameters  # Input is the inferred input parameters
output_dim = num_parameters * 2  # Two parameters per inferred parameter (e.g., shape and rate for gamma distributions)
hidden_dim = 256
amortized_net = AmortizedPosterior(input_dim, output_dim, hidden_dim)

"""STEP 5. Train and validate the neural network."""

# Convert the pandas DataFrame to a numpy array
train_input_array = train_input.to_numpy()
train_output_array = train_output.to_numpy()
val_input_array = val_input.to_numpy()
val_output_array = val_output.to_numpy()


# Convert input parameters and training output to PyTorch tensors
train_input_tensor = torch.tensor(train_input_array, dtype=torch.float32)
train_output_tensor = torch.tensor(train_output_array, dtype=torch.float32)


# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(amortized_net.parameters(), lr=0.001)


# Lists to store loss values for plotting
train_losses = []
val_losses = []

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients

    # Forward pass
    train_predictions = amortized_net(train_input_tensor)

    # Compute loss
    loss = criterion(train_predictions, train_output_tensor)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Validation
    with torch.no_grad():
        val_predictions = amortized_net(torch.tensor(val_input_array, dtype=torch.float32))
        val_loss = criterion(val_predictions, torch.tensor(val_output_array, dtype=torch.float32))
    # Store loss values for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


"""

STEP 6. Define the prior distributions.

Each parameter is associated with its own distribution and name,
allowing you to create a unified prior distribution for your inference problem.
This approach ensures that you can model different prior distributions for different parameters 
while using the sbi toolbox for inference.

"""

# Define the prior distributions
prior_distribution_params = [
    ("phy", dist.Uniform(0, 1)),
    ("cdom", dist.Gamma(2, 1)),
    ("spm", dist.Gamma(2, 1)),
    ("wind", dist.Uniform(0, 1)),
    ("depth", dist.Uniform(0, 1))]

# Create a list of transforms for each prior distribution
transforms_list = []

for param_name, prior in prior_distribution_params:
    if isinstance(prior, dist.Uniform):
        transforms_list.append(dist.transforms.IdentityTransform())
    elif isinstance(prior, dist.Gamma):
        # Use the LogTransform for gamma distributions
        transforms_list.append(dist.transforms.ExpTransform())
    # Add more conditions for other types of distributions if needed

# Combine the transforms using ComposeTransform
prior_transform = dist.transforms.ComposeTransform(transforms_list)

# Create the composite prior distribution using the actual prior distributions
prior_distributions = [prior for _, prior in prior_distribution_params]
prior = dist.Independent(dist.Product(prior_distributions), 1)

# Apply the prior_transform to the composite prior distribution
prior = dist.TransformedDistribution(prior, prior_transform)

# Step 5: Inference with sbi
posterior_model = inference.NeuralPosterior(amortized_net, prior, input_shape=(input_dim,))
inference_method = inference.SNPE(posterior_model, density_estimator='maf')

# Combine input parameters and corresponding output values
combined_train_data = torch.cat([train_input_tensor, train_output_tensor], dim=1)
print(combined_train_data)
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

# Append the combined data to the inference method
inference_method.append_simulations(combined_train_data)

# Train the inference method
inference_method.train()

# Step 6: Perform Inference
# Generate a new set of observed data from the simulator
new_observed_data = np.random.rand(num_parameters)  # Replace with actual simulation

# Perform inference using the trained sbi method
posterior = inference_method(combined_train_data, new_observed_data)

# Posterior distributions for inferred input parameters