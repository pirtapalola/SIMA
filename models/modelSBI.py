"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Split the data into training and validation sets.
STEP 3. Define the amortized neural network architecture.
STEP 4. Instantiate the amortized neural network.
STEP 5. Train and validate the neural network.
STEP 6. Evaluate the neural network with field-collected data.

Last updated on 21 August 2023 by Pirta Palola

"""

# Import libraries

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os


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
num_parameters = 150  # Chl-a, SPM, CDOM, wind speed, and depth
num_output_values = 5  # Hyperspectral reflectance between 400nm and 700nm at 2nm spectral resolution


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
    def __init__(self, input_dimension, output_dimension, hidden_dimension=256):
        super(AmortizedPosterior, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.hidden_dimension = hidden_dimension

        self.network_layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, input_data):
        posterior_estimate = self.network_layers(input_data)
        return posterior_estimate


"""STEP 4. Instantiate the amortized neural network."""

input_dim = num_parameters
output_dim = num_output_values
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

"""STEP 6. Evaluate the neural network with field-collected data."""

# After training, use the trained network for inference
# For example, to estimate posterior for new data:
print(len(train_output_tensor[1]))
new_data = train_input_tensor[1]  # Replace with your new data
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
posterior_estimate_new = amortized_net(new_data_tensor)

print("Estimated Posterior for New Data:", posterior_estimate_new)
