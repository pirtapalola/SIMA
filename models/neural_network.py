"""

Implementation of a neural network
STEP 1. Read in the datasets.


"""

from models.tools import create_input_dataframe
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.optim as optim

"""

STEP 1. Read in the datasets.

"""
# Create a list of all the filenames
path = 'data/simulated_data/'  # Define the file location
files = [f for f in os.listdir(path) if f.endswith('.txt')]  # Create a list of all the files in the folder


# Define the simulated dataset
num_simulation_runs = len(files)  # Number of reflectances simulated in HydroLight
num_parameters = 5  # Chl-a, SPM, CDOM, wind speed, and depth
num_input_values = 150  # Hyperspectral reflectance between 400nm and 700nm at 2nm spectral resolution


# Read the csv file containing the simulated Rrs data into a pandas dataframe
simulated_reflectance = pd.read_csv('data/HL_output_combined_dataframe.csv')
simulated_reflectance.iloc[:, 0] = files  # Replace the first column repeating "Rrs" with the corresponding file names
simulated_reflectance.rename(columns={simulated_reflectance.columns[0]: "File_ID"}, inplace=True)  # Rename the column

# Apply the function to create the input dataframe based on the information in the filenames
hydrolight_input = create_input_dataframe(files)
print(hydrolight_input)


"""

STEP 2. Define the amortized neural network architecture.

"""


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


"""

STEP 3. Instantiate the amortized neural network.

"""

# Define key variables
input_dim = num_input_values
output_dim = num_parameters
hidden_dim = 256
amortized_net = AmortizedPosterior(input_dim, output_dim, hidden_dim)


"""

STEP 4. Split the data into training and validation data.

"""

input_parameters = simulated_reflectance.drop(columns="File_ID")  # Drop the File_ID column
output_values = hydrolight_input

train_size = int(0.8 * num_simulation_runs)  # 80% for training
train_input = input_parameters[:train_size]
train_output = output_values[:train_size]

val_input = input_parameters[train_size:]
val_output = output_values[train_size:]

"""

STEP 5. Train and validate the neural network.

"""

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
