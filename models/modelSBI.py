"""

Simulation-based inference

STEP 1. Prepare the simulated data.
STEP 2. Split the data into training and validation sets.
STEP 3. Define the amortized neural network architecture.
STEP 4. Instantiate the amortized neural network.
STEP 5. Train and validate the neural network.
STEP 6. Evaluate the neural network with field-collected data.

"""

# Import libraries

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
STEP 1. Prepare the simulated data
    -The simulated data is split into input_parameters (5 parameters) and output_values (150 values).
    -After converting them to PyTorch tensors, they are concatenated into combined_input.

"""

# Define the simulated dataset
num_simulation_runs = 15000
num_parameters = 5
num_output_values = 150

# Specify the path to the folder that contains the simulated data
PATH = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/'

# Open the file. Each line is saved as a string in a list.
#with open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/final_setup/Icorals_final.txt') as f:
 #   concentrations = [line for line in f.readlines()]

presimulated_data = np.random.rand(num_simulation_runs, num_parameters + num_output_values)

input_parameters = presimulated_data[:, :num_parameters]
output_values = presimulated_data[:, num_parameters:]

"""STEP 2. Split the data into training and validation datasets."""

train_size = int(0.8 * num_simulation_runs)  # 80% for training
train_input = input_parameters[:train_size]
train_output = output_values[:train_size]

val_input = input_parameters[train_size:]
val_output = output_values[train_size:]


"""STEP 3. Define the amortized neural network architecture."""


class AmortizedPosterior(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(AmortizedPosterior, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim

        self.network_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
# Convert input parameters and training output to PyTorch tensors
train_input_tensor = torch.tensor(train_input, dtype=torch.float32)
train_output_tensor = torch.tensor(train_output, dtype=torch.float32)

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
        val_predictions = amortized_net(torch.tensor(val_input, dtype=torch.float32))
        val_loss = criterion(val_predictions, torch.tensor(val_output, dtype=torch.float32))
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
new_data = np.random.rand(1, num_parameters)  # Replace with your new data
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
posterior_estimate_new = amortized_net(new_data_tensor)

print("Estimated Posterior for New Data:", posterior_estimate_new)
