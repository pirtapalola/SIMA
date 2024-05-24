"""

Add Gaussian noise to theta.

Last updated on 24 May 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import numpy as np

# Read the csv file containing the inputs of each of the EcoLight simulation runs
simulator_input = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Ecolight_parameter_combinations_test.csv')

# Define the uncertainty
uncertainty_percentage = 5/100


# Function to add Gaussian noise based on the uncertainty
def add_noise(value, uncertainty):
    std_dev = value * uncertainty
    noise = np.random.normal(0, std_dev, size=value.shape)
    return value + noise


# Apply the function to each column
simulator_input['phy'] = add_noise(simulator_input['phy'].values, uncertainty_percentage)
simulator_input['cdom'] = add_noise(simulator_input['cdom'].values, uncertainty_percentage)
simulator_input['spm'] = add_noise(simulator_input['spm'].values, uncertainty_percentage)
simulator_input['wind'] = add_noise(simulator_input['wind'].values, uncertainty_percentage)
simulator_input['depth'] = add_noise(simulator_input['depth'].values, uncertainty_percentage)

# Display a sample of the modified DataFrame
print(simulator_input.head())

# Save the noisy data into a csv file
output_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Ecolight_parameter_combinations_noise4.csv'
simulator_input.to_csv(output_path, index=False)
