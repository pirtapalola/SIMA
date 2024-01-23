
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""STEP 1. Read the data."""

# Define the file path
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/Methods_Ecolight/Dec2023_lognormal_priors/checks/Check0/check0.csv'

# Read the csv file
df = pd.read_csv(observation_path)

# Read the measured reflectance
x_measured = df['reflectance1']
x_simulated = df['simulation_run_no2239']

# Calculate dx
dx = x_measured[1]-x_measured[0]

# Apply the numpy gradient method to calculate the 1st derivative
measured_1derivative = np.gradient(x_measured, dx)
simulated_1derivative = np.gradient(x_simulated, dx)

# Plot the PDF for visualization
plt.plot(measured_1derivative, label='measured')
plt.plot(simulated_1derivative, label='simulated')
plt.legend()
plt.show()
