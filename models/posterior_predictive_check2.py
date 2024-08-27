"""

MODELS: Posterior predictive check (part 2): plotting the posterior predictive
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

STEP 1. Read the data.
STEP 2. Plot the posterior predictive.

Last updated on 27 August 2024

"""


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""STEP 1. Read the data."""

# Read the CSV files containing the simulated reflectance data
simulated_reflectance_ppc = pd.read_csv('data/x_data/simulated_reflectance_no_noise_ppc.csv')
simulated_reflectance = pd.read_csv('data/x_data/simulated_reflectance_evaluate.csv')

# Pick one spectrum as an observation
observed = simulated_reflectance.iloc[0]

# Define your observed spectrum x_o (shape: [61])
x_o = np.array(observed)
print("Shape of x_o: ", x_o.shape)

# Define your simulated spectra from the posterior predictive distribution x_pp (shape: [1000, 61])
x_pp = np.array(simulated_reflectance_ppc)
print("Shape of x_pp: ", x_pp.shape)

# Create a list of wavelengths
wavelengths = np.arange(400, 705, 5)
print("Shape of wavelengths: ", wavelengths.shape)

"""STEP 2. Plot the posterior predictive."""


def plot_percentiles(x, y, alpha_fill=0.3, **kwargs):
    """Plots the mean and 5th to 95th percentile band of y.

    Args:
        x (array): Shape (l,)
        y (array): Shape (n, l)
    """
    y = np.asarray(y)  # Ensure y is a numpy array
    mean = np.mean(y, axis=0)
    perc_5 = np.percentile(y, 5, axis=0)
    perc_95 = np.percentile(y, 95, axis=0)

    (base_line,) = plt.plot(x, mean, **kwargs)
    kwargs["label"] = None
    kwargs["alpha"] = alpha_fill
    kwargs["facecolor"] = base_line.get_color()
    kwargs["edgecolor"] = None
    plt.fill_between(x, perc_5, perc_95, **kwargs)


# Create a figure
plt.figure(figsize=(10, 8))

# Set font size
font = {'size': 12}

# using rc function
plt.rc('font', **font)

# Plot the posterior predictive
plot_percentiles(wavelengths, x_pp, alpha_fill=0.3, label=f'Posterior predictive',
                 color='lightgreen')

# Plot the ground-truth spectrum
plt.plot(wavelengths, x_o, label='Ground-truth', color='dodgerblue')

# Setting up the plot appearance
plt.xlim([wavelengths.min(), wavelengths.max()])
plt.ylim([x_pp.min() - 0.0001, x_pp.max() + 0.0001])
# plt.axvline(x=wavelengths[0], linestyle='--', color='brown')

# Labels and legend
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend(loc='upper left')

# Show the plot
plt.show()
