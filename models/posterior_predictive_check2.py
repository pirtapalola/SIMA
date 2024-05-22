# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/'
                                    'simulated_reflectance_1000SNR_noise_sbc.csv')

# Pick one spectrum as an observation
observed = simulated_reflectance.iloc[0]

# Read the csv file containing the simulated reflectance data
simulated_reflectance_ppc = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/'
                                        'simulated_reflectance_no_noise.csv')

# Define your observed spectrum x_o (shape: [61])
x_o = np.array(observed)
print("Shape of x_o: ", x_o.shape)

# Define your simulated spectra from the posterior predictive distribution x_pp (shape: [1000, 61])
x_pp = np.array(simulated_reflectance_ppc)
print("Shape of x_pp: ", x_pp.shape)

# Create a list of wavelengths
wavelengths = []
for wavelength in range(400, 705, 5):
    wavelengths.append(wavelength)
wavelengths = np.array(wavelengths)


# Plot
def plot_percentiles(x, y, alpha_fill=0.3, **kwargs):
    """Plots the mean and 5th to 95th percentile band of y.

    Args:
        x (array): Shape (l,)
        y (array): Shape (n, l)
    """
    mean = np.mean(y, axis=0)
    perc_5 = np.percentile(y, 5, axis=0)
    perc_95 = np.percentile(y, 95, axis=0)

    (base_line,) = plt.plot(x, mean, **kwargs)
    kwargs["label"] = None
    kwargs["alpha"] = alpha_fill
    kwargs["facecolor"] = base_line.get_color()
    kwargs["edgecolor"] = None
    plt.fill_between(x, perc_5, perc_95, **kwargs)


# Mean Squared Error Calculation
mse_posterior = np.mean((np.mean(x_pp, axis=0) - x_o) ** 2)

# Plotting
plt.figure(figsize=(10, 5))

# Plot Posterior Predictive
plot_percentiles(wavelengths, x_pp, alpha_fill=0.3, label=f'Posterior Predictive (MSE={mse_posterior:.2f})',
                 color='purple')

# Plot Ground Truth (GT)
plt.plot(wavelengths, x_o, label='Ground Truth (GT)', color='red')

# Setting up the plot appearance
plt.xlim([wavelengths.min(), wavelengths.max()])
plt.ylim([x_pp.min() - 0.5, x_pp.max() + 0.5])
plt.axvline(x=0, linestyle='--', color='brown')

# Labels and legend
plt.xlabel('Wavelength')
plt.ylabel('Spectral Value')
plt.legend()

plt.show()
