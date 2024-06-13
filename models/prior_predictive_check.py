# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file containing the simulated reflectance data
simulated_reflectance = pd.read_csv(
    'C:/Users/kell5379/Documents/Chapter2_May2024/simulated_reflectance_1000SNR_noise_prior_predictive.csv')
prior_samples = np.array(simulated_reflectance)
print("Shape of prior_samples: ", prior_samples.shape)  # Define the simulated spectra (shape: [5000, 61])

# Read the CSV file containing the simulated reflectance data
field_reflectance = pd.read_csv(
    'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/hp_field_1000SNR.csv')
field1 = np.array(field_reflectance["RIM04"])
field2 = np.array(field_reflectance["RIM05"])
# field3 = np.array(field_reflectance["GID_2505"])
# field4 = np.array(field_reflectance["GID_2506"])
print("Shape of field observation: ", field1.shape)  # Define the simulated spectra (shape: [61,])

# Create a list of wavelengths
wavelengths = np.arange(400, 705, 5)
print("Shape of wavelengths: ", wavelengths.shape)


# Plot Prior Predictive
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


plt.figure(figsize=(10, 5))
plot_percentiles(wavelengths, prior_samples, alpha_fill=0.3, label='Prior Predictive', color='lightblue')

# Plot field observations
plt.plot(wavelengths, field1, label='Field observations', color='dodgerblue')
plt.plot(wavelengths, field2, color='dodgerblue')
# plt.plot(wavelengths, field3, label='GLORIA field observations', color='darkblue')
# plt.plot(wavelengths, field4, color='darkblue')

# Setting up the plot appearance
plt.xlim([wavelengths.min(), wavelengths.max()])
plt.ylim([prior_samples.min() - 0.001, 0.05])
# plt.axvline(x=wavelengths[0], linestyle='--', color='brown')

# Labels and legend
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()

plt.show()
