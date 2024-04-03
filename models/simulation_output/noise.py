"""

Add Gaussian noise to the simulated reflectance data.

Last updated on 29 March 2024 by Pirta Palola

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the simulated reflectance data
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
       'Jan2024_lognormal_priors/simulated_reflectance_no_noise.csv'
simulated_spectra = pd.read_csv(path)

# Create a list of wavelengths
wavelengths = list(simulated_spectra.columns)
wavelengths = [int(i) for i in wavelengths]
print(wavelengths)

# Check what the data looks like before adding noise
simulated_spectra_original = pd.read_csv(path)
no_noise = simulated_spectra_original.iloc[1]
print(no_noise)

# Define parameters
num_spectra = 30000  # number of spectra
num_wavelengths = 61  # 400-700nm at 5nm resolution

# Add Gaussian noise
for i in range(num_spectra):
    spectrum = simulated_spectra.iloc[i]  # Modify one spectrum at a time
    for wavelength in range(len(wavelengths)):
        std_dev = np.mean(spectrum) * 0.025
        noise = np.random.normal(0, std_dev, 1)  # Generate noise for each wavelength
        spectrum[wavelength] += noise  # Add noise to the current wavelength

# Print an example
noise_added = simulated_spectra.iloc[1]
print(noise_added)

# Plot an example of a spectrum before and after noise was added
plt.plot(wavelengths, no_noise, label="No noise")
plt.plot(wavelengths, noise_added, label="Noise added")
plt.legend()
plt.show()

# Save the results into a csv file
output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
              'Jan2024_lognormal_priors/simulated_reflectance_with_noise_0025.csv'
simulated_spectra.to_csv(output_path)
