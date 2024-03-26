"""

Add Gaussian noise to the simulated reflectance data.

Last updated on 26 March 2024 by Pirta Palola

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
wavelengths = [float(i) for i in wavelengths]
print(wavelengths)

# Check what the data looks like before adding noise
simulated_spectra_original = pd.read_csv(path)
print(simulated_spectra_original.iloc[10])
no_noise = simulated_spectra_original.iloc[10]

# Define parameters
num_spectra = 30000  # number of simulated spectra
num_wavelengths = 61  # 400-700nm at 5nm resolution
STR = 500  # signal-to-noise ration recommended by NASA for ocean color applications

# Generate unique random numbers for each spectrum
unique_random_numbers = np.random.rand(num_spectra, num_wavelengths)

# Add Gaussian noise to each wavelength for all spectra
for i in range(num_spectra):
    # Calculate standard deviation for current spectrum
    std_dev = 1 / STR * unique_random_numbers[i]

    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, num_wavelengths)

    # Add noise to current spectrum
    simulated_spectra.iloc[i] += noise

# Print an example
noise_added = simulated_spectra.iloc[10]
print(noise_added)

# Plot an example of a spectrum before and after noise was added
plt.plot(wavelengths, no_noise, label="No noise")
plt.plot(wavelengths, noise_added, label="Noise added")
plt.legend()
plt.show()

# Save the results into a csv file
output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
              'Jan2024_lognormal_priors/simulated_reflectance_with_noise_500STR.csv'
simulated_spectra.to_csv(output_path)
