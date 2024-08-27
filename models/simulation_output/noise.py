"""

SIMULATION OUTPUT III: Adding Gaussian noise to the spectral data.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

STEP 1. Read the simulated reflectance data.
STEP 2. Add Gaussian noise.
STEP 3. Save the results into a csv file.

Last updated on 27 August 2024

"""

# Import libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

"""STEP 1. Read the simulated reflectance data."""

path = "data/simulated_data/simulated_reflectance_no_noise_train.csv"
simulated_spectra = pd.read_csv(path)

# wavelengths = simulated_spectra["wavelength"]
# simulated_spectra = simulated_spectra.drop(columns=["wavelength"])

# Create a list
# wavelengths = [443, 490, 531, 565, 610, 665, 700]
wavelengths = []
for wavelength in range(400, 705, 5):
    wavelengths.append(wavelength)
print("Wavelengths: ", wavelengths)
print("Number of wavelengths: ", len(wavelengths))
print("Number of reflectance spectra: ", len(simulated_spectra["400"]))

# Define parameters
# num_spectra = 10  # number of spectra
site_IDs = simulated_spectra.columns.tolist()
num_wavelengths = len(wavelengths)  # 400-700nm at 5nm resolution

# Check what the data looks like before adding noise
simulated_spectra_original = pd.read_csv(path)
# no_noise = simulated_spectra_original[site_IDs[0]]
# no_noise = simulated_spectra_original.iloc[10]
# print(no_noise)

"""STEP 2. Add Gaussian noise."""

# Add Gaussian noise
for i in range(len(simulated_spectra["400"])):
    spectrum = simulated_spectra.iloc[i]  # Modify one spectrum at a time
    # snr_w = snr_df[str(i)]
    # spectrum = simulated_spectra[i]
    for wavelength in range(len(wavelengths)):
        std_dev = np.sqrt(np.mean(spectrum ** 2)) / np.sqrt(100)
        noise = np.random.normal(0, std_dev, 1)  # Generate noise for each wavelength
        spectrum[wavelength] += noise  # Add noise to the current wavelength

# Print an example
# noise_added = simulated_spectra.iloc[10]
# noise_added = simulated_spectra["ONE05"]
# print(noise_added)

# Plot an example of a spectrum before and after noise was added
# plt.plot(wavelengths, no_noise, label="No noise")
# plt.plot(wavelengths, noise_added, label="Noise added")
# plt.legend()
# plt.show()

"""STEP 3. Save the results into a csv file."""

output_path = "data/x_data/simulated_reflectance_100SNR.csv"
simulated_spectra.to_csv(output_path, index=False)
