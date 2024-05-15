"""

Add Gaussian noise to the reflectance data.

Last updated on 3 May 2024 by Pirta Palola

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the simulated reflectance data
path = 'C:/Users/kell5379/Documents/Chapter2_May2024/PPC/simulated_reflectance_no_noise.csv'
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
no_noise = simulated_spectra_original.iloc[10]
print(no_noise)

# snr_df = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                     #'Jan2024_lognormal_priors/SNR_CHIME/SNR_w.csv')

# Add Gaussian noise
for i in range(len(simulated_spectra["400"])):
    spectrum = simulated_spectra.iloc[i]  # Modify one spectrum at a time
    # snr_w = snr_df[str(i)]
    # spectrum = simulated_spectra[i]
    for wavelength in range(len(wavelengths)):
        std_dev = np.sqrt(np.mean(spectrum ** 2)) / np.sqrt(1000)
        noise = np.random.normal(0, std_dev, 1)  # Generate noise for each wavelength
        spectrum[wavelength] += noise  # Add noise to the current wavelength

# Print an example
noise_added = simulated_spectra.iloc[10]
# noise_added = simulated_spectra["ONE05"]
print(noise_added)

# Plot an example of a spectrum before and after noise was added
plt.plot(wavelengths, no_noise, label="No noise")
plt.plot(wavelengths, noise_added, label="Noise added")
plt.legend()
plt.show()

# Save the results into a csv file
output_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/PPC/simulated_reflectance_1000SNR.csv'
simulated_spectra.to_csv(output_path, index=False)
