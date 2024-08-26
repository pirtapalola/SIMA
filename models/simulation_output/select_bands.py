"""

SIMULATION OUTPUT IV: Select spectral bands.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Last updated on 26 August 2024

"""

# Import libraries
import pandas as pd

# Read the simulated reflectance data
path = "data/x_data/simulated_reflectance_100SNR.csv"
simulated_hyperspectral = pd.read_csv(path)
# simulated_hyperspectral = simulated_hyperspectral.drop(columns=["unique_ID"])  # Remove the "unique_ID" column.

# Create a list of wavelengths
wavelengths = []
for wavelength in range(400, 705, 5):
    wavelengths.append(wavelength)
# print("Wavelengths: ", wavelengths)
print("Number of wavelengths in the hyperspectral dataset: ", len(wavelengths))

# Define MicaSense Dual Camera System bands
b1 = [str(n) for n in range(440, 465, 5)]
b2 = [str(n) for n in range(460, 495, 5)]
b3 = [str(n) for n in range(525, 545, 5)]
b4 = [str(n) for n in range(545, 580, 5)]
b5 = [str(n) for n in range(645, 665, 5)]
b6 = [str(n) for n in range(660, 680, 5)]
b7 = [str(n) for n in range(695, 705, 5)]
micasense_bands = b1 + b2 + b3 + b4 + b5 + b6 + b7

# Select only the columns corresponding to the selected wavelengths
micasense_data = simulated_hyperspectral[micasense_bands]

# Calculate the mean of the selected columns for each row
micasense_df = pd.DataFrame()
micasense_df["b1"] = micasense_data[b1].mean(axis=1)
micasense_df["b2"] = micasense_data[b2].mean(axis=1)
micasense_df["b3"] = micasense_data[b3].mean(axis=1)
micasense_df["b4"] = micasense_data[b4].mean(axis=1)
micasense_df["b5"] = micasense_data[b5].mean(axis=1)
micasense_df["b6"] = micasense_data[b6].mean(axis=1)
micasense_df["b7"] = micasense_data[b7].mean(axis=1)

# Save the results into a csv file
output_path = 'data/x_data/multi_simulated_100SNR.csv'
micasense_df.to_csv(output_path, index=False)
