# Import libraries
import pandas as pd

# Read the simulated reflectance data
path = 'C:/Users/kell5379/Documents/Chapter2_May2024/field_data2/field_TET22_1000SNR_noise.csv'
simulated_hyperspectral = pd.read_csv(path)

# Create a list of wavelengths
wavelengths = []
for wavelength in range(400, 705, 5):
    wavelengths.append(wavelength)
# print("Wavelengths: ", wavelengths)
print("Number of wavelengths in the hyperspectral dataset: ", len(wavelengths))

# Define satellite bands

# Planet SuperDove bands and associated SNR values
planet_coastal_blue = [str(n) for n in range(430, 455, 5)]
planet_blue = [str(n) for n in range(465, 520, 5)]
planet_green1 = [str(n) for n in range(515, 555, 5)]
planet_green2 = [str(n) for n in range(550, 590, 5)]
planet_yellow = [str(n) for n in range(600, 625, 5)]
planet_red = [str(n) for n in range(650, 685, 5)]
planet_rededge = [str(700)]
planet_bands = planet_coastal_blue + planet_blue + planet_green1 + planet_green2 + planet_yellow + planet_red + planet_rededge
print(planet_bands[1])
planet_SNR = [193, 170, 150, 154, 138, 63, 57]

# Sentinel-2 bands and associated SNR values
S2_b1 = [str(n) for n in range(430, 455, 5)]
S2_b2 = [str(n) for n in range(460, 525, 5)]
S2_b3 = [str(n) for n in range(545, 580, 5)]
S2_b4 = [str(n) for n in range(650, 685, 5)]
S2_b5 = [str(n) for n in range(695, 705, 5)]
S2_bands = S2_b1 + S2_b2 + S2_b3 + S2_b4 + S2_b5
S2_SNR = [129, 154, 168, 142, 117]

# Select only the columns corresponding to the selected wavelengths
planet_data = simulated_hyperspectral[planet_bands]
S2_data = simulated_hyperspectral[S2_bands]

#
# Calculate the mean of the selected columns for each row
planet_df = pd.DataFrame()
planet_df['coastal_blue'] = planet_data[planet_coastal_blue].mean(axis=1)
planet_df['blue'] = planet_data[planet_blue].mean(axis=1)
planet_df['green1'] = planet_data[planet_green1].mean(axis=1)
planet_df['green2'] = planet_data[planet_green2].mean(axis=1)
planet_df['yellow'] = planet_data[planet_yellow].mean(axis=1)
planet_df['red'] = planet_data[planet_red].mean(axis=1)
planet_df['rededge'] = planet_data[planet_rededge].mean(axis=1)
print("Planet SuperDove\n", planet_df)

S2_df = pd.DataFrame()
S2_df["b1"] = S2_data[S2_b1].mean(axis=1)
S2_df["b2"] = S2_data[S2_b2].mean(axis=1)
S2_df["b3"] = S2_data[S2_b3].mean(axis=1)
S2_df["b4"] = S2_data[S2_b4].mean(axis=1)
S2_df["b5"] = S2_data[S2_b5].mean(axis=1)
print("Sentinel-2\n", S2_df)

# Save the results into a csv file
output_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/downsampled/field_S2_1000SNR.csv'
S2_df.to_csv(output_path, index=False)
