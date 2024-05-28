# Import libraries
import pandas as pd

# Read the simulated reflectance data
path = ('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
        'field_surface_reflectance_1000SNR_transposed.csv')
simulated_hyperspectral = pd.read_csv(path)
simulated_hyperspectral = simulated_hyperspectral.drop(columns=["unique_ID"])  # Remove the "unique_ID" column.

# Create a list of wavelengths
wavelengths = []
for wavelength in range(400, 705, 5):
    wavelengths.append(wavelength)
# print("Wavelengths: ", wavelengths)
print("Number of wavelengths in the hyperspectral dataset: ", len(wavelengths))

# Define satellite bands

# Planet SuperDove bands and associated SNR values
planet_coastal_blue = [str(n) for n in range(435, 460, 5)]
planet_blue = [str(n) for n in range(465, 530, 5)]
planet_green1 = [str(n) for n in range(510, 550, 5)]
planet_green2 = [str(n) for n in range(550, 590, 5)]
planet_yellow = [str(n) for n in range(600, 625, 5)]
planet_red = [str(n) for n in range(650, 685, 5)]
planet_rededge = [str(n) for n in range(695, 705, 5)]
planet_bands = (planet_coastal_blue + planet_blue + planet_green1 + planet_green2
                + planet_yellow + planet_red + planet_rededge)
print(planet_bands[1])
planet_SNR = [193, 170, 150, 154, 138, 63, 57]

# Sentinel-2 bands and associated SNR values
S2_b1 = [str(n) for n in range(430, 460, 5)]
S2_b2 = [str(n) for n in range(450, 550, 5)]
S2_b3 = [str(n) for n in range(540, 590, 5)]
S2_b4 = [str(n) for n in range(645, 690, 5)]
S2_b5 = [str(n) for n in range(695, 705, 5)]
S2_bands = S2_b1 + S2_b2 + S2_b3 + S2_b4 + S2_b5
S2_SNR = [129, 154, 168, 142, 117]

# MicaSense Dual Camera System bands
b1 = [str(n) for n in range(440, 465, 5)]
b2 = [str(n) for n in range(460, 495, 5)]
b3 = [str(n) for n in range(525, 545, 5)]
b4 = [str(n) for n in range(545, 580, 5)]
b5 = [str(n) for n in range(645, 665, 5)]
b6 = [str(n) for n in range(660, 680, 5)]
b7 = [str(n) for n in range(695, 705, 5)]
micasense_bands = b1 + b2 + b3 + b4 + b5 + b6 + b7

# Select only the columns corresponding to the selected wavelengths
planet_data = simulated_hyperspectral[planet_bands]
S2_data = simulated_hyperspectral[S2_bands]
micasense_data = simulated_hyperspectral[micasense_bands]

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

micasense_df = pd.DataFrame()
micasense_df["b1"] = micasense_data[b1].mean(axis=1)
micasense_df["b2"] = micasense_data[b2].mean(axis=1)
micasense_df["b3"] = micasense_data[b3].mean(axis=1)
micasense_df["b4"] = micasense_data[b4].mean(axis=1)
micasense_df["b5"] = micasense_data[b5].mean(axis=1)
micasense_df["b6"] = micasense_data[b6].mean(axis=1)
micasense_df["b7"] = micasense_data[b7].mean(axis=1)
print("MicaSense\n", S2_df)

# Save the results into a csv file
output_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/micasense_field_1000SNR.csv'
micasense_df.to_csv(output_path, index=False)
