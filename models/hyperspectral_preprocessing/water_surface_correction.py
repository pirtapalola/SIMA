"""
PRE-PROCESSING HYPERSPECTRAL DATA II: Water surface correction
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Calculate just-above water reflectance from just-below water reflectance.
STEP 1. Read the data.
STEP 2. Calculate just-above water reflectance.

Last updated on 1 May 2024

"""

# Import libraries
import pandas as pd

"""STEP 1. Read the data."""

# Read the csv file containing the observation data
observation_path = 'data/field_data/interpolated_reflectance_tetiaroa_2022.csv'
obs_df = pd.read_csv(observation_path)
obs_df = obs_df.drop(columns=["wavelength"])
print(obs_df)
# Create a list of sample IDs
sample_IDs = list(obs_df.columns)
print(sample_IDs)

"""STEP 2. Calculate just-above water reflectance."""


# Define a function that applies the water surface correction
def water_surface_correction(sample_id):
    results = []
    for value in obs_df[sample_id]:
        result = (0.5*value)/(1-1.5*value)  # Lee et al. (1999)
        results.append(result)
    return results


# Create an empty list to store the results
Rrs_list = []

# Loop through the files
for i in sample_IDs:
    corrected_values = water_surface_correction(i)
    Rrs_list.append(corrected_values)

# Save the data in a csv file
Rrs_df = pd.DataFrame(Rrs_list)
Rrs_df = Rrs_df.transpose()
print(Rrs_df)
Rrs_df.to_csv("data/field_data/processed_reflectance_tetiaroa_2022.csv", index=False)
