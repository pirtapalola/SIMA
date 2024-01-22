"""

Calculate just-above water reflectance from just-below water reflectance following Eq. A9 from Lee et al. (1999).
STEP 1. Read the data.
STEP 2. Calculate just-above water reflectance.

Last updated on 22 January 2024 by Pirta Palola

"""
# Import libraries
import pandas as pd

"""STEP 1. Read the data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/Methods_Ecolight/In_water_calibration_2022/smooth_surface_reflectance_2022.csv'
obs_df = pd.read_csv(observation_path)

# Create a list of sample IDs
sample_IDs = list(obs_df.columns)
print(sample_IDs)

"""STEP 2. Calculate just-above water reflectance."""


# Define a function that applies the water surface correction
def water_surface_correction(sample_id):
    results = []
    for value in obs_df[sample_id]:
        result = (0.5*value)/(1-1.5*value)
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
Rrs_df.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
              "In_water_calibration_2022/above_water_reflectance.csv")
