"""

Calculate just-above water reflectance from just-below water reflectance following Eq. A9 from Lee et al. (1999).

Last updated on 22 January 2024 by Pirta Palola

"""
# Import libraries
import pandas as pd

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/DPhil/' \
                   'Chapter2/Methods/Methods_Ecolight/In_water_calibration_2022/smooth_surface_reflectance_2022.csv'
obs_df = pd.read_csv(observation_path)

# Create a list of sample IDs
sample_IDs = list(obs_df.columns)
print(sample_IDs)