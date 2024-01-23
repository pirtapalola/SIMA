"""

Smoothen the reflectance data to get rid of noise.

Last updated on 21 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd

# Access the data
df = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                 "In_water_calibration_2022/processed_surface_reflectance_2022.csv")

# List of sample IDs
sample_IDs = list(df.columns)
sample_IDs = sample_IDs[1::]  # Delete the first element of the list
print(sample_IDs)

# Create an empty list
smooth_list = []

# Define the window size
window_size = 4

# Apply a moving average to smoothen the data
for sample in sample_IDs:
    smooth = df[sample].rolling(window=window_size, min_periods=1).mean()
    smooth_list.append(smooth)

# Create a dataframe
smooth_df = pd.DataFrame(smooth_list)
smooth_df = smooth_df.transpose()
print(smooth_df)
smooth_df.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                 "In_water_calibration_2022/smooth_surface_reflectance_2022.csv")
