"""

Calculate the mean value of each benthic cover type from the global reflectance dataset.

Last updated on 18 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd

# Read the csv file containing the simulated Rrs data into a pandas dataframe
global_df = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/'
                        'Chapter2/Methods/global_benthic_reflectance_hochberg.csv')

# Remove the dot and the number from each column name
global_df.columns = global_df.columns.str.replace(r'\.\d+$', '', regex=True)

# Calculate row average for columns with the same name
global_df_mean = global_df.groupby(global_df.columns, axis=1).mean()
print(global_df_mean.keys())

# Save to a csv file
global_df_mean.to_csv('C:/Users/pirtapalola/Documents/DPhil/'
                      'Chapter2/Methods/global_benthic_reflectance_hochberg_mean.csv')
