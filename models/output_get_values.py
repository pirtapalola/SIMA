"""

Get the values of the parameters of each simulation run from the filenames.

"""

# Import libraries

import os
import pandas as pd


def extract_values_from_filename(filename):
    # Remove ".txt" from the filename
    filename = filename.replace(".txt", "")

    # Assuming the format "Mcoralbrown_00_00_021_461_672_100"
    parts = filename.split('_')

    # Extract parameter values from the filename
    try:
        water = int(parts[1]) / 1000.0
        phy = int(parts[2]) / 1000.0
        cdom = int(parts[3]) / 1000.0
        spm = int(parts[4]) / 1000.0
        wind = int(parts[5]) / 100.0
        depth = int(parts[6]) / 100.0
    except ValueError:
        return filename  # Return filename if there's an error

    return water, phy, cdom, spm, wind, depth


testing = extract_values_from_filename("Mcoralbrown_00_001_042_1166_779_1.txt")
print(testing)

# Load CSV file into a DataFrame
csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/' \
       'Methods/Methods_Ecolight/Dec2023_lognormal_priors/' \
       'Ecolight_parameter_combinations.csv'
df_csv = pd.read_csv(csv_file_path)

# Create a dictionary to store parameter combinations
param_dict = {}

# Populate the dictionary with parameter combinations from the CSV file
for index, row in df_csv.iterrows():
    filename = "_".join(str(round(value, 3) if col not in ['wind', 'depth'] else round(value, 2)).replace('.', '') for col, value in zip(df_csv.columns, row))
    param_dict[filename] = row.tolist()

# Create a list of all the filenames
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/' \
       'Methods/Methods_Ecolight/Dec2023_lognormal_priors/' \
       'EL_test_2_dec2023/EL_test_2_dec2023'
files = [f for f in os.listdir(path) if f.endswith('.txt')]

# Create a DataFrame to store the results
result_df = pd.DataFrame(columns=['water', 'phy', 'cdom', 'spm', 'wind', 'depth'])

# List to store problematic filenames
error_files = []

# Extract values from filenames and accumulate problematic filenames
for filename in files:
    values = extract_values_from_filename(filename)
    if isinstance(values, str):  # Check if values is a string (filename)
        error_files.append(values)

# Print all the filenames that caused errors
if error_files:
    print("Files with errors:")
    for filename in error_files:
        print(filename)

# Extract values from filenames and populate the result DataFrame
for filename in files:
    # Extract values directly from filename
    values = extract_values_from_filename(filename)
    if values:
        result_df.loc[len(result_df)] = values

# Display the resulting DataFrame
print(result_df)
