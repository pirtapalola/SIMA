"""

Get the values of the parameters of each simulation run from the filenames.

"""

# Import libraries
import pandas as pd
import csv

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'Dec2023_lognormal_priors/Ecolight_parameter_combinations.csv'
combinations = pd.read_csv(csv_file_path)

# Sort the DataFrame based on all the columns
#sorted_df = combinations.sort_values(by=['phy', 'cdom', 'spm', 'wind', 'depth'])



"""Read a csv file and store each row as a tuple in a list."""

data_list = []

with open(csv_file_path, 'r', newline='') as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        # Convert each row to a tuple and append it to the list
        row_tuple = tuple(row)
        data_list.append(row_tuple)

# Remove the first element of the list (a tuple containing the column names)
data_list = data_list[1:]

# Drop the last element from each tuple
modified_data_list = [tuple(row[:-1]) for row in data_list]

# Convert each item in each tuple into a float
float_data_list = [tuple(map(float, row)) for row in modified_data_list]


# Create the file ID associated with each tuple
def convert_tuple(tup):
    empty_string = ''
    for item in tup:
        element = round(item, 3)
        empty_string = empty_string + '_' + str(element)
        new_string = empty_string.replace('.', '')
    return new_string


string_id = []
for i in float_data_list:
    string_id.append(convert_tuple(i))

print("The first row of the dataframe: ", combinations.loc[0])
print("The first tuple: ", float_data_list[0])
print("The first file ID: ", string_id[0])

# Print the original and sorted DataFrames
print("Original DataFrame:")
print(combinations)

combinations_df = combinations
combinations_df["filename"] = string_id
# Sort the DataFrame based on the 'filename' column
sorted_df = combinations_df.sort_values(by='filename')

print("\nSorted DataFrame:")
print(sorted_df)

"""I have a list that contains file IDs for each file in a folder ("_00_001_016_052_524"). 
I want to read the files in that folder in the same order as the file ID are listed. 
The names of the files contain other things at the start and/or the end of the string, 
but each does contain the file ID somewhere within the string."""

import glob

folder_path = '/path/to/your/folder'  # Replace with the path to your folder
file_ids = ['_00_001_016_052_524', '_01_002_017_053_525', ...]  # Replace with your file IDs

file_paths = []

# Iterate through file IDs and match corresponding files in the folder
for file_id in file_ids:
    pattern = f"{folder_path}/*{file_id}*"
    matching_files = glob.glob(pattern)

    if matching_files:
        # Assuming there is only one matching file for each file ID
        file_paths.append(matching_files[0])

# Now, file_paths contains the paths to the files in the order specified by file_ids

# Now you can read the files in the specified order
for file_path in file_paths:
    with open(file_path, 'r') as file:
        # Perform operations on the file content as needed
        content = file.read()
        print(f"File content for {file_path}:\n{content}")

"""
# STEP 1: Create a list of all the files in the folder
# Create a list of all the filenames
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/' \
       'Methods/Methods_Ecolight/Dec2023_lognormal_priors/setup'
files = [f for f in os.listdir(path) if f.endswith('.txt')]


# STEP 2: Extract the input parameter values from the filenames


def extract_parameters(filename):
    # Remove the prefix "Icorals_" and split the remaining string by underscores
    filename = filename.replace("_coralbrown.txt", "")
    parameters_str = filename[len("Icorals_"):].split('_')

    # Insert decimal points for each parameter
    parameters = [float(f"{param[:-2]}.{param[-2:]}") for param in parameters_str]

    # Assign parameter names
    water, chl, cdom, spm, wind, depth = parameters

    return {
        'water': water,
        'chl': chl,
        'cdom': cdom,
        'spm': spm,
        'wind': wind,
        'depth': depth
    }


# Example usage:
filename = "Icorals_00_00_021_461_672_1006_coralbrown.txt"
parameters = extract_parameters(filename)
print(parameters)

# STEP 3. Extract the parameter values from all the filenames.

# Create a DataFrame to store the parameter values
result_df = pd.DataFrame(columns=['water', 'phy', 'cdom', 'spm', 'wind', 'depth'])

# Extract values from filenames and populate the result DataFrame
for filename in files:
    # Extract values directly from filename
    values = extract_parameters(filename)
    if values:
        result_df.loc[len(result_df)] = values

result_df = result_df.drop(columns="phy")
print(result_df)

# STEP: Compare the resulting dataframe to the original CSV file

combinations = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                           'Dec2023_lognormal_priors/Ecolight_parameter_combinations.csv')
print(combinations)

# Merge dataframes based on common columns 'phy', 'cdom', and 'spm'
merged_df = pd.merge(result_df, combinations, on=['water', 'cdom', 'spm', 'wind', 'depth'])

# Display the merged dataframe
print(merged_df)"""

"""
def extract_values_from_filename(filename):
    # Remove ".txt" from the filename
    filename = filename.replace(".txt", "")

    # Assuming the format "Mcoralbrown_00_00_021_461_672_100"
    parts = filename.split('_')

    # Extract parameter values from the filename
    try:
        water = int(parts[1]) / 100.0
        phy = int(parts[2]) / 100.0
        cdom = int(parts[3]) / 100.0
        spm = int(parts[4]) / 100.0
        wind = int(parts[5]) / 10.0
        depth = int(parts[6]) / 10.0
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
        param_dict[filename.replace(".txt", "")]['filename'] = filename

# Display the resulting DataFrame
print(result_df)

# Add the filename as a column to the CSV file
df_csv['filename'] = df_csv.apply(lambda row: param_dict.get("_".join(str(round(value, 3) if col not in ['wind', 'depth'] else round(value, 2)).replace('.', '') for col, value in zip(df_csv.columns, row)), {}).get('filename', ''), axis=1)
df_csv.to_csv(csv_file_path, index=False)"""
