"""

Create a csv file storing the EL output in a correct format.
STEP 1. Read the parameter combinations from a csv file.
STEP 2. Store each row in the csv file as a tuple in a list.
STEP 3. Create the file ID associated with each tuple.
STEP 4. Add the file IDs as a column in the dataframe.
STEP 5. Use the list of file IDs to create a list of filepaths
        so that each file can be accessed in the order defined by the list of file IDs.
STEP 6. Extract reflectance from the EL output files.

Last updated on 4 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import csv
import glob


"""STEP 1. Read the parameter combinations from a csv file."""

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'Jan2024_lognormal_priors/Ecolight_parameter_combinations.csv'
combinations = pd.read_csv(csv_file_path)


"""STEP 2. Store each row in the csv file as a tuple in a list."""

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

"""STEP 3. Create the file ID associated with each tuple."""


# Use the same function that was used for naming the EL output files in the "inference" script.
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

"""STEP 4. Add the file IDs as a column in the dataframe."""

combinations_df = combinations
combinations_df["filename"] = string_id

"""STEP 5. Use the list of file IDs to create a list of filepaths 
so that each file can be accessed in the order defined by the list of file IDs."""

# Specify path to the folder containing the output files
folder_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
              'Jan2024_lognormal_priors/simulated_dataset/simulated_dataset'
file_ids = combinations_df["filename"]

# Create an empty list to store the filepaths
file_paths = []

# Iterate through file IDs and match corresponding files in the folder
for file_id in file_ids:
    pattern = f"{folder_path}/*{file_id}*"
    matching_files = glob.glob(pattern)

    if matching_files:
        # Assuming there is only one matching file for each file ID
        file_paths.append(matching_files[0])

# Now, file_paths contains the paths to the files in the order specified by file_ids.
# Save the file_paths into a csv file.
df = pd.DataFrame(file_paths)
df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
          'Jan2024_lognormal_priors/file_paths_output_processing.csv')
