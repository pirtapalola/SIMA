"""

MODELS: Checking the simulation output and identifying missing files.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

STEP 1. Specify paths.
STEP 2. Create a list of file IDs corresponding to the parameter combinations.
STEP 3. Check for matching files.
STEP 4. Create a new runlist.txt file for the missing simulations.
STEP 5. Copy the missing set-up files into a new folder.

Last updated on 27 August 2024

"""


# Import libraries
import pandas as pd
import csv
import os
import shutil

"""STEP 1. Specify paths."""

csv_file_path = 'data/simulation_setup/Ecolight_parameter_combinations.csv'
data_folder_path = 'data/simulated_dataset/'
file_runlist_path = 'data/simulation_setup/runlist_all.txt'
output_file_path = 'data/simulation_setup/runlist.txt'
all_setup_folder_path = 'data/setup_files/'

"""STEP 2. Create a list of file IDs corresponding to the parameter combinations."""

combinations = pd.read_csv(csv_file_path)
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

print("Number of parameter combinations: ", len(string_id))

"""STEP 3. Check for matching files."""


def get_file_names(folder_path):
    # Get the list of filenames in the specified folder
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return file_names


# Get the list of file names in the folder
file_list = get_file_names(data_folder_path)

# Matching files
matches = [string1 for string1 in file_list for string2 in string_id if string2 in string1]

# Print the number of matches
print(f"Number of matches: {len(matches)}")

# Using set to store unique values from list2 with no match in list1
non_match_values = set(string2 for string2 in string_id if not any(string2 in file_name for file_name in file_list))

# Print the unique values with no match
print("Unique values in list2 with no match in list1:", len(non_match_values))

"""STEP 4. Create a new runlist.txt file for the missing simulations."""

# Open the file and read lines
with open(file_runlist_path, 'r') as file:
    lines = file.readlines()

# Strip newline characters from each line and store in a list
lines = [line.strip() for line in lines]
print("Number of lines in the runlist_all.txt file: ", len(lines))

# Create a set of filenames from lines that match non_match_values
non_match_values_runlist = {file_name for file_name in lines if
                            any(string2 in file_name for string2 in non_match_values)}

# Convert the set to a list if needed
non_match_values_runlist = list(non_match_values_runlist)

# Print the list of filenames
print(len(non_match_values_runlist))

"""
# Write non_match_values_runlist to a new text file
with open(output_file_path, 'w') as output_file:
    for item in non_match_values_runlist:
        output_file.write(item + '')
"""

"""STEP 5. Copy the missing set-up files into a new folder."""


def copy_selected_files(source_folder, destination_folder, selected_files):
    # Ensure the destination folder exists, create it if necessary
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in selected_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        # Copy the file
        shutil.copy2(source_path, destination_path)


new_folder = 'data/missing_files/'

# Copy selected files from source to destination folder
copy_selected_files(all_setup_folder_path, new_folder, non_match_values_runlist)
