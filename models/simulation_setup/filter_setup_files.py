"""

Filter the set-up files to retrieve those that have NAP > 10 g/m3.
STEP 1. Read the parameter combinations from a csv file.
STEP 2. Store each row in the csv file as a tuple in a list.
STEP 3. Filter the list so that only the tuples that meet the condition NAP >= 10 g/m3 are included.
STEP 4. Create the file IDs associated with the selected tuples.
STEP 5. Use the list of file IDs to create a list of filepaths.
STEP 6. Modify the selected files.

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

# Convert each item in each tuple into a float
float_data_list = [tuple(map(float, row)) for row in data_list]
print(len(float_data_list))  # This corresponds to the number of simulation runs

"""STEP 3. Filter the list so that only the tuples that meet the condition [NAP] >= 10 g/m3 are included."""

# Create a new list that only contains the tuples that meet the condition.
filtered_list = [tup for tup in float_data_list if tup[3] >= 10]
print(len(filtered_list))  # This is the number of files that meet the condition

"""STEP 4. Create the file IDs associated with the selected tuples."""


# Use the same function that was used for naming the EL output files in the "inference" script.
def convert_tuple(tup):
    empty_string = ''
    for item in tup:
        element = round(item, 3)
        empty_string = empty_string + '_' + str(element)
        new_string = empty_string.replace('.', '')
    return new_string


# Apply the function to the filtered data list
string_id = []
for i in filtered_list:
    string_id.append(convert_tuple(i))

"""STEP 5. Use the list of file IDs to create a list of filepaths."""

# Specify path to the folder containing the set-up files
folder_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
              'Jan2024_lognormal_priors/setup'
file_ids = string_id

# Create an empty list to store the filepaths
file_paths = []

# Iterate through file IDs and match corresponding files in the folder
for file_id in file_ids:
    pattern = f"{folder_path}/*{file_id}*"
    matching_files = glob.glob(pattern)

    if matching_files:
        # Assuming there is only one matching file for each file ID
        file_paths.append(matching_files[0])

# Now, file_paths contains the paths to the files specified by file_ids.

"""STEP 6. Modify the selected files."""


def modify_setup_files(setup_file_path):
    with open(setup_file_path, 'r') as setup_file:
        lines = setup_file.readlines()
        # Update the selected lines
        lines[14] = r'..\data\defaults\astarmin_average.txt' + '\n'
        lines[22] = r'..\data\defaults\bstarmin_average.txt' + '\n'
    # Write the updated content back to the file
    with open(setup_file_path, 'w') as setup_file:
        setup_file.writelines(lines)


for selected_file in file_paths:
    modify_setup_files(selected_file)
