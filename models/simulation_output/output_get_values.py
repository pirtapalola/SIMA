"""

Create a csv file storing the EL output in a correct format.
STEP 1. Read the parameter combinations from a csv file.
STEP 2. Store each row in the csv file as a tuple in a list.
STEP 3. Create the file ID associated with each tuple.
STEP 4. Add the file IDs as a column in the dataframe.
STEP 5. Use the list of file IDs to create a list of filepaths
        so that each file can be accessed in the order defined by the list of file IDs.
STEP 6. Extract reflectance from the EL output files.

"""

# Import libraries
import pandas as pd
import csv
import glob
from io import StringIO
import os
from multiprocessing import Pool

"""STEP 1. Read the parameter combinations from a csv file."""

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'check_simulation_Feb2024/Ecolight_parameter_combinations.csv'
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
              'check_simulation_Feb2024/el_setup3'
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

"""STEP 6. Extract reflectance from the EL output files."""


# A function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        lines = text.strip().split('\n')[629:780]  # Extract lines 630-780
        data = "\n".join(lines)  # Join the lines and create a StringIO object
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=r'\s+', header=None)  # Read the data into a pandas DataFrame
        df = df.T  # Transpose the DataFrame to get the desired format
        df.columns = df.iloc[0]  # Set the first row as the header
        df = df.iloc[:3]
        df = df.drop([0, 2])
    return df


# Apply the function to all the files
def main():
    output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/check_simulation_Feb2024/' \
                  'simulated_rrs_check_simulation.csv'

    # Number of processes to use (adjust as needed)
    num_processes = 4

    # Create a Pool of processes
    with Pool(num_processes) as pool:
        data_frames = pool.map(process_file, file_paths)

    # Concatenate DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_path, index=False)


# This condition checks if the script is being run directly as the main program
# (not imported as a module into another script).
# This ensures that the code within the main() function only runs when the script is executed directly.

if __name__ == '__main__':  # Check if the script is being run directly
    main()  # If so, call the main() function to start executing the script
