"""

Extract reflectance from the EL output files.
STEP 1. Read the data from the csv file.
STEP 2. Select which lines to extract from the txt files.
STEP 3. Save the extracted data into a pickled file.

"""

import pandas as pd
from io import StringIO
from multiprocessing import Pool
import pickle

"""STEP 1. Read the data from the csv file."""

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'Jan2024_lognormal_priors/file_paths_output_processing.csv'
file_paths_df = pd.read_csv(csv_file_path)
file_path_list = file_paths_df["file_paths"]

"""STEP 2. Select which lines to extract from the txt files."""


# A function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as output_file:
        text = output_file.read()
        lines = text.strip().split('\n')[629:780]  # Extract lines 630-780
        data = "\n".join(lines)  # Join the lines and create a StringIO object
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=r'\s+', header=None, dtype='float32')  # Read the data into a pandas DataFrame
        df = df.T  # Transpose the DataFrame to get the desired format
        df.columns = df.iloc[0]  # Set the first row as the header
        df = df.iloc[:3]
        df = df.drop([0, 2])
    return df


test = process_file(file_path_list[0])
print(test)

"""STEP 3. Save the extracted data into a pickled file."""

# Apply the function to all the files

"""
def main():
    output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Jan2024_lognormal_priors/' \
                  'simulated_reflectance.pkl'

    # Number of processes to use (adjust as needed)
    num_processes = 6

    # Create a Pool of processes
    with Pool(num_processes) as pool:
        data_frames = pool.map(process_file, file_paths)

    # Save the combined DataFrames to a pickle file
    with open(output_path, 'wb') as data_file:
        pickle.dump(data_frames, data_file)


# This condition checks if the script is being run directly as the main program
if __name__ == '__main__':
"""    """main()
"""