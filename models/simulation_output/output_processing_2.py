"""

Extract reflectance from the EL output files.
STEP 1. Read the data from the csv file.
STEP 2. Select which lines to extract from the txt files.
STEP 3. Save the extracted data into a csv file.

Last updated on 30 April 2024 by Pirta Palola

"""

import pandas as pd
from io import StringIO
from multiprocessing import Pool

"""STEP 1. Read the data from the csv file."""

csv_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                'Simulated_evaluation_dataset/file_paths_output_processing.csv'
file_paths_df = pd.read_csv(csv_file_path)
file_path_list = file_paths_df["file_paths"]

"""STEP 2. Select which lines to extract from the txt files."""


# A function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as output_file:
        text = output_file.read()
        lines = text.strip().split('\n')[273:335]  # Extract lines 630-780
        data = "\n".join(lines)  # Join the lines and create a StringIO object
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=r'\s+', header=None)  # Read the data into a pandas DataFrame
        df = df.T  # Transpose the DataFrame to get the desired format
        df.columns = df.iloc[0]  # Set the first row as the header
        df = df.iloc[:3]  # These two lines extract Rrs data
        df = df.drop([0, 2])
        # df = df.iloc[:4] These two lines extract Ed data
        # df = df.drop([0, 1, 3])
    return df


"""STEP 3. Save the extracted data into a csv file."""

# Apply the function to all the files.


def main():
    output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                  'Simulated_evaluation_dataset/simulated_reflectance_no_noise.csv'
    # Number of processes to use (adjust as needed)
    num_processes = 4
    # Create a Pool of processes
    with Pool(num_processes) as pool:
        data_frames = pool.map(process_file, file_path_list)
    # Concatenate DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)
    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_path, index=False)


# This condition checks if the script is being run directly as the main program
# (not imported as a module into another script).
if __name__ == '__main__':  # Check if the script is being run directly
    main()  # If so, call the main() function to start executing the script

"""
# Save the results in a pickled file
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