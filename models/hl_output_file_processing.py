"""

Process the HydroLight output files into a format compatible with the SBI workflow.
Create a single csv file that contains all the simulated Rrs data.

"""

import os
from multiprocessing import Pool
from io import StringIO
import pandas as pd


# Function to process a single file
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
        df = df.drop([0, 1])
    return df


def main():
    folder_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Dec2023_lognormal_priors/' \
                  'EL_test_2_dec2023/EL_test_2_dec2023'  # Folder path
    output_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Dec2023_lognormal_priors/' \
                  'simulated_rrs_dec23_lognorm.csv'

    # List all text files in the folder
    text_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Number of processes to use (adjust as needed)
    num_processes = 4

    # Create a Pool of processes
    with Pool(num_processes) as pool:
        data_frames = pool.map(process_file, text_files)

    # Concatenate DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_path, index=False)


# This condition checks if the script is being run directly as the main program
# (not imported as a module into another script).
# This ensures that the code within the main() function only runs when the script is executed directly.

if __name__ == '__main__':  # Check if the script is being run directly
    main()  # If so, call the main() function to start executing the script
