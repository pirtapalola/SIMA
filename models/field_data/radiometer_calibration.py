"""

RADIOMETERS: TriOS RAMSES radiance and irradiance
TASK: Apply the in-water calibration factors to the raw radiance/irradiance data exported from the MSDA_EX software.

STEP 1. Access the datafiles.
STEP 2. Visualise the data.
STEP 3. Save the data into a csv file.

Last updated by Pirta Palola 15 November 2023.

"""

# Import libraries.
import pandas as pd
import os
import matplotlib.pyplot as plt

""" STEP 1. Access the raw data files. """

# Create a list of all the filenames
path = 'C:/Users/kell5379/Documents/Code/example_data/RIM2022/'  # Define the file location
all_files = [f for f in os.listdir(path) if f.endswith('.dat')]  # Create a list of all the files in the folder

# Create a list with the raw data files (exclude the calibrated data files)
raw_data_files = all_files[::2]
print("The number of raw data files is " + str(len(all_files)) + "/2 = " + str((len(raw_data_files))))

# Save the wavelength range of the TriOS radiometers into a list
trios_wavelength_df = pd.read_csv("C:/Users/kell5379/ocean-optics/data/trios_wavelength_range.csv")  # Read csv
trios_wavelength = list(trios_wavelength_df[trios_wavelength_df.columns[0]])  # Transform into a list
print(trios_wavelength)


# Save the data into a list
def read_trios_files(file_name, path_name, calibration_factors):
    new_data = pd.read_csv(path_name + file_name, sep=' ', usecols=[2])  # Select the column containing the data
    new_data = new_data.iloc[44:235]  # Select the rows containing the data
    new_data = new_data.astype('float64', copy=True, errors='raise')  # Transform the data into float
    new_data = new_data.divide(10)  # Divide each value by ten
    data_list = list(new_data[new_data.columns[0]])
    [a * b for a, b in zip(data_list, cal_aq)]
    return data_list


# Save the wavelength range of the TriOS radiometers into a list
trios_wavelength_df = pd.read_csv("C:/Users/kell5379/ocean-optics/data/trios_wavelength_range.csv")  # Read csv
trios_wavelength = list(trios_wavelength_df[trios_wavelength_df.columns[0]])  # Transform into a list
print(trios_wavelength)

# Create an empty pandas dataframe
trios_data = pd.DataFrame(columns=data_files)  # Name the columns with the filenames
trios_data["wavelength"] = trios_wavelength  # Add the wavelengths as a column

# Loop through all the data files and save the lists as columns in the pandas dataframe
for item in data_files:
    trios_data[item] = read_trios_files(item, path)

print(trios_data)