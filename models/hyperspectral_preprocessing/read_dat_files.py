"""

RADIOMETERS: TriOS RAMSES radiance and irradiance
TASK: Read the radiance/irradiance data exported from the MSDA_EX software and save into a csv file.

STEP 1. Access the datafiles.
STEP 2. Visualise the data.
STEP 3. Save the data into a csv file.

Last updated by Pirta Palola 23 January 2024

"""

# Import libraries.
import pandas as pd
import os
import matplotlib.pyplot as plt

""" STEP 1. Access the datafiles. """

# Define the sample ID
sample_ID = "OPU07"

# Create a list of all the filenames
path = "C:/Users/pirtapalola/Documents/Data/Fieldwork2023/Moorea2023/Hyperspectral/OPU07/"  # Define the file location
all_files = [f for f in os.listdir(path) if f.endswith('.dat')]  # Create a list of all the files in the folder

# Create a list with the calibrated data files (exclude the raw data files)
data_files = all_files[1::2]
print(all_files)
print(data_files)
print("The number of calibrated data files is " + str(len(all_files)) + "/2 = " + str((len(data_files))))


# Save the data into a list
def read_trios_files(file_name, path_name):
    new_data = pd.read_csv(path_name + file_name, sep=' ', usecols=[2])  # Select the column containing the data
    new_data = new_data.iloc[44:235]  # Select the rows containing the data
    new_data = new_data.astype('float64', copy=True, errors='raise')  # Transform the data into float
    new_data = new_data.divide(10)  # Divide each value by ten
    data_list = list(new_data[new_data.columns[0]])
    return data_list


# Save the wavelength range of the TriOS radiometers into a list
trios_wavelength_df = pd.read_csv("C:/Users/pirtapalola/Documents/Data/Fieldwork2023/trios_wavelength_range.csv")
trios_wavelength = trios_wavelength_df["wavelength"]
print(trios_wavelength)

# Create an empty pandas dataframe
trios_data = pd.DataFrame(columns=data_files)  # Name the columns with the filenames
trios_data["wavelength"] = trios_wavelength  # Add the wavelengths as a column

# Loop through all the data files and save the lists as columns in the pandas dataframe
for item in data_files:
    trios_data[item] = read_trios_files(item, path)

print(trios_data)

""" STEP 2. Visualise the data. """

# Create a plot with wavelength on the x-axis
trios_data.plot(x="wavelength", y=data_files, kind="line", figsize=(10, 10))

# Display the plot
plt.show()

""" STEP 3. Save the data into a csv file. """

trios_data.to_csv("C:/Users/pirtapalola/Documents/Data/Fieldwork2023/Moorea2023/Hyperspectral/Uncorrected/"
                  + sample_ID + "_2023_uncorrected.csv")
