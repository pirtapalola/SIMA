from scipy.io import loadmat
import pandas as pd
import numpy as np

# Load the matlab file in python using scipy.
global_data = loadmat('C:/Users/pirtapalola/Documents/DPhil/'
                      'Chapter2/Hochberg/Hochberg_spectral_library_395-705_nm_2021-07-09.mat')

# Print the keys of the matlab file.
print(global_data.keys())

# Convert the data into numpy arrays and check their length.
classname_np = np.array(global_data['classname'])
reflectance_np = np.array(global_data['reflectance'])
wavelength_np = np.array(global_data['wavelength'])
print(len(classname_np))
print(len(reflectance_np))  # 5931 records
print(len(wavelength_np[0]))  # 311 wavelengths

# Organise the data into a pandas dataframe.
global_dataframe = pd.DataFrame(reflectance_np)
global_dataframe = global_dataframe.T  # Transpose the dataframe

# Create the column names.
classes_list = classname_np.tolist()

# Create a function that loops over the list so as to extract the classnames
# The structure of the list before extraction is list(list(array), ...(...))


def extract_1st(list_input):
    list1 = [item[0] for item in list_input]
    list2 = [item[0] for item in list1]
    return list2


classes = extract_1st(classes_list)  # Apply the function.

# Rename the columns using the class names.
global_dataframe = global_dataframe.set_axis(classes, axis="columns", copy=False)

# Add wavelengths as the first column in the dataframe.
global_dataframe.insert(0, "wavelength", wavelength_np[0], True)
print(global_dataframe)

# Save the data into a csv file.

global_dataframe.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/global_benthic_reflectance_hochberg.csv')
