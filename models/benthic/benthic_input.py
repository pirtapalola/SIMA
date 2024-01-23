"""

Input bottom reflectance data into Hydrolight by replacing the default data file.

STEP 1. Upload the data.
STEP 2. Insert the bottom reflectance data into the default data file.
STEP 3. Save the new bottom reflectance input data file as a txt file.

Last updated on 05 December 2023 by Pirta Palola

"""

# Import libraries

import pandas as pd

"""STEP 1. Upload the data."""

global_data = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                          'global_bottom_irradiance_reflectance.csv')
wavelength_data = global_data['wavelength']
# coral_brown = global_data['coral_brown']
# coral_blue = global_data['coral_blue']
sand = global_data['sand']
wavelength = [str(i) for i in wavelength_data]
cb = [str(i) for i in sand]

# Create empty lists to store data
lines = []
data_list = []

# Read the txt file
with open(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/avg_coral.txt', 'r') as fp1:
    lines = fp1.readlines()  # Add each line of the txt file as an element in the list

"""Step 2. Insert the bottom reflectance data into the default data file."""

# Delete the first and the last elements of the list (these will not be modified)
data1 = lines[30:]  # These elements contain the file header and the wavelength range 300-395
data = data1[:-61]  # These elements contain the wavelength range 705-1000 and the last row of the text file

# Iterate through each element in the data list and replace the second part with the values from another list
coral_brown_list0 = [str(i)+'\n' for i in sand]
coral_brown_list = coral_brown_list0[0::5]
modified_data = [f'{element.split()[0]}   {coral_brown_list[i]}' for i, element in enumerate(data)]
benthic_reflectance = [' ' + string for string in modified_data]

# Add new elements using the addition operator
modified_input_data = lines[:30] + benthic_reflectance + lines[-61:]

"""STEP 3. Save the new bottom reflectance input data file as a txt file."""

# Specify the file path
file_path = r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/sand.txt'

# Write the list to the text file
with open(file_path, 'w') as fp2:
    for line in modified_input_data:
        fp2.write(line)

"""

Below is code for reading the reflectance data from the txt file and saving it in a csv file.

# Split each string in the list into two separate strings and store as sublists in a list
for i in range(len(data2)):
    data_list.append(data2[i].split("   "))

print(data_list)

# Create separate lists for the Hydrolight benthic reflectance data
wavelength_HL = []
reflectance_HL = []

for x in range(len(data1)):
    wavelength_HL.append(data_list[x][0].replace(' ', ''))  # Remove empty spaces

for x in range(len(data_list)):
    reflectance_HL.append(data_list[x][1].replace('\n', ''))  # Remove extra characters

# Add the Hydrolight benthic reflectance data to a pandas dataframe
hl_reflectance_df = pd.DataFrame({"wavelength": wavelength_HL, "avg_coral": reflectance_HL})
print(hl_reflectance_df)

# Save the data into a csv file.
hl_reflectance_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral_HL.csv')"""
