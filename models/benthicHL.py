import pandas as pd

global_data = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                          'global_bottom_irradiance_reflectance.csv')
wavelength_data = global_data['wavelength']
coral_brown = global_data['coral_brown']
wavelength = [str(i) for i in wavelength_data]
cb = [str(i) for i in coral_brown]

# Define the path of the default bottom reflectance file
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/avg_coral.txt'


# hydrolight_file[65] = r'D:\HE53\data\User\microplastics\MPzdata.txt' + '\n'

def write_wavelength_file(wavelength_list):
    with open(path, 'w') as fp:
        for item in wavelength_list:
            fp.write(' ' + item + '   ' + '\n')


# write_wavelength_file(wavelength)

# Create empty lists to store data
lines = []
data_list = []

# Read the txt file
with open(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/avg_coral.txt', 'r') as fp1:
    lines = fp1.readlines()  # Add each line of the txt file as an element in the list

# Delete the first elements and the last elements of the list
data1 = lines[30:]
data = data1[:-61]

# Iterate through each element in the data list and replace the second part with the values from another list
coral_brown_list0 = [str(i)+'\n' for i in coral_brown]
coral_brown_list = coral_brown_list0[0::5]
modified_data = [f'{element.split()[0]}   {coral_brown_list[i]}' for i, element in enumerate(data)]
benthic_reflectance = [' ' + string for string in modified_data]

# Add new elements using the addition operator
modified_input_data = lines[:30] + benthic_reflectance + lines[-61:]

# Specify the file path
file_path = r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/coral_brown.txt'

# Write the list to the text file
with open(file_path, 'w') as fp2:
    for line in modified_input_data:
        fp2.write(line)

"""
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
