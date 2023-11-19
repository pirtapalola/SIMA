import pandas as pd

tetiaroa_data = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral_tetiaroa.csv')
wavelength_data = tetiaroa_data['wavelength']
reflectance_data = tetiaroa_data['reflectance']
wavelength = [str(i) for i in wavelength_data]
reflectance = [str(i) for i in reflectance_data]

# open file in write mode
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral.txt'


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
with open(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral.txt', 'r') as fp:
    lines = fp.readlines()  # Add each line of the txt file as an element in the list

# Delete the first 6 elements and the last element of the list
data = lines[5:]
data = data[:-1]

# Split each string in the list into two separate strings and store as sublists in a list
for i in range(len(data)):
    data_list.append(data[i].split("   "))

# Create separate lists for the Hydrolight benthic reflectance data
wavelength_HL = []
reflectance_HL = []

for x in range(len(data)):
    wavelength_HL.append(data_list[x][0].replace(' ', ''))  # Remove empty spaces

for x in range(len(data_list)):
    reflectance_HL.append(data_list[x][1].replace('\n', ''))  # Remove extra characters

# Add the Hydrolight benthic reflectance data to a pandas dataframe
hl_reflectance_df = pd.DataFrame({"wavelength": wavelength_HL, "avg_coral": reflectance_HL})
print(hl_reflectance_df)

# Save the data into a csv file.
hl_reflectance_df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral_HL.csv')
