"""This code creates input files for HydroLight simulations with different water constituent concentrations"""

import pandas as pd
import itertools

# Open the file. Each line is saved as a string in a list.

with open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/final_setup/Icorals_final.txt') as f:
    concentrations = [line for line in f.readlines()]

# Print the line that specifies water constituent concentrations.
# 1st element: water
# 2nd element: phytoplankton
# 3rd element: CDOM
# 4th element: SPM

# Define lists that contain the different concentrations of each water constituent.
water = [0.0]
phy = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
cdom = [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5]
spm = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30]

# Create all the possible combinations of water constituent concentrations.
combinations = list(itertools.product(water, phy, cdom, spm))
print(type(combinations[0]))  # print the number of combinations
# Create a list that contains an ID number for each combination 1, 2, ..., n
combination_ID = [i for i in range(0, len(combinations))]

# Save the combinations in a csv file
df = pd.DataFrame(combinations, columns=['water', 'phy', 'cdom', 'spm'])
# print(df)
# df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/water_constituent_combinations.csv')


# Create a new class
class HydroLightParameters:
    def __init__(self, name):
        self.name = name
        self.concentration = {}
        self.id = {}

# Add each combination of water constituent concentrations to the corresponding combination ID

    def add_concentration(self, measurement_id, data):
        if measurement_id in self.concentration.keys():
            self.concentration[measurement_id] = \
                pd.concat([self.concentration[measurement_id], data])

        else:
            self.concentration[measurement_id] = data
            self.concentration[measurement_id].name = measurement_id

    def add_id(self, measurement_id, data):
        if measurement_id in self.id.keys():
            self.id[measurement_id] = \
                pd.concat([self.id[measurement_id], data])

        else:
            self.id[measurement_id] = data


# Create a dictionary to store all the IDs and the corresponding concentration data
dict_parameters = {k: HydroLightParameters(k) for k in combination_ID}


# Create strings that contain information of water constituent combinations
# The strings will be used to name the files
def convert_tuple(tup):
    empty_string = ''
    for item in tup:
        empty_string = empty_string + '_' + str(item)
        new_string = empty_string.replace('.', '')
    return new_string


string_id = []
for i in combinations:
    string_id.append(convert_tuple(i))

print(string_id)


# Define a function that applies the add_concentration() and add_id() functions
def add_data_to_dict(data_dictionary, num_str):
    data_dictionary[num_str].add_concentration('combination', pd.Series(combinations[num_str]))
    data_dictionary[num_str].add_id('id', string_id)


# Apply the function to all the sampling sites
for i in combination_ID:
    add_data_to_dict(dict_parameters, i)

# Check that each combination of concentrations can be accessed from the dictionary using the correct ID number
# print(dict_parameters[0].concentration['combination'])

# Define the different wind speeds and depths
wind = [0, 2, 5]
depth = [1, 2, 3, 5]
print(concentrations[51])  # The first element on line 51 specifies wind speed
print(concentrations[53])  # The last element on line 53 specifies depth

def new_input_files(combination, hydrolight_file, id_string):
    strcomb = ', '.join(str(n) for n in combination)
    str0 = strcomb + ', \n'
    hydrolight_file[6] = str0
    wind_speed = str(5)  # Change this value to change wind speed
    water_depth = str(5)  # Change this value to change depth
    hydrolight_file[51] = (wind_speed + ', 1.34, 20, 35\n')
    hydrolight_file[53] = ('0, 2, 0, ' + water_depth + ', \n')
    # open file in write mode
    path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/Icorals' + id_string \
           + '_' + wind_speed + '_' + water_depth + '.txt'
    with open(path, 'w') as fp:
        for item in hydrolight_file:
            fp.write(item)
    return hydrolight_file


# Create a list containing the combination IDs as strings
combination_ID_string = [str(i) for i in combination_ID]

# Apply the function to all the data
for i in combination_ID:
    new_input_files(dict_parameters[i].concentration['combination'], concentrations, string_id[i])

# Check that only the 6th line was changed

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/Icorals_00_001_001_001_5_5.txt', 'r')
f2 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/final_setup/Icorals_final.txt', 'r')

f1_data = f1.readlines()
f2_data = f2.readlines()
num_lines = (len(f1_data))

for x in range(0, num_lines):
    # compare each line one by one
    if f1_data[x] != f2_data[x]:
        print("Difference detected - Line ", x, ":")
        print("\tFile 1:", f1_data[x], end='')
        print("\tFile 2:", f2_data[x], end='')

# close the files
f1.close()
f2.close()
