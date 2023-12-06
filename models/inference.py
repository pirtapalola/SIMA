"""

Generate the prior distributions of the theta parameters.
Sample the prior distributions to create parameterisations for the simulator.

STEP 1. Access the Ecolight setup file.
STEP 2. Create the priors.


Last updated on 05 December 2023 by Pirta Palola

"""

# Import libraries

import torch
import pandas as pd

"""STEP 1. Access the Ecolight setup file."""

# Define the path
PATH = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/'

# Open the file. Each line is saved as a string in a list.

with open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/final_setup/Icorals_final.txt') as f:
    concentrations = [line for line in f.readlines()]

"""STEP 2. Create the priors.
    Chl-a range: 0-10 (mg/m-3)
    CDOM range: 0-5 (m-1 at 440 nm)
    SPM range: 0-30 (g m-3)
    wind speed range: 0-10 m/s
    depth range: 0-20 m"""


# Define a function that generates and samples the prior assuming a Gamma distribution
def prior_gamma_distribution(parameter_name, alpha, beta):
    prior_gamma = torch.distributions.gamma.Gamma(torch.tensor([alpha]), torch.tensor([beta]))
    input_data = prior_gamma.sample((15000,))
    input_df = pd.DataFrame(input_data)
    input_df.to_csv(PATH + parameter_name + '_prior.csv')
    return input_data


# Define a function that generates and samples the prior assuming a uniform distribution
def prior_uniform_distribution(parameter_name, prior_min, prior_max):
    prior_uniform = torch.distributions.uniform.Uniform(low=prior_min, high=prior_max)
    input_data = prior_uniform.sample((10000,))
    input_df = pd.DataFrame(input_data)
    input_df.to_csv(PATH + parameter_name + '_prior.csv')
    return input_data


# Define a function that generates and samples the prior assuming a Gamma distribution
def prior_lognormal_distribution(parameter_name, loc, scale):
    prior_lognormal = torch.distributions.log_normal.LogNormal(torch.tensor([loc]), torch.tensor([scale]))
    input_data = prior_lognormal.sample((5000,))
    input_df = pd.DataFrame(input_data)
    input_df.to_csv(PATH + parameter_name + '_prior.csv')
    return input_data


# Apply the functions to generate the prior distributions that will be used for the simulations
prior_chl = prior_lognormal_distribution('chl', 0.4, 1.6)
prior_cdom = prior_lognormal_distribution('cdom', 0.2, 1.6)
prior_spm = prior_lognormal_distribution('spm', 1.3, 1.1)
prior_wind = prior_uniform_distribution('wind', 0.0, 10.0)
prior_depth = prior_uniform_distribution('depth', 0.0, 20.0)

prior_chl = prior_chl.tolist()
prior_cdom = prior_cdom.tolist()
prior_spm = prior_spm.tolist()
prior_wind = prior_wind.tolist()
prior_depth = prior_depth.tolist()

prior_chl_list = []
prior_cdom_list = []
prior_spm_list = []
prior_wind_list = []
prior_depth_list = []


def change_data_format(data_list, empty_list):
    for i in data_list:
        new_value = i[0]
        empty_list.append(round(new_value, 3))
    return empty_list


def change_data_format1(data_list1, empty_list1):
    for i in data_list1:
        empty_list1.append(round(i, 2))
    return empty_list1


prior_chl1 = change_data_format(prior_chl, prior_chl_list)
prior_cdom1 = change_data_format(prior_cdom, prior_cdom_list)
prior_spm1 = change_data_format(prior_spm, prior_spm_list)
prior_wind1 = change_data_format1(prior_wind, prior_wind_list)
prior_depth1 = change_data_format1(prior_depth, prior_depth_list)

combinations = []
for x in range(0, len(prior_chl)):
    new_combination = (0.0, prior_chl1[x], prior_cdom1[x],
                       prior_spm1[x], prior_wind1[x], prior_depth1[x])
    combinations.append(new_combination)

# Save the combinations in a csv file
df_combinations = pd.DataFrame(combinations, columns=['water', 'phy', 'cdom', 'spm', 'wind', 'depth'])
# print(df)
df_combinations.to_csv('C:/Users/pirtapalola/'
                       'Documents/DPhil/Chapter2/Methods/SBI_water_constituent_combinations.csv')

'''
# Check that the sampled prior distributions look realistic
chl = []
cdom = []
spm = []
wind = []
depth = []
parameters = [chl, cdom, spm, wind, depth]

for i in input_data:
    chl.append(i[0])
    cdom.append(i[1])
    spm.append(i[2])
    wind.append(i[3])
    depth.append(i[4])

for n in parameters:
    plt.hist(n, bins=100)
    plt.show()
'''


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
combination_ID = [i for i in range(0, len(prior_chl))]
dict_parameters = {k: HydroLightParameters(k) for k in combination_ID}


print(combinations[0])


# Create strings that contain information of water constituent combinations
# The strings will be used to name the files
def convert_tuple(tup):
    empty_string = ''
    for item in tup:
        element = round(item, 3)
        empty_string = empty_string + '_' + str(element)
        new_string = empty_string.replace('.', '')
    return new_string


string_id = []
for i in combinations:
    string_id.append(convert_tuple(i))

print(string_id[0])


# Define a function that applies the add_concentration() and add_id() functions
def add_data_to_dict(data_dictionary, num_str):
    data_dictionary[num_str].add_concentration('combination', pd.Series(combinations[num_str]))
    data_dictionary[num_str].add_id('id', string_id)


# Apply the function to all the sampling sites
for i in combination_ID:
    add_data_to_dict(dict_parameters, i)

# Check that each combination of concentrations can be accessed from the dictionary using the correct ID number
# print(dict_parameters[0].concentration['combination'])

# Check the lines of the input file that specify wind speed, depth, and benthic cover type
print(concentrations[51])  # The first element on line 51 specifies wind speed
print(concentrations[53])  # The last element on line 53 specifies depth
print(concentrations[61])  # Line 61 specifies the benthic cover type

# Create a list that only contains information on water constituents

combinations_water = []
combinations_wind = []
combinations_depth = []

for x in combinations:
    combinations_water.append(x[:-2])
    combinations_wind.append(x[-2])
    combinations_depth.append(x[-1])


def new_input_files(combination_iop, combination_w, combination_d, hydrolight_file, id_string):
    str0 = ', '.join(str(n) for n in combination_iop)
    # strcomb1 = ', '.join(str(n) for n in combination_w)
    str1 = str(combination_w)
    # strcomb2 = ', '.join(str(n) for n in combination_d)
    str2 = str(combination_d)
    hydrolight_file[2] = 'coral_' + id_string + '\n'  # rename the output file
    hydrolight_file[6] = str0 + ', \n'  # change the water constituent concentrations
    hydrolight_file[51] = (str1 + ', 1.34, 20, 35\n')  # change wind speed
    hydrolight_file[53] = ('0, 2, 0, ' + str2 + ', \n')  # change depth
    hydrolight_file[61] = 'avg_coral.txt' + '\n'  # specify the benthic reflectance
    hydrolight_file[12] = r'..\data\defaults\apstarchl.txt' + '\n'
    hydrolight_file[14] = r'..\data\defaults\astarmin_average.txt' + '\n'
    hydrolight_file[22] = r'..\data\defaults\bstarmin_average.txt' + '\n'
    hydrolight_file[65] = r'..\data\User\microplastics\MPzdata.txt' + '\n'

    # open file in write mode
    path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_sbi_setup/Icorals' + id_string + '_coral' + '.txt'
    with open(path, 'w') as fp:
        for item in hydrolight_file:
            fp.write(item)
    return hydrolight_file


# Create a list containing the combination IDs as strings
combination_ID_string = [str(i) for i in combination_ID]
"""
# Apply the function to all the data
for i in combination_ID:
    new_input_files(combinations_water[i], combinations_wind[i], combinations_depth[i], concentrations, string_id[i])

# Check that only the 6th, 51st, and 53rd lines were changed

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_sbi_setup/Icorals'
          + string_id[100] + '_coral' + '.txt', 'r')
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
f2.close()"""
