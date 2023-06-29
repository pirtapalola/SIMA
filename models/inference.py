import torch
import pandas as pd

PATH = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/'

'''Create the priors
    Chl-a range: 0-10 (mg/m-3)
    CDOM range: 0-5 (m-1 at 440 nm)
    SPM range: 0-30 (g m-3)
    wind speed range: 0-10 m/s
    depth range: 0-20 m'''

# Number of dimensions in the parameter space
num_dim = 5


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
    input_data = prior_uniform.sample((15000,))
    input_df = pd.DataFrame(input_data)
    input_df.to_csv(PATH + parameter_name + '_prior.csv')
    return input_data


# Apply the functions to generate the prior distributions that will be used for the simulations
prior_chl = prior_gamma_distribution('chl', 1.1, 1.1)
prior_cdom = prior_gamma_distribution('cdom', 1.2, 3.0)
prior_spm = prior_gamma_distribution('spm', 3.0, 0.6)
prior_wind = prior_uniform_distribution('wind', 0.0, 10.0)
prior_depth = prior_uniform_distribution('depth', 0.0, 20.0)

prior_chl = prior_chl.tolist()
prior_cdom = prior_cdom.tolist()
prior_spm = prior_spm.tolist()

prior_chl_list = []
prior_cdom_list = []
prior_spm_list = []


def change_data_format(data_list, empty_list):
    for i in data_list:
        new_value = i[0]
        empty_list.append(new_value)
    return empty_list


prior_chl1 = change_data_format(prior_chl, prior_chl_list)
prior_cdom1 = change_data_format(prior_cdom, prior_cdom_list)
prior_spm1 = change_data_format(prior_spm, prior_spm_list)

combinations = []
for x in range(0, len(prior_chl)):
    new_combination = (0.0, prior_chl1[x], prior_cdom1[x], prior_spm1[x])
    combinations.append(new_combination)

print(combinations)
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
