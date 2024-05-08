"""

Posterior predictive check
STEP 1. Draw samples from the posterior.
STEP 2. Save the samples into a csv file.

Last updated on 8 May 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import pickle
import torch
from sbi.analysis import pairplot

"""STEP 1. Load the posterior and the data."""

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Noise_1000SNR/Noise_1000SNR/"
          "loaded_posteriors/loaded_posterior17.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)


# Open the file. Each line is saved as a string in a list.
with open('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/Icorals_final.txt') as f:
    concentrations = [line for line in f.readlines()]

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/'
                                    'simulated_reflectance_1000SNR_noise_sbc.csv')

# Pick one spectrum as an observation
x_o = simulated_reflectance.iloc[0]


"""STEP 2. Draw samples from the posterior."""

loaded_posterior.set_default_x(x_o)

# Draw theta samples from the posterior
posterior_samples = loaded_posterior.sample((5000,))

# Print the posterior samples
# print(posterior_samples)

# Plot the posterior samples
_ = pairplot(
    samples=posterior_samples,
    limits=[[0, 7], [0, 2.5], [0, 30], [0, 20], [0, 20]],
    points_colors=["red", "red", "red", "red", "red"],
    figsize=(8, 8),
    labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
    offdiag="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=20)
)

# Write the posterior samples into a pandas dataframe
df = pd.DataFrame(posterior_samples)
water_column = [0 for number in range(len(df[0]))]
df.insert(0, 'water', pd.Series(water_column))

combinations_columns = ['water', 'phy', 'cdom', 'spm', 'wind', 'depth']
df.columns = combinations_columns
print(df)

# Save the posterior samples into a csv file
df.to_csv("C:/Users/kell5379/Documents/Chapter2_May2024/PPC/posterior_samples.csv", index=False)

"""STEP 3. Store the simulation parameterizations in a dictionary."""

prior_chl1 = df["phy"]
prior_cdom1 = df["cdom"]
prior_spm1 = df["spm"]
prior_wind1 = df["wind"]
prior_depth1 = df["depth"]

combinations = []
for x in range(0, len(df["phy"])):
    new_combination = (0.0, prior_chl1[x], prior_cdom1[x],
                       prior_spm1[x], prior_wind1[x], prior_depth1[x])
    combinations.append(new_combination)

# Save the combinations in a csv file
df_combinations = pd.DataFrame(combinations, columns=['water', 'phy', 'cdom', 'spm', 'wind', 'depth'])
# print(df)
df_combinations.to_csv('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/Ecolight_parameter_combinations_ppc.csv')


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
combination_ID = [i for i in range(0, len(combinations))]
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
print(concentrations[42])  # The first element on line 51 specifies wind speed
print(concentrations[44])  # The last element on line 53 specifies depth
print(concentrations[52])  # Line 61 specifies the benthic cover type

# Create a list that only contains information on water constituents

combinations_water = []
combinations_wind = []
combinations_depth = []

for x in combinations:
    combinations_water.append(x[:-2])
    combinations_wind.append(x[-2])
    combinations_depth.append(x[-1])

"""STEP 4. Write the new Ecolight set-up files."""


def new_input_files(combination_iop, combination_w, combination_d, hydrolight_file, id_string):
    str0 = ', '.join(str(round(n, 3)) for n in combination_iop)
    # strcomb1 = ', '.join(str(n) for n in combination_w)
    str1 = str(round(combination_w, 3))
    # strcomb2 = ', '.join(str(n) for n in combination_d)
    str2 = str(round(combination_d, 3))
    hydrolight_file[2] = 'coralbrown' + id_string + '\n'  # rename the output file
    hydrolight_file[6] = str0 + ', \n'  # change the water constituent concentrations
    hydrolight_file[42] = (str1 + ', 1.34, 20, 35\n')  # change wind speed
    hydrolight_file[44] = ('0, 2, 0, ' + str2 + ', \n')  # change depth
    hydrolight_file[52] = 'coral_brown.txt' + '\n'  # specify the benthic reflectance
    hydrolight_file[12] = r'..\data\defaults\apstarchl.txt' + '\n'
    hydrolight_file[14] = r'..\data\defaults\astarmin_calcareoussand.txt' + '\n'
    hydrolight_file[22] = r'..\data\defaults\bstarmin_calcareoussand.txt' + '\n'
    hydrolight_file[56] = r'..\data\User\microplastics\MPzdata.txt' + '\n'

    # open file in write mode
    path = 'C:/Users/kell5379/Documents/Chapter2_May2024/PPC/' \
           'setup/Icorals' \
           + id_string + '_coralbrown' + '.txt'
    with open(path, 'w') as fp:
        for item in hydrolight_file:
            fp.write(item)
    return hydrolight_file


# Create a list containing the combination IDs as strings
combination_ID_string = [str(i) for i in combination_ID]

# Apply the function to all the data
for i in combination_ID:
    new_input_files(combinations_water[i], combinations_wind[i], combinations_depth[i], concentrations, string_id[i])

# Check that only the 6th, 51st, and 53rd lines were changed

# reading files
f1 = open('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/'
          'setup/Icorals'
          + string_id[1] + '_coralbrown' + '.txt', 'r')
f2 = open('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/'
          'Icorals_final.txt', 'r')

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
