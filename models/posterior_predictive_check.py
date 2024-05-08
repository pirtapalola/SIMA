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
combinations = pd.DataFrame(posterior_samples)
water_column = [0 for number in range(len(combinations[0]))]
combinations.insert(0, 'water', pd.Series(water_column))

combinations_columns = ['water', 'phy', 'cdom', 'spm', 'wind', 'depth']
combinations.columns = combinations_columns
print(combinations)

# Save the posterior samples into a csv file
combinations.to_csv("C:/Users/kell5379/Documents/Chapter2_May2024/PPC/posterior_samples.csv", index=False)

"""STEP 3. Store the simulation parameterizations in a dictionary."""


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
combination_ID = [i for i in range(len(combinations["water"]))]
dict_parameters = {k: HydroLightParameters(k) for k in combination_ID}


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
for i in range(len(combinations)):
    string_id.append(convert_tuple(combinations.iloc[i]))

print(string_id[0])


# Define a function that applies the add_concentration() and add_id() functions
def add_data_to_dict(data_dictionary, num_str):
    data_dictionary[num_str].add_concentration('combination', pd.Series(combinations.iloc[num_str]))
    data_dictionary[num_str].add_id('id', string_id)


# Apply the function to all the sampling sites
for i in combination_ID:
    add_data_to_dict(dict_parameters, i)

# Check that each combination of concentrations can be accessed from the dictionary using the correct ID number
print(dict_parameters[10].concentration['combination'])
