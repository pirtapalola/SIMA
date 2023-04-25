"""This code calculates:
- The backscattering coefficient of phytoplankton following Eq. 14 (Theenathayalan & Shanmugam, 2021)
- Total backscattering and absorption"""

import pandas as pd


# Read the csv files
chl = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                  'Plymouth_iop/plymouth_iop_model/chlorophyll.csv')
chl_specific_bb = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                              'Plymouth_iop/plymouth_iop_model/bb_ph_star.csv')
aY_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                    'Plymouth_iop/plymouth_iop_model/plymouth_cdom_absorption.csv')
aB_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                    'Plymouth_iop/plymouth_iop_model/plymouth_ph_absorption.csv')
aS_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                    'Plymouth_iop/plymouth_iop_model/plymouth_nap_absorption.csv')
bbB_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                     'Plymouth_iop/plymouth_iop_model/plymouth_ph_backscatter.csv')
bbS_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                     'Plymouth_iop/plymouth_iop_model/plymouth_nap_backscatter.csv')
bbW_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                     'Plymouth_iop/plymouth_iop_model/Bwater.csv')
aW_df = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                    'Plymouth_iop/plymouth_iop_model/Awater.csv')
depth_data = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                         'Plymouth_iop/plymouth_iop_model/depth_data.csv')

# Create a list of site IDs
sites = chl['Unique_ID'].tolist()  # ...site IDs

"""Backscattering coefficient of phytoplankton"""

# Create a list of...
chl_list = chl['chl'].tolist()  # ...chl-a concentrations in ug/L
chl_bb_star = chl_specific_bb['bb_ph_star'].tolist()
# ...the specific backscattering coefficient of phytoplankton b_bB_star(λ)
# b_bB_star(λ) was taken from (Theenathayalan & Shanmugam, 2021)


# Eq. 14 b_bB(λ) = b_bB_star(λ)*B^0.67547 (Theenathayalan & Shanmugam, 2021)
# This function does not work. The calculations were conducted in excel in the meantime.
"""def chl_backscatter(chl_concentration, chl_specific_backscatter):
    for i in range(len(chl_concentration)):
        chl_bb = [chl_specific_backscatter[x] * ((chl_concentration[i])**0.67547)
                  for x in range(len(chl_specific_backscatter))]
    return chl_bb"""


"""Total backscattering and absorption"""
# Create a new class


class Site:
    """A sampling site in the study."""
    def __init__(self, name):
        self.name = name
        self.measurements = {}  # You could add self.benthic = [] jne with absorption and backscatter
        self.depth = {}
# Add a new measurement dataset to a Site instance

    def add_measurement(self, measurement_id, data):
        if measurement_id in self.measurements.keys():
            self.measurements[measurement_id] = \
                pd.concat([self.measurements[measurement_id], data])

        else:
            self.measurements[measurement_id] = data
            self.measurements[measurement_id].name = measurement_id

    def add_depth(self, measurement_id, data):
        if measurement_id in self.depth.keys():
            self.depth[measurement_id] = \
                pd.concat([self.depth[measurement_id], data])

        else:
            self.depth[measurement_id] = data


# Create a dictionary with site IDs as keys
dict_sites = {k: Site(k) for k in sites}
print(sites)
print(dict_sites)

# Define a function that applies the add_measurement() function
def add_data(data_dictionary, site_str):
    data_dictionary[site_str].add_measurement('aW', pd.Series(aW_df['absorption']))
    data_dictionary[site_str].add_measurement('bbW', pd.Series(bbW_df['backscattering']))
    data_dictionary[site_str].add_measurement('aB', pd.Series(aB_df[site_str]))
    data_dictionary[site_str].add_measurement('aS', pd.Series(aS_df[site_str]))
    data_dictionary[site_str].add_measurement('aY', pd.Series(aY_df[site_str]))
    data_dictionary[site_str].add_measurement('bbB', pd.Series(bbB_df[site_str]))
    data_dictionary[site_str].add_measurement('bbS', pd.Series(bbS_df[site_str]))
    data_dictionary[site_str].add_depth('bottom_depth', pd.Series(depth_data[site_str][0]))


# Apply the function to all the sampling sites
for i in sites:
    add_data(dict_sites, i)