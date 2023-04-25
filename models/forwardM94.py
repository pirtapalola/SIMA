import os
import pandas as pd
import numpy as np


os.chdir(r'C:/Users/pirtapalola/Documents/iop_data/kd_data')
sites = ['ONE02', 'ONE03', 'ONE07', 'ONE08', 'ONE09', 'ONE10', 'ONE11', 'ONE12',
         'RIM01', 'RIM02', 'RIM03', 'RIM04', 'RIM05', 'RIM06']


def read_csv_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


benthic_reflectance = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/benthic_reflectance.csv')
total_absorption = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/total_a.csv')
total_backscattering = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/total_bb.csv')
wavelength = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/wavelength.csv')
wavelength_range = list(wavelength['wavelength'])
kd_coefficients = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/kd_coefficients.csv')
depth_data = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/depth_data.csv')


# Define the forward model as per Maritorena et al. (1994) Equation 9a
def maritorena(site_name, absorption_list, backscattering_list, benthic, diffuse_attenuation, depth_var):
    kappa = [absorption_list[x] + backscattering_list[x] for x in range(len(absorption_list))]
    u_var = [backscattering_list[x] / kappa[x] for x in range(len(backscattering_list))]
    r_water = [(0.084 + 0.17 * n) * n for n in u_var]
    r_total = [r_water[x] + (benthic[x] - r_water[x])
               * np.exp(-2 * diffuse_attenuation[x] * depth_var) for x in range(len(benthic))]
    maritorena_data = pd.DataFrame()
    maritorena_data[site_name] = r_total
    maritorena_data.to_csv(('C:/Users/pirtapalola/Documents/iop_data/data/M94_data/'+site_name+'M94'+'.csv'))


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


dict_sites = {k: Site(k) for k in sites}


# Define a function that applies the add_measurement() function
def add_data(data_dictionary, site_str):
    data_dictionary[site_str].add_measurement('benthic_reflectance', pd.Series(benthic_reflectance[site_str]))
    data_dictionary[site_str].add_measurement('absorption', pd.Series(total_absorption[site_str]))
    data_dictionary[site_str].add_measurement('backscatter', pd.Series(total_backscattering[site_str]))
    data_dictionary[site_str].add_measurement('kd_coefficients', pd.Series(kd_coefficients[site_str]))
    data_dictionary[site_str].add_depth('bottom_depth', depth_data[site_str][0])


# Apply the function to all the sampling sites
for i in sites:
    add_data(dict_sites, i)

for i in sites:
    maritorena(i, dict_sites[i].measurements['absorption'], dict_sites[i].measurements['backscatter'],
               dict_sites[i].measurements['benthic_reflectance'],
               dict_sites[i].measurements['kd_coefficients'], dict_sites[i].depth['bottom_depth'])
