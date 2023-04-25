"""This code implements the forward models of Lee tal. (1999) and Maritorena et al. (1994)"""

# Import libraries

import pandas as pd
import numpy as np
import math


sites = ['ONE02', 'ONE03', 'ONE07', 'ONE08', 'ONE09', 'ONE10', 'ONE11', 'ONE12',
         'RIM01', 'RIM02', 'RIM03', 'RIM04', 'RIM05', 'RIM06',
         'LAG01B', 'ONE05A', 'ONE05B', 'ONE06A', 'ONE06B', 'RIM08A', 'RIM08B']


def read_csv_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


benthic_reflectance = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/benthic_reflectance.csv')
total_absorption = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/total_a.csv')
total_backscattering = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/total_bb.csv')
wavelength = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/wavelength.csv')
wavelength_range = list(wavelength['wavelength'])
kd_coefficients = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/kd_coefficients.csv')
depth_data = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/depth_data.csv')
theta_air_data = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/input/sun_zenith_angle.csv')
theta_air_list = list(theta_air_data['SZA'])

"""Lee et al. (1999)"""
# Sub-surface solar zenith angle in radians
# Refractive index of seawater (temperature = 20C, salinity = 25g/kg, light wavelength = 589.3 nm)
water_refractive_index = 1.33784
inv_refractive_index = 1.0 / water_refractive_index

# Sub-surface viewing angle in radians
theta_w = [math.asin(inv_refractive_index * math.sin(math.radians(x))) for x in theta_air_list]
inv_cos_theta_w = [1.0 / math.cos(x) for x in theta_w]
inv_cos_theta_w_data = pd.DataFrame(columns=sites)
inv_cos_theta_w_data.loc[0] = inv_cos_theta_w


def lee(site_name, absorption_list, backscattering_list, benthic, depth_var, inv_cos_theta_w_var):
    kappa = [absorption_list[x] + backscattering_list[x]
             for x in range(len(absorption_list))]  # Lee et al. (1999) Eqs.(6)
    u_var = [backscattering_list[x] / kappa[x] for x in range(len(backscattering_list))]
    # Lee et al. (1999) Eqs. (5)
    # Optical path-elongation factors for scattered photons
    du_water = [1.03 * np.power((1.00 + (2.40 * n)), 0.50) for n in u_var]  # from the water column
    du_bottom = [1.04 * np.power((1.00 + (5.40 * n)), 0.50) for n in u_var]  # from the bottom
    # Lee et al. (1999) Eq. (4)
    # Remotely sensed sub-surface reflectance for optically deep water
    rrsdp = [(0.084 + 0.17 * n) * n for n in u_var]
    kappa_d = [n * depth_var for n in kappa]
    # Lee et al. (1999) Eq. (3)
    # Remote sensing reflectance
    # Assumption: nadir-viewing sensor
    rrs = [(rrsdp[x] * (1.0 - np.exp(-(inv_cos_theta_w_var + du_water[x]) * kappa_d[x]))
            + ((1.0 / math.pi) * benthic[x]
               * np.exp(-(inv_cos_theta_w_var + du_bottom[x]) * kappa_d[x]))) for x in range(len(rrsdp))]
    # Save the results in a pandas dataframe
    rrs_data = pd.DataFrame()
    rrs_data[(site_name+'_L99')] = rrs
    # Save the dataframe to a csv
    rrs_data.to_csv(('C:/Users/pirtapalola/Documents/iop_data/data/L99_data/'+site_name+'L99.csv'))


# Define the forward model as per Maritorena et al. (1994) Equation 9a
def maritorena(site_name, absorption_list, backscattering_list, benthic, diffuse_attenuation, depth_var):
    kappa = [absorption_list[x] + backscattering_list[x] for x in range(len(absorption_list))]
    u_var = [backscattering_list[x] / kappa[x] for x in range(len(backscattering_list))]
    r_water = [(0.084 + 0.17 * n) * n for n in u_var]
    r_total = [r_water[x] + (benthic[x] - r_water[x])
               * np.exp(-2 * diffuse_attenuation[x] * depth_var) for x in range(len(benthic))]
    maritorena_data = pd.DataFrame()
    maritorena_data[(site_name+'_M94')] = r_total
    maritorena_data.to_csv(('C:/Users/pirtapalola/Documents/iop_data/data/M94_data/'+site_name+'M94.csv'))


# Create a new class
class Site:
    """A sampling site in the study."""
    def __init__(self, name):
        self.name = name
        self.measurements = {}  # You could add self.benthic = [] jne with absorption and backscatter
        self.depth = {}
        self.theta_w = {}
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

    def add_theta_w(self, measurement_id, data):
        if measurement_id in self.depth.keys():
            self.theta_w[measurement_id] = \
                pd.concat([self.depth[measurement_id], data])

        else:
            self.theta_w[measurement_id] = data


dict_sites = {k: Site(k) for k in sites}


# Define a function that applies the add_measurement() function
def add_data(data_dictionary, site_str):
    data_dictionary[site_str].add_measurement('benthic_reflectance', pd.Series(benthic_reflectance[site_str]))
    data_dictionary[site_str].add_measurement('absorption', pd.Series(total_absorption[site_str]))
    data_dictionary[site_str].add_measurement('backscatter', pd.Series(total_backscattering[site_str]))
    data_dictionary[site_str].add_measurement('kd_coefficients', pd.Series(kd_coefficients[site_str]))
    data_dictionary[site_str].add_depth('bottom_depth', depth_data[site_str][0])
    data_dictionary[site_str].add_theta_w('theta_w', inv_cos_theta_w_data[site_str][0])


# Apply the function to all the sampling sites
for i in sites:
    add_data(dict_sites, i)

for i in sites:
    maritorena(i, dict_sites[i].measurements['absorption'], dict_sites[i].measurements['backscatter'],
               dict_sites[i].measurements['benthic_reflectance'],
               dict_sites[i].measurements['kd_coefficients'], dict_sites[i].depth['bottom_depth'])

for i in sites:
    lee(i, dict_sites[i].measurements['absorption'], dict_sites[i].measurements['backscatter'],
        dict_sites[i].measurements['benthic_reflectance'], dict_sites[i].depth['bottom_depth'],
        dict_sites[i].theta_w['theta_w'])
