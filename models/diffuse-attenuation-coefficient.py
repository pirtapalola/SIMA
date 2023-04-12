"""This python script calculates the diffuse attenuation coefficients for each sampling site."""

# Import libraries
import math
import pandas as pd

# Access the absorption, backscattering, and benthic reflectance data
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
theta_air_data = read_csv_data('C:/Users/pirtapalola/Documents/iop_data/data/sun_zenith_angle.csv')
theta_air_list = list(theta_air_data['SZA'])


# Create a new class
class Site:
    """A sampling site in the study."""
    def __init__(self, name):
        self.name = name
        self.measurements = {}  # You could add self.benthic = [] jne with absorption and backscatter
# Add a new measurement dataset to a Site instance

    def add_measurement(self, measurement_id, data):
        if measurement_id in self.measurements.keys():
            self.measurements[measurement_id] = \
                pd.concat([self.measurements[measurement_id], data])

        else:
            self.measurements[measurement_id] = data
            self.measurements[measurement_id].name = measurement_id


dict_sites = {k: Site(k) for k in sites}


# Define a function that applies the add_measurement() function
def add_data(data_dictionary, site_str):
    data_dictionary[site_str].add_measurement('benthic_reflectance', pd.Series(benthic_reflectance[site_str]))
    data_dictionary[site_str].add_measurement('absorption', pd.Series(total_absorption[site_str]))
    data_dictionary[site_str].add_measurement('backscatter', pd.Series(total_backscattering[site_str]))


# Apply the function to all the sampling sites
for i in sites:
    add_data(dict_sites, i)

# print(dict_sites['ONE02'].measurements['backscatter'])

# Sub-surface solar zenith angle in radians
# Refractive index of seawater (temperature = 20C, salinity = 25g/kg, light wavelength = 589.3 nm)

water_refractive_index = 1.33784
inv_refractive_index = 1.0 / water_refractive_index

# Sub-surface viewing angle in radians

theta_w = [math.asin(inv_refractive_index * math.sin(math.radians(x))) for x in theta_air_list]
inv_cos_theta_w = [1.0 / math.cos(x) for x in theta_w]

# Calculate the diffuse attenuation coefficient for each site


def calculate_kd_coefficient(site_str, inv_cos_theta_w_var, absorption_list, backscattering_list):
    kappa = [absorption_list[x] + backscattering_list[x]
             for x in range(len(absorption_list))]
    kd_coefficient = [inv_cos_theta_w_var*kappa[x] for x in range(len(absorption_list))]
# Save the results in a pandas dataframe
    kd_data = pd.DataFrame()
    kd_data[site_str] = kd_coefficient
    # Save the dataframe to a csv
    kd_data.to_csv(('C:/Users/pirtapalola/Documents/iop_data/kd_data/'+site_str+'_kd'+'.csv'))


for i in sites:
    calculate_kd_coefficient(i, inv_cos_theta_w[0], dict_sites[i].measurements['absorption'],
                             dict_sites[i].measurements['backscatter'])
