"""This python script depicts a forward model based on the work of
Lee et al. (1998, 1999),
Brando et al. (2009),
Giardino et al. (2012), and
Petit et al. (2017)"""

# Import libraries
import numpy as np
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


# Create a new class
class Site:
    """A sampling site in the study."""
    def __init__(self, name):
        self.name = name
        self.measurements = {}
# Add a new measurement dataset to a Site instance

    def add_measurement(self, measurement_id, data):
        if measurement_id in self.measurements.keys():
            self.measurements[measurement_id] = \
                pd.concat([self.measurements[measurement_id], data])

        else:
            self.measurements[measurement_id] = data
            self.measurements[measurement_id].name = measurement_id


ONE02 = Site('ONE02')


ONE02.add_measurement('benthic_reflectance', pd.Series(benthic_reflectance['ONE02']))
print(ONE02.measurements.keys())
print(list(ONE02.measurements['benthic_reflectance']))
ONE02_benthic = list(ONE02.measurements['benthic_reflectance'])
ONE02.add_measurement('absorption', pd.Series(total_absorption['ONE02']))
ONE02.add_measurement('backscatter', pd.Series(total_backscattering['ONE02']))
ONE02_a = list(ONE02.measurements['absorption'])
ONE02_bb = list(ONE02.measurements['backscatter'])

# Sub-surface solar zenith angle in radians
# Refractive index of seawater (temperature = 20C, salinity = 25g/kg, light wavelength = 589.3 nm)

water_refractive_index = 1.33784
inv_refractive_index = 1.0 / water_refractive_index
theta_air = 38.28285

# If the sensors are not nadir-viewing, define the following:
# Then, use Eq. (9) from Lee et al. (1999) to calculate remote sensing reflectance instead of Eq. (3)
# theta_o = math.asin(inv_refractive_index * math.sin(math.radians(off_nadir)))
# du_column_scaled = du_column * inv_cos_theta_0
# du_bottom_scaled = du_bottom * inv_cos_theta_0
# inv_cos_theta_0 = 1.0 / math.cos(theta_o)

# Sub-surface viewing angle in radians

theta_w = math.asin(inv_refractive_index * math.sin(math.radians(theta_air)))

# Lee et al. (1999) Eqs.(6)


kappa = [ONE09_a[x] + ONE09_bb[x] for x in range(len(ONE09_a))]
u_var = [ONE09_bb[x]/kappa[x] for x in range(len(ONE09_bb))]

# Lee et al. (1999) Eqs. (5)
# Optical path-elongation factors for scattered photons

du_water = [1.03 * np.power((1.00 + (2.40 * n)), 0.50) for n in u_var]  # from the water column
du_bottom = [1.04 * np.power((1.00 + (5.40 * n)), 0.50) for n in u_var]  # from the bottom

# Lee et al. (1999) Eq. (4)
# Remotely sensed sub-surface reflectance for optically deep water

rrsdp = [(0.084 + 0.17 * n) * n for n in u_var]

# Define variables

inv_cos_theta_w = 1.0 / math.cos(theta_w)
depth = 0.9
kappa_d = [n * depth for n in kappa]

# Lee et al. (1999) Eq. (3)
# Remote sensing reflectance
# Assumption: nadir-viewing sensor

rrs_ONE09 = [(rrsdp[x] * (1.0 - np.exp(-(inv_cos_theta_w + du_water[x]) * kappa_d[x])) +
              ((1.0 / math.pi) * ONE09_rrs_benthic[x] *
               np.exp(-(inv_cos_theta_w + du_bottom[x]) * kappa_d[x]))) for x in range(len(rrsdp))]

# Save the results in a pandas dataframe

rrs_data = pd.DataFrame()
rrs_data['ONE09'] = rrs_ONE09

# Save the dataframe to a csv
rrs_data.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/forward_model_results.csv')
