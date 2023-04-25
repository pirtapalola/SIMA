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
# dict_sites['ONE02']

ONE02 = Site('ONE02')
ONE02_benthic = []
ONE02_a = []
ONE02_bb = []

ONE03 = Site('ONE03')
ONE03_benthic = []
ONE03_a = []
ONE03_bb = []

ONE07 = Site('ONE07')
ONE07_benthic = []
ONE07_a = []
ONE07_bb = []

ONE08 = Site('ONE08')
ONE08_benthic = []
ONE08_a = []
ONE08_bb = []

ONE09 = Site('ONE09')
ONE09_benthic = []
ONE09_a = []
ONE09_bb = []

ONE10 = Site('ONE10')
ONE10_benthic = []
ONE10_a = []
ONE10_bb = []

ONE11 = Site('ONE11')
ONE11_benthic = []
ONE11_a = []
ONE11_bb = []

ONE12 = Site('ONE12')
ONE12_benthic = []
ONE12_a = []
ONE12_bb = []

RIM01 = Site('RIM01')
RIM01_benthic = []
RIM01_a = []
RIM01_bb = []

RIM02 = Site('RIM02')
RIM02_benthic = []
RIM02_a = []
RIM02_bb = []

RIM03 = Site('RIM03')
RIM03_benthic = []
RIM03_a = []
RIM03_bb = []

RIM04 = Site('RIM04')
RIM04_benthic = []
RIM04_a = []
RIM04_bb = []

RIM05 = Site('RIM05')
RIM05_benthic = []
RIM05_a = []
RIM05_bb = []

RIM06 = Site('RIM06')
RIM06_benthic = []
RIM06_a = []
RIM06_bb = []


def adding_data(site_variable, site_str, empty_list_benthic, empty_list_absorption, empty_list_backscatter):
    site_variable.add_measurement('benthic_reflectance', pd.Series(benthic_reflectance[site_str]))
    site_variable.add_measurement('absorption', pd.Series(total_absorption[site_str]))
    site_variable.add_measurement('backscatter', pd.Series(total_backscattering[site_str]))
    empty_list_benthic.append(list(site_variable.absorption['benthic_reflectance']))
    empty_list_absorption.append(list(site_variable.absorption['absorption']))
    empty_list_backscatter.append(list(site_variable.absorption['backscatter']))


adding_data(ONE02, 'ONE02', ONE02_benthic, ONE02_a, ONE02_bb)
adding_data(ONE03, 'ONE03', ONE03_benthic, ONE03_a, ONE03_bb)
adding_data(ONE07, 'ONE07', ONE07_benthic, ONE07_a, ONE07_bb)
adding_data(ONE08, 'ONE08', ONE08_benthic, ONE08_a, ONE08_bb)
adding_data(ONE09, 'ONE09', ONE09_benthic, ONE09_a, ONE09_bb)
adding_data(ONE10, 'ONE10', ONE10_benthic, ONE10_a, ONE10_bb)
adding_data(ONE11, 'ONE11', ONE11_benthic, ONE11_a, ONE11_bb)
adding_data(ONE12, 'ONE12', ONE12_benthic, ONE12_a, ONE12_bb)
adding_data(RIM01, 'RIM01', RIM01_benthic, RIM01_a, RIM01_bb)
adding_data(RIM02, 'RIM02', RIM02_benthic, RIM02_a, RIM02_bb)
adding_data(RIM03, 'RIM03', RIM03_benthic, RIM03_a, RIM03_bb)
adding_data(RIM04, 'RIM04', RIM04_benthic, RIM04_a, RIM04_bb)
adding_data(RIM05, 'RIM05', RIM05_benthic, RIM05_a, RIM05_bb)
adding_data(RIM06, 'RIM06', RIM06_benthic, RIM06_a, RIM06_bb)

ONE02_benthic = ONE02_benthic[0]
ONE03_benthic = ONE03_benthic[0]
ONE07_benthic = ONE07_benthic[0]
ONE08_benthic = ONE08_benthic[0]
ONE09_benthic = ONE09_benthic[0]
ONE10_benthic = ONE10_benthic[0]
ONE11_benthic = ONE11_benthic[0]
ONE12_benthic = ONE12_benthic[0]
RIM01_benthic = RIM01_benthic[0]
RIM02_benthic = RIM02_benthic[0]
RIM03_benthic = RIM03_benthic[0]
RIM04_benthic = RIM04_benthic[0]
RIM05_benthic = RIM05_benthic[0]
RIM06_benthic = RIM06_benthic[0]

ONE02_a = ONE02_a[0]
ONE03_a = ONE03_a[0]
ONE07_a = ONE07_a[0]
ONE08_a = ONE08_a[0]
ONE09_a = ONE09_a[0]
ONE10_a = ONE10_a[0]
ONE11_a = ONE11_a[0]
ONE12_a = ONE12_a[0]
RIM01_a = RIM01_a[0]
RIM02_a = RIM02_a[0]
RIM03_a = RIM03_a[0]
RIM04_a = RIM04_a[0]
RIM05_a = RIM05_a[0]
RIM06_a = RIM06_a[0]

ONE02_bb = ONE02_bb[0]
ONE03_bb = ONE03_bb[0]
ONE07_bb = ONE07_bb[0]
ONE08_bb = ONE08_bb[0]
ONE09_bb = ONE09_bb[0]
ONE10_bb = ONE10_bb[0]
ONE11_bb = ONE11_bb[0]
ONE12_bb = ONE12_bb[0]
RIM01_bb = RIM01_bb[0]
RIM02_bb = RIM02_bb[0]
RIM03_bb = RIM03_bb[0]
RIM04_bb = RIM04_bb[0]
RIM05_bb = RIM05_bb[0]
RIM06_bb = RIM06_bb[0]


# Sub-surface solar zenith angle in radians
# Refractive index of seawater (temperature = 20C, salinity = 25g/kg, light wavelength = 589.3 nm)

water_refractive_index = 1.33784
inv_refractive_index = 1.0 / water_refractive_index

# If the sensors are not nadir-viewing, define the following:
# Then, use Eq. (9) from Lee et al. (1999) to calculate remote sensing reflectance instead of Eq. (3)
# theta_o = math.asin(inv_refractive_index * math.sin(math.radians(off_nadir)))
# du_column_scaled = du_column * inv_cos_theta_0
# du_bottom_scaled = du_bottom * inv_cos_theta_0
# inv_cos_theta_0 = 1.0 / math.cos(theta_o)

# Sub-surface viewing angle in radians

theta_w = [math.asin(inv_refractive_index * math.sin(math.radians(x))) for x in theta_air_list]
inv_cos_theta_w = [1.0 / math.cos(x) for x in theta_w]


def forward_model(site_name, absorption_list, backscattering_list, benthic, depth_var, inv_cos_theta_w_var):
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
    rrs_data[site_name] = rrs
    # Save the dataframe to a csv
    rrs_data.to_csv(('C:/Users/pirtapalola/Documents/iop_data/data/'+site_name+'.csv'))


forward_model('ONE02', ONE02_a, ONE02_bb, ONE02_benthic, 0.90, inv_cos_theta_w[0])
forward_model('ONE03', ONE03_a, ONE03_bb, ONE03_benthic, 0.68, inv_cos_theta_w[1])
forward_model('ONE07', ONE07_a, ONE07_bb, ONE07_benthic, 1.0, inv_cos_theta_w[2])
forward_model('ONE08', ONE08_a, ONE08_bb, ONE08_benthic, 1.2, inv_cos_theta_w[3])
forward_model('ONE09', ONE09_a, ONE09_bb, ONE09_benthic, 0.9, inv_cos_theta_w[4])
forward_model('ONE10', ONE10_a, ONE10_bb, ONE10_benthic, 1.0, inv_cos_theta_w[5])
forward_model('ONE11', ONE11_a, ONE11_bb, ONE11_benthic, 1.0, inv_cos_theta_w[6])
forward_model('ONE12', ONE12_a, ONE12_bb, ONE12_benthic, 1.0, inv_cos_theta_w[7])
forward_model('RIM01', RIM01_a, RIM01_bb, RIM01_benthic, 1.4, inv_cos_theta_w[8])
forward_model('RIM02', RIM02_a, RIM02_bb, RIM02_benthic, 1.3, inv_cos_theta_w[9])
forward_model('RIM03', RIM03_a, RIM03_bb, RIM03_benthic, 1.1, inv_cos_theta_w[10])
forward_model('RIM04', RIM04_a, RIM04_bb, RIM04_benthic, 1.2, inv_cos_theta_w[11])
forward_model('RIM05', RIM05_a, RIM05_bb, RIM05_benthic, 1.15, inv_cos_theta_w[12])
forward_model('RIM06', RIM06_a, RIM06_bb, RIM06_benthic, 1.3, inv_cos_theta_w[13])
