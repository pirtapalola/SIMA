"""This python script depicts a forward model based on the work of
Lee et al. (1998, 1997),
Brando et al. (2009),
Giardino et al. (2012), and
Petit et al. (2017)"""

# Import libraries
import numpy as np
import math

# Access the absorption, backscattering, and benthic reflectance data


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

kappa = a + bb
u_var = bb/kappa

# Lee et al. (1999) Eqs. (5)
# Optical path-elongation factors for scattered photons

du_water = 1.03 * np.power((1.00 + (2.40 * u_var)), 0.50)  # from the water column
du_bottom = 1.04 * np.power((1.00 + (5.40 * u_var)), 0.50)  # from the bottom

# Lee et al. (1999) Eq. (4)
# Remotely sensed sub-surface reflectance for optically deep water

rrsdp = (0.084 + 0.17 * u_var) * u_var

# Define variables

inv_cos_theta_w = 1.0 / math.cos(theta_w)
depth = 0.9
kappa_d = kappa * depth

# Lee et al. (1999) Eq. (3)
# Remote sensing reflectance
# Assumption: nadir-viewing sensor

rrs = (rrsdp * (1.0 - np.exp(-(inv_cos_theta_w + du_water) * kappa_d)) +
       ((1.0 / math.pi) * r_substratum *
        np.exp(-(inv_cos_theta_w + du_bottom) * kappa_d)))
