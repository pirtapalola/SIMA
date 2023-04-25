"""This code calculates:
- The backscattering coefficient of phytoplankton following Eq. 14 (Theenathayalan & Shanmugam, 2021)
- Total backscattering and absorption"""

import pandas as pd

"""Backscattering coefficient of phytoplankton"""

# Read the csv file
chl = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                  'Plymouth_iop/plymouth_iop_model/chlorophyll.csv')
chl_specific_bb = pd.read_csv('C:/Users/pirtapalola/Documents/iop_data/data/'
                              'Plymouth_iop/plymouth_iop_model/bb_ph_star.csv')

# Create a list of...
sites = chl['Unique_ID']  # ...site IDs
chl_list = chl['chl']  # ...chl-a concentrations in ug/L
chl_bb_star = chl_specific_bb['bb_ph_star']  # ...the specific backscattering coefficient of phytoplankton b_bB_star(λ)
# b_bB_star(λ) was taken from (Theenathayalan & Shanmugam, 2021)


# Eq. 14 b_bB(λ) = b_bB_star(λ)*B^0.67547 (Theenathayalan & Shanmugam, 2021)
# This function does not work. The calculations were conducted in excel in the meantime.
def chl_backscatter(chl_concentration, chl_specific_backscatter):
    for i in range(len(chl_concentration)):
        chl_bb = [chl_specific_backscatter[x] * ((chl_concentration[i])**0.67547)
                  for x in range(len(chl_specific_backscatter))]
    return chl_bb


"""Total backscattering and absorption"""

