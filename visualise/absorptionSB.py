"""This code reads the csv files containing the absorption coefficients
and organises the data into a single dataframe.
Examples of NAP and phytoplankton absorption coefficients are plotted."""

import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

os.chdir(r'C:/Users/pirtapalola/Documents/iop_data/plymouth_data')
the_list = []

# Create a list that contains the paths of all the csv files in a folder
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/plymouth_data'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)

# Define a function that takes a list of paths, reads all the csv files,
# and adds the first column to a new pandas dataframe
# This function reads the NAP data column


def readdataframe_nap(list_paths):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[10])
    return df


# This function reads the phytoplankton data column


def readdataframe_ph(list_paths):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[11])
    return df


plymouth_nap = readdataframe_nap(the_list)
plymouth_ph = readdataframe_ph(the_list)


# Add the wavelength range to the dataframes as a column
wavelength_list = [x for x in range(350, 851)]
plymouth_nap['wavelength_index'] = wavelength_list
plymouth_ph['wavelength_index'] = wavelength_list

# Use the column containing wavelengths to rearrange the column values in the right order.
# Create a new column with the wavelength range that will be used in the plot.
plymouth_nap.set_index('wavelength_index')
plymouth_nap = plymouth_nap.sort_index(ascending=False)
plymouth_nap['wavelength'] = wavelength_list

plymouth_ph.set_index('wavelength_index')
plymouth_ph = plymouth_ph.sort_index(ascending=False)
plymouth_ph['wavelength'] = wavelength_list

# plot examples: NAP and phytoplankton absorption coefficients

fig2, ax2 = plt.subplots()
ax2.plot(plymouth_nap['wavelength'], plymouth_nap['LAG104A_0m_4L_10112022.csv'], label='Non-algal particulate matter')
ax2.plot(plymouth_ph['wavelength'], plymouth_ph['LAG104A_0m_4L_10112022.csv'], label='Phytoplankton')
ax2.set_xlabel('Wavelength (nm)')  # Add an x-label to the axes.
ax2.set_ylabel('$a$')  # Add a y-label to the axes.
ax2.set_title("Absorption coefficents")  # Add a title to the axes.
ax2.set_xlim([400, 700])  # Set the scale of the x-axis
ax2.set_xticks(np.arange(400, 720, 20, dtype=int))  # Modify the tick marks of the x-axis
ax2.legend()  # Add a legend.
plt.show()

plymouth_nap.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/plymouth_nap.csv')
plymouth_ph.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/plymouth_ph.csv')
