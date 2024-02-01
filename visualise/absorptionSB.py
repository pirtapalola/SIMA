"""This code reads the csv files containing the absorption coefficients
and organises the data into a single dataframe.
Examples of NAP and phytoplankton absorption coefficients are plotted."""

import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

os.chdir(r'C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/particulate_absorption_coefficients')
the_list = []

# Create a list that contains the paths of all the csv files in a folder
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/'
                                 r'particulate_absorption_coefficients'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)

# Define a function that takes a list of paths, reads all the csv files,
# and adds the selected column to a new pandas dataframe


def readdataframe_iop(list_paths, col_name):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[col_name])
    return df


plymouth_ph = readdataframe_iop(the_list, "a_phytoplankton")
plymouth_nap = readdataframe_iop(the_list, "a_detrital")
plymouth_spm = readdataframe_iop(the_list, "a_particulate")


# Add the wavelength range to the dataframes as a column
wavelength_list = [x for x in range(350, 851)]
plymouth_nap['wavelength_index'] = wavelength_list
plymouth_ph['wavelength_index'] = wavelength_list
plymouth_spm['wavelength_index'] = wavelength_list

# Use the column containing wavelengths to rearrange the column values in the right order.
# Create a new column with the wavelength range that will be used in the plot.
plymouth_nap.set_index('wavelength_index')
plymouth_nap = plymouth_nap.sort_index(ascending=False)
plymouth_nap['wavelength'] = wavelength_list

plymouth_ph.set_index('wavelength_index')
plymouth_ph = plymouth_ph.sort_index(ascending=False)
plymouth_ph['wavelength'] = wavelength_list

plymouth_spm.set_index('wavelength_index')
plymouth_spm = plymouth_spm.sort_index(ascending=False)
plymouth_spm['wavelength'] = wavelength_list

print(plymouth_nap)
# plot examples: NAP and phytoplankton absorption coefficients

fig2, ax2 = plt.subplots()
ax2.plot(plymouth_nap['wavelength'], plymouth_nap['LAG01A_22.csv'], label='Non-algal particulate matter')
ax2.plot(plymouth_ph['wavelength'], plymouth_ph['LAG01A_22.csv'], label='Phytoplankton')
ax2.plot(plymouth_ph['wavelength'], plymouth_spm['LAG01A_22.csv'], label='Suspended particulate matter')
ax2.set_xlabel('Wavelength (nm)')  # Add an x-label to the axes.
ax2.set_ylabel('$a$')  # Add a y-label to the axes.
ax2.set_title("Absorption coefficents")  # Add a title to the axes.
ax2.set_xlim([400, 700])  # Set the scale of the x-axis
ax2.set_xticks(np.arange(400, 720, 20, dtype=int))  # Modify the tick marks of the x-axis
ax2.legend()  # Add a legend.
plt.show()

plymouth_nap.to_csv('C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/absorption_nap.csv')
plymouth_ph.to_csv('C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/absorption_ph.csv')
plymouth_spm.to_csv('C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/absorption_spm.csv')
