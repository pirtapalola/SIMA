"""

This code applies cubic spline interpolation to the hyperspectral reflectance data
calculated from TriOS RAMSES radiometric measurements.

STEP 1. Access the reflectance data.
STEP 2. Apply the cubic spline method to the data.
STEP 3. Create a plot to visualise the interpolation.
STEP 4. Save the interpolated data in a csv file.

This code was last modified by Pirta Palola on 19 January 2024

"""


# Import libraries
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pylab as plt

"""STEP 1. Access the reflectance data."""

# Specify file location
path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/" \
       "Methods_Ecolight/In_water_calibration_2022/ONE02_2022_calibrated.csv"

# Create a pandas dataframe
reflectance_data = pd.read_csv(path)
trios_wavelength = reflectance_data['wavelength'].to_numpy()  # Wavelengths measured by the TriOS RAMSES radiometers

"""STEP 2. Apply the cubic spline method to the data."""

# Apply the cubic spline method to the data.
x = trios_wavelength
y0 = reflectance_data['reflectance']
cs0 = CubicSpline(x, y0)
xs = np.arange(319, 951, 2)  # Define the wavelength range and spectral resolution of the end product
index_list = []  # Create an index list that corresponds to the number of rows in the end product
for z in range(len(xs)):
    index_list.append(z)
empty_list = []
data_list = []
for i in xs:
    n = cs0(i)
    empty_list.append(n)
for element in index_list:
    data_list.append(float(empty_list[element]))

"""STEP 3. Create a plot to visualise the interpolation."""

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y0, label='Data', color='black', linestyle=':', linewidth=5)
ax.plot(xs, cs0(xs), label='Cubic spline interpolation', color='orange', linewidth=2)
ax.legend(loc='lower left', ncol=2)
ax.set_xlim(319, 800)
ax.set_ylim(0, 0.08)
plt.show()

"""STEP 4. Save the interpolated data in a csv file."""

# Add the data in a pandas data frame
interpolated_reflectance = pd.DataFrame()

# Create a list containing the wavelengths
wavelength_319_951 = []
a = np.arange(319, 951, 2)
for q in a:
    wavelength_319_951.append(q)


# Create a function that filters the desired wavelengths from the data
def benthic_reflectance_function(reflectance_df, benthic_df):
    reflectance_df['wavelength'] = wavelength_319_951
    reflectance_df['benthic'] = benthic_df
    reflectance_df = reflectance_df[reflectance_df.wavelength > 319]
    reflectance_df = reflectance_df[reflectance_df.wavelength < 951]
    return reflectance_df


# Apply the function and save the output as a csv file
interpolated_reflectance_df = benthic_reflectance_function(interpolated_reflectance, data_list)
interpolated_reflectance_df.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                                   "Methods_Ecolight/In_water_calibration_2022/ONE02_2022_interpolated.csv")
