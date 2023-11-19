"""

This code applies cubic spline interpolation to the hyperspectral reflectance data from Hochberg et al.
The reflectance data is interpolated to 1nm intervals.

STEP 1. Access the reflectance data.
STEP 2. Apply the cubic spline method to the data.
STEP 3. Create a plot to visualise the interpolation.
STEP 4. Save the interpolated data in a csv file.

This code was last modified by Pirta Palola on 19 November 2023.

"""

# Import libraries
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pylab as plt

"""STEP 1. Access the reflectance data."""

# Create a pandas dataframe with the default average reflectance of coral from Hydrolight
hl_avg_coral_reflectance = pd.read_csv(
    'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral_HL.csv')

# Create a numpy array containing the wavelengths
hl_wavelength = hl_avg_coral_reflectance['wavelength'].to_numpy()

"""STEP 2. Apply the cubic spline method to the data."""

# Apply the cubic spline method to the data.
x = hl_wavelength
y0 = hl_avg_coral_reflectance['reflectance']
cs0 = CubicSpline(x, y0)
xs = np.arange(319, 951, 1)  # Define the wavelength range and spectral resolution of the end product
index_list = []  # Create an index list that corresponds to the number of rows in the end product
for z in range(0, 632):
    index_list.append(z)
refl_list = []
refl_lis2 = []
for i in xs:
    n = cs0(i)
    refl_list.append(n)
for element in index_list:
    refl_lis2.append(float(refl_list[element]))

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
P_lob22_reflectance = pd.DataFrame()

# Create a list containing the wavelengths
wavelength_319_951 = []
a = range(319, 951)
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
hl_reflectance = benthic_reflectance_function(P_lob22_reflectance, refl_lis2)
print(hl_reflectance)
hl_reflectance.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/hl_avg_coral_interpolated.csv')
