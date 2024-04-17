"""

This code applies cubic spline interpolation to SNR and Lref values of the Chime satellite.
The reflectance data is interpolated to 1nm intervals.

STEP 1. Access the reflectance data.
STEP 2. Apply the cubic spline method to the data.
STEP 3. Create a plot to visualise the interpolation.
STEP 4. Save the interpolated data in a csv file.

This code was last modified by Pirta Palola on 17 April 2024.

"""

# Import libraries
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pylab as plt

"""STEP 1. Access the reflectance data."""

# Create a pandas dataframe with the default average reflectance of coral from Hydrolight
snr_points = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                         "Jan2024_lognormal_priors/SNR_CHIME/chime_lref.csv")

# Create a numpy array containing the wavelengths
snr_wavelength = snr_points['wavelength'].to_numpy()

"""STEP 2. Apply the cubic spline method to the data."""

# Apply the cubic spline method to the data.
x = snr_wavelength
y0 = snr_points['L_ref']
cs0 = CubicSpline(x, y0)
xs = np.arange(400, 705, 5)  # Define the wavelength range and spectral resolution of the end product
index_list = []  # Create an index list that corresponds to the number of rows in the end product
for z in range(len(xs)):
    index_list.append(z)
refl_list = []
refl_lis2 = []
for i in xs:
    n = cs0(i)
    refl_list.append(n)
for element in index_list:
    refl_lis2.append(float(refl_list[element]))
print(len(refl_lis2))

"""STEP 3. Create a plot to visualise the interpolation."""

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y0, label='Data', color='black', linestyle=':', linewidth=5)
ax.plot(xs, cs0(xs), label='Cubic spline interpolation', color='orange', linewidth=2)
ax.legend(loc='lower left', ncol=2)
ax.set_xlim(400, 700)
# ax.set_ylim(0, 0.08)
plt.show()

"""STEP 4. Save the interpolated data in a csv file."""

# Add the data in a pandas data frame
snr_dataframe = pd.DataFrame()

# Create a list containing the wavelengths
wavelength_list = []
a = range(399, 701)
for q in a:
    wavelength_list.append(q)


# Create a function that filters the desired wavelengths from the data
def filter_function(df, snr_df):
    df['wavelength'] = xs
    df['SNR'] = snr_df
    df = df[df.wavelength > 399]
    df = df[df.wavelength < 701]
    return df


# Apply the function and save the output as a csv file
snr_interpolated = filter_function(snr_dataframe, refl_lis2)
print(snr_interpolated)
snr_interpolated.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                        "Jan2024_lognormal_priors/SNR_CHIME/Lref_chime.csv")
