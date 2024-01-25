"""

This code applies cubic spline interpolation to the hyperspectral reflectance data
calculated from TriOS RAMSES radiometric measurements.

STEP 1. Access the reflectance data.
STEP 2. Apply the cubic spline method to the data.
STEP 3. Save the interpolated data in csv files.

Last updated on 25 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np

"""STEP 1. Access the reflectance data."""

# Specify file location
path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/" \
       "Methods_Ecolight/In_water_calibration_2023/calibrated_surface_reflectance_2023.csv"

# Create a pandas dataframe
reflectance_data = pd.read_csv(path)

# Wavelengths measured by the TriOS RAMSES radiometers
trios_wavelength = reflectance_data['wavelength'].to_numpy()

# Sample IDs
sample_IDs = list(reflectance_data.columns)
sample_IDs = sample_IDs[1::]  # Delete the first element of the list
print(sample_IDs)

"""STEP 2. Apply the cubic spline method to the data."""


def cubic_spline_interpolation(y0_list, x_list):
    # Define the wavelength range and spectral resolution of the end product
    xs = np.arange(319, 951, 2)
    # Create an index list that corresponds to the number of rows in the end product
    index_list = []
    for z in range(len(xs)):
        index_list.append(z)
    # Define the cubic spline method
    cs0 = CubicSpline(x_list, y0_list)
    # Create empty lists to store the results
    empty_list = []
    data_list = []
    # Save the results in a list
    for i in xs:
        n = cs0(i)
        empty_list.append(n)
    for element in index_list:
        data_list.append(float(empty_list[element]))
    return data_list


# Apply the cubic spline function to the data and save the results in a list.
interpolation_results_list = []
for x in sample_IDs:
    interpolation_result = cubic_spline_interpolation(reflectance_data[x], trios_wavelength)
    interpolation_results_list.append(interpolation_result)

print("Number of sample IDs: ", len(sample_IDs))
print("Number of measurements: ", len(interpolation_results_list))

# Create a plot to visualise the interpolation.

# fig, ax = plt.subplots(figsize=(6.5, 4))
# ax.plot(x, y0, label='Data', color='black', linestyle=':', linewidth=5)
# ax.plot(xs, cs0(xs), label='Cubic spline interpolation', color='orange', linewidth=2)
# ax.legend(loc='lower left', ncol=2)
# ax.set_xlim(319, 800)
# ax.set_ylim(0, 0.08)
# plt.show()

"""STEP 3. Save the interpolated data in a csv file."""

# Create a DataFrame
interpolated_reflectance_df = pd.DataFrame(interpolation_results_list)
interpolated_reflectance_df = interpolated_reflectance_df.transpose()
print(interpolated_reflectance_df)
interpolated_reflectance_df.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                                   "In_water_calibration_2023/interpolated_surface_reflectance_2023.csv")

