"""
PRE-PROCESSING HYPERSPECTRAL DATA I: Interpolation
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

This code applies cubic spline interpolation to the hyperspectral reflectance data
calculated from TriOS RAMSES radiometric measurements.

STEP 1. Access the reflectance data.
STEP 2. Apply the cubic spline method to the data.
STEP 3. Save the interpolated data in csv files.

Last updated on 26 August 2024

"""

# Import libraries
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np

"""STEP 1. Access the reflectance data."""

# Read the csv files containing the data
reflectance_data = pd.read_csv("data/field_data/unprocessed_reflectance_tetiaroa_2022.csv")

# Wavelengths measured by the TriOS RAMSES radiometers
trios_wavelength = reflectance_data["wavelength"].to_numpy()

# Sample IDs
sample_IDs = list(reflectance_data.columns)
sample_IDs = sample_IDs[1::]  # Delete the first element of the list
print(sample_IDs)

"""STEP 2. Apply the cubic spline method to the data."""


def cubic_spline_interpolation(y0_list, x_list):
    # Define the wavelength range and spectral resolution of the end product
    xs = np.arange(400, 705, 5)
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
wavelength_range = np.arange(400, 705, 5)
interpolated_reflectance_df = pd.DataFrame(interpolation_results_list)
interpolated_reflectance = interpolated_reflectance_df.transpose()
interpolated_reflectance.columns = sample_IDs
interpolated_reflectance.insert(loc=0, column="wavelength", value=wavelength_range)
print(interpolated_reflectance)
interpolated_reflectance.to_csv("data/field_data/interpolated_reflectance_tetiaroa_2022.csv", index=False)
