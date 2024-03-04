"""

Apply the in-water calibration factor to the hyperspectral data measured with the TriOS RAMSES radiometers.

STEP 1. Read the data.
STEP 2. Calibrate the reflectance data.
STEP 3. Save the calibrated values into a csv file.

Last updated on 4 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd

"""STEP 1. Read the data."""

# Access the calibration factors from a csv file
calibration_factors = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                                  "Methods_Ecolight/In_water_calibration_2022/correction_factors_2022.csv")

air_878A_2022 = calibration_factors["Air_878A_2022"]  # Air calibration factor for the irradiance (E) sensor
aq_878A_2022 = calibration_factors["Aq_878A_2022"]  # In-water calibration factor for the irradiance (E) sensor
air_8789_2022 = calibration_factors["Air_8789_2022"]  # Air calibration factor for the radiance (L) sensor
aq_8789_2022 = calibration_factors["Aq_8789_2022"]  # In-water calibration factor for the radiance (L) sensor

# Access the TriOS RAMSES data
measurement = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                          "In_water_calibration_2023/uncorrected_COO28_44_surface_2023.csv")

# List of sample IDs
sample_IDs = list(measurement.columns)
print(sample_IDs)
#sample_IDs = sample_IDs[1::]  # Delete the first element of the list
#print(sample_IDs)

# List of irradiance sample IDs
e_sample_IDs = sample_IDs[0::2]
print(e_sample_IDs)

# List of radiance sample IDs
l_sample_IDs = sample_IDs[1::2]
print(l_sample_IDs)

"STEP 2. Calibrate the reflectance data."


def calibrate_reflectance(air878a, aq878a, air8789, aq8789, e_data, l_data):
    # Multiply the in-air calibrated data with the in-air calibration factor to get back to the raw values.
    e_raw = [e_data[item] * air878a[item] for item in range(len(e_data))]
    l_raw = [l_data[item] * air8789[item] for item in range(len(e_data))]
    # Divide the raw values with the in-water calibration factor.
    e_cal = [e_raw[item] / aq878a[item] for item in range(len(e_data))]
    l_cal = [l_raw[item] / aq8789[item] for item in range(len(e_data))]
    # Calculate reflectance (radiance divided by irradiance).
    reflectance = [l_cal[item] / e_cal[item] for item in range(len(e_cal))]
    return reflectance


# Create an empty list to store the data
refl_list = []

# Apply the function to all the samples using a loop
for x in range(len(e_sample_IDs)):
    refl = calibrate_reflectance(air_878A_2022, aq_878A_2022, air_8789_2022, aq_8789_2022,
                                 measurement[e_sample_IDs[x]], measurement[l_sample_IDs[x]])
    refl_list.append(refl)  # Append the results into a list

"STEP 3. Save the calibrated values into csv files."

# Create a DataFrame
calibrated_reflectance_df = pd.DataFrame(refl_list)
calibrated_reflectance_df = calibrated_reflectance_df.transpose()
print(calibrated_reflectance_df)
calibrated_reflectance_df.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
                                 "In_water_calibration_2023/calibrated_COO28_44_surface_2023.csv")
