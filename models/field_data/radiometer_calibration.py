"""

Apply the in-water calibration factor to the hyperspectral data measured with the TriOS RAMSES radiometers.

STEP 1. Read the data.
STEP 2. Calibrate the reflectance data.
STEP 3. Save the calibrated values into a csv file.

Last updated on 20 January 2024 by Pirta Palola

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
                          "In_water_calibration_2022/uncorrected_surface_measurements_tetiaroa_2022.csv")

# List of sample IDs
sample_IDs = list(measurement.columns)
print(sample_IDs)

"STEP 2. Calibrate the reflectance data."


def calibrate_reflectance(air878A, aq878A, air8789, aq8789, e_data, l_data):
    # Multiply the in-air calibrated data with the in-air calibration factor to get back to the raw values.
    e_raw = [e_data[i] * air878A[i] for i in range(len(e_data))]
    l_raw = [l_data[i] * air8789[i] for i in range(len(e_data))]
    # Divide the raw values with the in-water calibration factor.
    e_cal = [e_raw[i] / aq878A[i] for i in range(len(e_data))]
    l_cal = [l_raw[i] / aq8789[i] for i in range(len(e_data))]
    # Calculate reflectance (radiance divided by irradiance).
    reflectance = l_cal/e_cal
    return reflectance


"STEP 3. Save the calibrated values into a csv file."

cal_values = pd.DataFrame()
cal_values["one02_e1_cal"] = rim01_e1_cal
cal_values["one02_l1_cal"] = rim01_l1_cal
cal_values["one02_rrs"] = [rim01_l1_cal[i] / rim01_e1_cal[i] for i in range(len(rim01_e1_cal))]

cal_values.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                  "Methods_Ecolight/In_water_calibration_2022/ONE02_2022_calibrated.csv")