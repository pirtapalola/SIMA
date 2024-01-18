"""

Apply the in-water calibration factor to the hyperspectral data measured with the TriOS RAMSES radiometers.

STEP 1. Read the data.
STEP 2. Multiply the in-air calibrated data with the in-air calibration factor to get back to the raw values.
STEP 3. Divide the raw values with the in-water calibration factor.

Last updated on 18 January 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd

# Access the calibration factors from a csv file
calibration_factors = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                                  "Methods_Ecolight/In_water_calibration_2022/correction_factors_2022.csv")

air_878A_2022 = calibration_factors["Air_878A_2022"]
aq_878A_2022 = calibration_factors["Aq_878A_2022"]
air_8789_2022 = calibration_factors["Air_8789_2022"]
aq_8789_2022 = calibration_factors["Aq_8789_2022"]

# Access the TriOS RAMSES data
measurement = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                          "Methods_Ecolight/In_water_calibration_2022/RIM01_2022.csv")
rim01_e1_uncal = measurement["Mean_e1"]
rim01_l1_uncal = measurement["Mean_l1"]

"STEP 2. Multiply the in-air calibrated data with the in-air calibration factor to get back to the raw values."

rim01_e1_raw = [rim01_e1_uncal[i] * air_878A_2022[i] for i in range(len(rim01_e1_uncal))]
rim01_l1_raw = [rim01_l1_uncal[i] * air_8789_2022[i] for i in range(len(rim01_l1_uncal))]

"STEP 3. Divide the raw values with the in-water calibration factor."

rim01_e1_cal = [rim01_e1_raw[i] / aq_878A_2022[i] for i in range(len(rim01_e1_raw))]
rim01_l1_cal = [rim01_l1_raw[i] / aq_8789_2022[i] for i in range(len(rim01_l1_raw))]

"STEP 4. Save the calibrated values into a csv file."

cal_values = pd.DataFrame()
cal_values["rim01_e1_cal"] = rim01_e1_cal
cal_values["rim01_l1_cal"] = rim01_l1_cal
cal_values["rim01_rrs"] = [rim01_l1_cal[i] / rim01_e1_cal[i] for i in range(len(rim01_e1_cal))]

cal_values.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                  "Methods_Ecolight/In_water_calibration_2022/RIM01_2022_calibrated.csv")
