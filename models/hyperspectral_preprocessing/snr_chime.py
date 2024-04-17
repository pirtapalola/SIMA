"""

Calculate SNR per wavelength.
SNR(L)=SNR(L_ref )× √(L/L_ref)

Last updated on 17 April 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import numpy as np

# Read the data
path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Jan2024_lognormal_priors/"
L_w = pd.read_csv(path + "simulated_Lw_no_noise.csv")  # Read the simulated radiance (L_w) data
L_ref = pd.read_csv(path + "SNR_CHIME/Lref_chime.csv")  # Read the L_ref data
SNR_chime = pd.read_csv(path + "SNR_CHIME/SNR_chime.csv")  # Read the Chime SNR data

L_ref = [int(i) for i in L_ref["L_ref"]]
SNR_chime = [int(i) for i in SNR_chime["SNR"]]

print(L_ref)
print(SNR_chime)
print(L_w)


# Convert from W m-2 sr-1 nm-1 to W m-2 sr-1 um-1
def multiply1000(x):
    return x * 1000


wavelengths = L_w.columns.tolist()
for col in wavelengths:
    L_w[col] = L_w[col].apply(multiply1000)  # Multiply each value by 1000
print(L_w)

# Number of rows in the dataset
num_rows = len(L_w["400"])

# Create an empty dataframe to save the results
SNR = pd.DataFrame()

# Apply scaling to calculate SNR
for row in range(num_rows):
    spectrum = L_w.iloc[row]  # Modify one row at a time
    new_spectrum = L_ref*(np.sqrt(spectrum/L_ref))  # SNR(L)=SNR(L_ref )× √(L/L_ref)
    SNR[str(row)] = new_spectrum  # Save the result as a column in the dataframe

# Save the results into a csv file
SNR.to_csv(path + "SNR_CHIME/SNR_w.csv", index=False)
