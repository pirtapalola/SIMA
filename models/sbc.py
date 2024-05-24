"""

Simulation-based calibration
STEP 1. Read the simulated dataset.
STEP 2. Define theta and x.
STEP 3. Load the posterior.
STEP 4. Run SBC.

Last updated on 24 May 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from sbi.analysis import check_sbc, run_sbc
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sbi.analysis.plot import sbc_rank_plot

"""STEP 1. Read the simulated dataset."""

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Evaluation_data/'
                                    'simulated_reflectance_1000SNR_evaluate.csv')

#wavelengths = [443, 490, 531, 565, 610, 665, 700]
#wavelengths = [str(item) for item in wavelengths]
#simulated_reflectance = simulated_reflectance[wavelengths]

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Evaluation_data/'
                             'Ecolight_parameter_combinations_evaluate.csv')
ecolight_input = ecolight_input.drop(columns=["water"])  # Remove the "water" column.
print(ecolight_input)

"""STEP 2. Define theta and x."""

# Define theta and x.
# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in ecolight_input["phy"]]
samples_cdom = [i+constant for i in ecolight_input["cdom"]]
samples_nap = [i+constant for i in ecolight_input["spm"]]
samples_wind = ecolight_input["wind"]
samples_depth = ecolight_input["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_phy = [round(item, 3) for item in samples_phy]
samples_cdom = np.log(samples_cdom)
samples_cdom = [round(item, 3) for item in samples_cdom]
samples_nap = np.log(samples_nap)
samples_nap = [round(item, 3) for item in samples_nap]
samples_wind = np.log(samples_wind)
samples_wind = [round(item, 3) for item in samples_wind]

# Save the transformed data in a dataframe
transformed_dictionary = {"phy": samples_phy, "cdom": samples_cdom, "spm": samples_nap, "wind": samples_wind,
                          "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)
print("Transformed theta: ", transformed_theta)

# Define thetas as a tensor
param_list = []
for param in range(len(transformed_theta["phy"])):
    param_tensor = transformed_theta.iloc[param]
    param_list.append(param_tensor)

thetas = torch.tensor(param_list)
print("Thetas: ", thetas)
print("Number of parameter sets: ", len(thetas))
print("Shape of thetas: ", thetas.shape)

# Define xs as a tensor
refl_list = []
for refl in range(len(transformed_theta["phy"])):
    refl_value = simulated_reflectance.iloc[refl]
    refl_list.append(refl_value)

xs = torch.tensor(refl_list)
print("X: ", xs)
print("Number of reflectance sets: ", len(xs))
print("Shape of X: ", xs.shape)

"""STEP 3. Load the posterior."""

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/1000SNR/Loaded_posteriors/"
          "loaded_posterior4.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 3. Run SBC."""

# For each inference, draw 1000 posterior samples.
num_posterior_samples = 1000
ranks, dap_samples = run_sbc(
    thetas, xs, loaded_posterior, num_posterior_samples=num_posterior_samples)

# Calculate and print stats.
check_stats = check_sbc(
    ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples)
print(
    f"""kolmogorov-smirnov p-values \n
    check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()}""")
print(
    f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()}")
print(f"- c2st accuracies check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()}")

f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    plot_type="hist",
    parameter_labels=["Phytoplankton", "CDOM", "NAP", "Wind", "Depth"],
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

f, ax = sbc_rank_plot(ranks=ranks, num_posterior_samples=num_posterior_samples,
                      plot_type="cdf", parameter_labels=["Phytoplankton", "CDOM", "NAP", "Wind", "Depth"])
plt.show()
