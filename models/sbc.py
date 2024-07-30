"""

Simulation-based calibration
STEP 1. Read the simulated dataset.
STEP 2. Define theta and x.
STEP 3. Load the posterior.
STEP 4. Run SBC.

Last updated on 30 July 2024 by Pirta Palola

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

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Evaluation_data/'
                             'Ecolight_parameter_combinations_evaluate.csv')
ecolight_input = ecolight_input.drop(columns=["water"])  # Remove the "water" column.
print(ecolight_input)

"""STEP 2. Define theta and x."""

# Define theta.
samples_phy = ecolight_input["phy"]
samples_cdom = ecolight_input["cdom"]
samples_nap = ecolight_input["spm"]
samples_wind = ecolight_input["wind"]
samples_depth = ecolight_input["depth"]

# Save the data in a dataframe
theta_dictionary = {"phy": samples_phy,
                    "cdom": samples_cdom,
                    "spm": samples_nap,
                    "wind": samples_wind,
                    "depth": samples_depth}

theta_dataframe = pd.DataFrame(data=theta_dictionary)
print("Theta: ", theta_dataframe)

# Create a list containing theta
param_list = []
for param in range(len(theta_dataframe["phy"])):
    param_tensor = theta_dataframe.iloc[param]
    param_list.append(param_tensor)

# Convert the list into a tensor
thetas = torch.tensor(param_list)

# Check that the number of parameter sets and the shape are correct
print("Thetas: ", thetas)
print("Number of parameter sets: ", len(thetas))
print("Shape of thetas: ", thetas.shape)

# Create a list containing x
refl_list = []
for refl in range(len(theta_dataframe["phy"])):
    refl_value = simulated_reflectance.iloc[refl]
    refl_list.append(refl_value)

# Convert the list into a tensor
xs = torch.tensor(refl_list)

# Check that the number of parameter sets and the shape are correct
print("X: ", xs)
print("Number of reflectance sets: ", len(xs))
print("Shape of X: ", xs.shape)

"""STEP 3. Load the posterior."""

# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/1000SNR/Loaded_posteriors/"
          "loaded_posterior14_hyper.pkl", "rb") as handle:
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
    parameter_labels=["Phytoplankton", "CDOM", "Mineral particles", "Wind", "Depth"],
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

f, ax = sbc_rank_plot(ranks=ranks, num_posterior_samples=num_posterior_samples,
                      plot_type="cdf", parameter_labels=["Phytoplankton", "CDOM", "Mineral particles", "Wind", "Depth"])
plt.show()
