# Import libraries
import pandas as pd
from sbi.analysis import pairplot, check_sbc, run_sbc
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sbi.analysis.plot import sbc_rank_plot

"""STEP 1."""

# Read the csv file containing the simulated reflectance data sbi/diagnostics/sbc.py
simulated_reflectance = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/' 
                                    'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                                    'simulated_reflectance_with_noise_0025.csv')
print(simulated_reflectance)

# Read the csv file containing the inputs of each of the EcoLight simulation runs
ecolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/'
                             'Methods/Methods_Ecolight/Jan2024_lognormal_priors/'
                             'Ecolight_parameter_combinations.csv')
ecolight_input = ecolight_input.drop(columns=["water"])  # Remove the "water" column.
print(ecolight_input)

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
transformed_dictionary = {"phy": samples_phy, "cdom": samples_cdom, "spm": samples_nap,
                          "wind": samples_wind, "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)
print("Transformed theta: ", transformed_theta)

param_list = []
for param in range(1000):
    param_tensor = transformed_theta.iloc[param]
    param_list.append(param_tensor)

thetas = torch.tensor(param_list)
print(thetas)
print(len(thetas))

refl_list = []
for refl in range(1000):
    refl_value = simulated_reflectance.iloc[refl]
    refl_list.append(refl_value)

xs = torch.tensor(refl_list)
print(xs)
print(len(xs))

"""STEP 2. Load the posterior."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Jan2024_lognormal_priors/noise_0025/loaded_posteriors/loaded_posterior1.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

"""STEP 3. Run SBC."""

# run SBC: for each inference we draw 1000 posterior samples.
num_posterior_samples = 1_000
ranks, dap_samples = run_sbc(
    thetas, xs, loaded_posterior, num_posterior_samples=num_posterior_samples)

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
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

f, ax = sbc_rank_plot(ranks, 1_000, plot_type="cdf")
plt.show()
