"""

Conduct a prior predictive check.

Last updated on 5 February 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
from models.tools import TruncatedLogNormal
import torch
import matplotlib.pyplot as plt

"""STEP 1. Load the observation data."""

# Read the csv file containing the observation data
observation_path = 'C:/Users/pirtapalola/Documents/Methodology/In_situ_data/2022/'

# Read the file containing the observed parameters
obs_parameters = pd.read_csv(observation_path + 'parameters_tetiaroa_2022.csv')

# Create a list of sample IDs
sample_IDs = list(obs_parameters.columns)
print(sample_IDs)

# Sample from the prior distributions
prior_chl = TruncatedLogNormal(0, 5, 5)
samples_chl = prior_chl.sample(torch.Size([1000]))
prior_cdom = TruncatedLogNormal(0, 5, 5)
samples_cdom = prior_cdom.sample(torch.Size([1000]))
prior_spm = TruncatedLogNormal(0, 5, 30)
samples_spm = prior_spm.sample(torch.Size([1000]))
prior_wind = torch.distributions.LogNormal(1.85, 0.33)
samples_wind = prior_wind.sample(torch.Size([1000]))
prior_depth = torch.distributions.uniform.Uniform(0.1, 10.0)
samples_depth = prior_depth.sample(torch.Size([1000]))

# Store the samples in a dataframe
prior_df = pd.DataFrame({"chl": samples_chl, "cdom": samples_cdom, "spm": samples_spm,
                         "wind": samples_wind, "depth": samples_depth})

# Create an empty list
prior_list = []

# Iterate over each row
for index, rows in prior_df.iterrows():
    # Create list for the current row
    my_list = [rows.chl, rows.cdom, rows.spm, rows.wind, rows.depth]
    # append the list to the final list
    prior_list.append(my_list)

# Print the list
print(len(prior_list))

prior_samples = torch.tensor(prior_list)

# Conduct the prior predictive check
_ = pairplot(
    samples=prior_samples,
    points=obs_parameters["RIM04"],
    limits=[[0, 5], [0, 3], [0, 30], [0, 20], [0, 10]],
    points_colors=["red", "red", "red"],
    figsize=(8, 8),
    offdiag="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=20)
)

plt.show()
