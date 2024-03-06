"""

Prior predictive check.

Last updated on 6 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
from models.tools import TruncatedLogNormal
import torch
import matplotlib.pyplot as plt

"""STEP 1. Load the observation data."""

observation_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
                   'Jan2024_lognormal_priors/field_data/'
obs_parameters = pd.read_csv(observation_path + 'parameters.csv')
print(obs_parameters)
obs_parameters = obs_parameters.drop(columns="unique_ID")

"""STEP 2. Sample from the prior distributions."""

prior_chl = TruncatedLogNormal(0, 5, 7)
samples_chl = prior_chl.sample(torch.Size([30000]))
prior_cdom = TruncatedLogNormal(0, 5, 2.5)
samples_cdom = prior_cdom.sample(torch.Size([30000]))
prior_spm = TruncatedLogNormal(0, 5, 30)
samples_spm = prior_spm.sample(torch.Size([30000]))
prior_wind = torch.distributions.LogNormal(1.85, 0.33)
samples_wind = prior_wind.sample(torch.Size([30000]))
prior_depth = torch.distributions.uniform.Uniform(0.1, 20.0)
samples_depth = prior_depth.sample(torch.Size([30000]))

# Store the samples in a dataframe
prior_df = pd.DataFrame({"chl": samples_chl, "cdom": samples_cdom, "spm": samples_spm,
                         "wind": samples_wind, "depth": samples_depth})


# Convert the dataframe into a tensor
def dataframe_to_tensor(dataframe):
    # Create an empty list
    empty_list = []
    # Iterate over each row
    for index, rows in dataframe.iterrows():
        # Create a list for the current row
        my_list = [rows.chl, rows.cdom, rows.spm, rows.wind, rows.depth]
        # Append the list to the empty_list
        empty_list.append(my_list)
    # Print the length of the list
    print(len(empty_list))
    # Create a tensor
    data_torch = torch.tensor(empty_list)
    return data_torch


# Apply the function
prior_samples = dataframe_to_tensor(prior_df)
observation_parameters = dataframe_to_tensor(obs_parameters)


# Conduct the prior predictive check
_ = pairplot(
    samples=prior_samples,
    points=observation_parameters,
    limits=[[0, 2], [0, 1], [0, 10], [0, 10], [0, 20]],
    points_colors=["red"],
    figsize=(8, 8),
    labels=["Phytoplankon", "CDOM", "NAP", "Wind speed", "Depth"],
    offdiag="scatter",
    scatter_offdiag=dict(s=5),
    points_offdiag=dict(markersize=5)
)
plt.show()
