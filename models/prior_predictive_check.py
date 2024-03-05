"""

Conduct a prior predictive check.

Last updated on 5 March 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
from sbi.analysis import pairplot
import torch
import matplotlib.pyplot as plt

"""STEP 2. Convert the data into tensors."""


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
observation_parameters = dataframe_to_tensor(prior_param)


"""STEP 3. Conduct the prior predictive check."""

# Plot
_ = pairplot(
    samples=prior_samples,
    points=prior_param,
    limits=[[0, 7], [0, 2.5], [0, 30], [0, 20], [0, 20]],
    points_colors=["red", "red", "red"],
    figsize=(8, 8),
    offdiag="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=5)
)

plt.show()
