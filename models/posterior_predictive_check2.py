from sbi.analysis import pairplot
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sbi import analysis

# Read the csv file containing the simulated reflectance data
simulated_reflectance = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/'
                                    'simulated_reflectance_1000SNR_noise_sbc.csv')

# Pick one spectrum as an observation
observed = simulated_reflectance.iloc[0]

# Read the csv file containing the simulated reflectance data
simulated_reflectance_ppc = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/PPC/'
                                        'simulated_reflectance_no_noise.csv')

# x_pp = simulated_reflectance.iloc[0]

# Define your observed spectrum x_o (shape: [61])
x_o = torch.tensor([observed])

# Define your simulated spectra from the posterior predictive distribution x_pp (shape: [1000, 61])
x_pp = torch.tensor(simulated_reflectance_ppc.values)

# Plot

# plot posterior/prior samples
fig, ax = analysis.pairplot(
    x_pp,
    # limits=plotting_limits,
    figsize=(4, 4),
    points=x_o,
    # labels=your_labels,
    points_colors=["red"],  # colors for the points_1 and points_2
    upper="contour",  # 'scatter' hist, scatter, contour, cond, None
    **{  #'title': f'Paramter prediction given model {model_sample.numpy()}', # title
        "kde_offdiag": {"bins": 50},  # bins for kde on the off-diagonal
        "points_offdiag": {"markersize": 3, "marker": "x"},
        "contour_offdiag": {"levels": [0.023, 0.5, 0.977]},
        "points_diag": {"ls": "-", "lw": 1, "alpha": 1},
    }
)
