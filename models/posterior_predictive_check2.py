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

# Plot posterior predictives with mean +- std
# Often +-2std is used, so change factor to 2


def plot_std(x, y, alpha_fill=0.3, factor=1, **kwargs):
    """plots the mean +-std of y

    Args:
        x (array): (l))
        y (array): (n,l)
        factor (float): factor to multiply std with
    """
    mean = np.mean(y, 0)
    std = np.std(y, 0) * factor

    (base_line,) = plt.plot(x, mean, **kwargs)
    kwargs["label"] = None
    kwargs["alpha"] = alpha_fill
    kwargs["facecolor"] = base_line.get_color()
    kwargs["edgecolor"] = None  # "green"
    plt.fill_between(x, mean - std, mean + std, **kwargs)


plt.show()
