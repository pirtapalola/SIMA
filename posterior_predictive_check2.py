from sbi.analysis import pairplot
import pandas as pd
import matplotlib.pyplot as plt
import torch

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

# Plot the pairplot
_ = pairplot(
    samples=x_pp,
    points=x_o,
    limits=torch.tensor([[-2.0, 5.0]] * 61),  # Adjust limits based on your data if needed
    points_colors=["red"],
    figsize=(8, 8),
    offdiag="scatter",
    scatter_offdiag=dict(marker=".", s=5),
    points_offdiag=dict(marker="+", markersize=20),
    labels=[rf"$x_{d}$" for d in range(61)],  # Assuming 61 wavelengths
)

plt.show()
