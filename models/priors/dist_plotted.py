"""
PRIORS II: Visualise the prior distributions.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Visualise the prior distributions:
 -Plot the 30,000 samples drawn from each prior distribution to a histogram.
 -Add the corresponding PDF as a curve.

Last updated on 26 August 2024

"""

# Import libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt
from models.tools import TruncatedLogNormal
import numpy as np

# Create an instance of TruncatedLogNormal
prior1 = TruncatedLogNormal(loc=0, scale=5, upper_bound=7)
prior2 = TruncatedLogNormal(loc=0, scale=5, upper_bound=2.5)
prior3 = TruncatedLogNormal(loc=0, scale=5, upper_bound=30)
prior4 = torch.distributions.LogNormal(1.85, 0.33)
prior5 = torch.distributions.uniform.Uniform(low=0.1, high=20.0)

# Plot the PDF for visualization
x_values1 = np.linspace(0.01, 7, 1000)
x_values2 = np.linspace(0.01, 2.5, 1000)
x_values3 = np.linspace(0.01, 30, 1000)
x_values4 = np.linspace(0.1, 20, 1000)

pdf_values1 = prior1.pdf(torch.tensor(x_values1))
pdf_values2 = prior2.pdf(torch.tensor(x_values2))
pdf_values3 = prior3.pdf(torch.tensor(x_values3))
pdf_values4 = prior4.log_prob(torch.tensor(x_values4)).exp().numpy()

# Import the samples drawn from each prior
prior_samples = pd.read_csv("data/simulation_setup/priors_summary.csv")
samples_phy = prior_samples["Phytoplankton"].to_numpy()
samples_cdom = prior_samples["CDOM"].to_numpy()
samples_nap = prior_samples["Min"].to_numpy()
samples_wind = prior_samples["Wind"].to_numpy()
samples_depth = prior_samples["Depth"].to_numpy()


# Create a figure with four subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

# Plot the PDF and histogram of prior1 in the first subplot
axes[0, 0].plot(x_values1, pdf_values1, '#184e77', linewidth=2, label='$\mu=0$, $\sigma=5$')
axes[0, 0].hist(samples_phy, bins=1000, alpha=0.5, color='#76c893', density=True)
axes[0, 0].legend()
# axes[0, 0].set_title('Phytoplankton')
axes[0, 0].set_xlim(left=0, right=7)  # Set x-axis limits
axes[0, 0].set_ylim(bottom=0, top=5.0)  # Set y-axis limits
axes[0, 0].set_xlabel('Phytoplankton concentration (mg/$\mathregular{m^3}$)')
axes[0, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior2 in the second subplot
axes[0, 1].plot(x_values2, pdf_values2, '#184e77', linewidth=2, label='$\mu=0$, $\sigma=5$')
axes[0, 1].hist(samples_cdom, bins=1000, alpha=0.5, color='#76c893', density=True)
axes[0, 1].legend()
# axes[0, 1].set_title('CDOM')
axes[0, 1].set_xlim(left=0, right=2.5)  # Set x-axis limits
axes[0, 1].set_ylim(bottom=0, top=5.0)  # Set y-axis limits
axes[0, 1].set_xlabel('CDOM absorption ($\mathregular{m^-1}$ at 440 nm)')
axes[0, 1].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior3 in the third subplot
axes[1, 0].plot(x_values3, pdf_values3, '#184e77', linewidth=2, label='$\mu=0$, $\sigma=5$')
axes[1, 0].hist(samples_nap, bins=1000, alpha=0.5, color='#76c893', density=True)
axes[1, 0].legend()
# axes[1, 0].set_title('SPM')
axes[1, 0].set_xlim(left=0, right=30)  # Set x-axis limits
axes[1, 0].set_ylim(bottom=0, top=5.0)  # Set y-axis limits
axes[1, 0].set_xlabel('Mineral particle concentration (g/$\mathregular{m^3}$)')
axes[1, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior4 in the fourth subplot
x = torch.linspace(prior4.icdf(torch.tensor(0.00001)).item(),
                   prior4.icdf(torch.tensor(0.999)).item(), 5000)
p = prior4.log_prob(x).exp().numpy()
axes[1, 1].plot(x, p, '#184e77', linewidth=2, label='$\mu=1.85$, $\sigma=0.33$')
axes[1, 1].hist(samples_wind, bins=1000, alpha=0.5, color='#76c893', density=True)
axes[1, 1].legend()
# axes[1, 1].set_title('Wind')
axes[1, 1].set_xlim(left=0, right=20)  # Set x-axis limits
axes[1, 1].set_ylim(bottom=0, top=0.5)  # Set y-axis limits
axes[1, 1].set_xlabel('Wind speed (m/s)')
axes[1, 1].set_ylabel('Probability Density')


# Plot the PDF and histogram of prior5 in the fifth subplot
x = torch.linspace(prior5.icdf(torch.tensor(0.00001)).item(),
                   prior5.icdf(torch.tensor(0.999)).item(), 5000)
p = prior5.log_prob(x).exp().numpy()
axes[2, 0].plot(x, p, '#184e77', linewidth=2, label='$min=0.10$, $max=20.00$')
axes[2, 0].hist(samples_depth, bins=1000, alpha=0.5, color='#76c893', density=True)
axes[2, 0].legend()
# axes[2, 0].set_title('Depth')
axes[2, 0].set_xlim(left=0, right=20)  # Set x-axis limits
axes[2, 0].set_ylim(bottom=0, top=0.5)  # Set y-axis limits
axes[2, 0].set_xlabel('Depth (m)')
axes[2, 0].set_ylabel('Probability Density')

# Adjust layout to prevent clipping of titles
fig.tight_layout()

# Show the plot
plt.show()
