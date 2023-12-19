"""

Visualise the prior distributions:
 -Plot the 3000 samples drawn from each prior distribution to a histogram.
 -Add the corresponding prior distribution as a curve.

Last updated on 19 December 2023 by Pirta Palola

"""

# Import libraries
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Define the priors
prior1 = torch.distributions.LogNormal(0.1, 1.7)
prior2 = torch.distributions.LogNormal(0.05, 1.7)
prior3 = torch.distributions.LogNormal(0.4, 1.1)
prior4 = torch.distributions.LogNormal(1.85, 0.33)

# Import the samples drawn from each prior (3000 randomly drawn samples)
prior_samples = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                            "Methods_Ecolight/priors/priors_summary.csv")
samples_phy = prior_samples["Phytoplankton"].to_numpy()
samples_cdom = prior_samples["CDOM"].to_numpy()
samples_nap = prior_samples["NAP"].to_numpy()
samples_wind = prior_samples["Wind"].to_numpy()

# Create a figure with four subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot the PDF and histogram of prior1 in the first subplot
x = torch.linspace(prior1.icdf(torch.tensor(0.00001)).item(),
                   prior1.icdf(torch.tensor(0.999)).item(), 5000)
p = prior1.log_prob(x).exp().numpy()
axes[0, 0].plot(x, p, '#184e77', linewidth=2, label='$\mu=0.1$, $\sigma=1.7$')
axes[0, 0].hist(samples_phy, bins=1000, alpha=0.5, color='#76c893', density=True, label='Samples (n = 3000)')
axes[0, 0].legend()
axes[0, 0].set_title('Phytoplankton')
axes[0, 0].set_xlim(left=0, right=10)  # Set x-axis limits
axes[0, 0].set_ylim(bottom=0, top=1.0)  # Set y-axis limits
axes[0, 0].set_xlabel('Concentration (mg/$\mathregular{m^3}$)')
axes[0, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior2 in the second subplot
x = torch.linspace(prior2.icdf(torch.tensor(0.00001)).item(),
                   prior2.icdf(torch.tensor(0.999)).item(), 5000)
p = prior2.log_prob(x).exp().numpy()
axes[0, 1].plot(x, p, '#184e77', linewidth=2, label='$\mu=0.05$, $\sigma=1.7$')
axes[0, 1].hist(samples_cdom, bins=1000, alpha=0.5, color='#76c893', density=True, label='Samples (n = 3000)')
axes[0, 1].legend()
axes[0, 1].set_title('CDOM')
axes[0, 1].set_xlim(left=0, right=5)  # Set x-axis limits
axes[0, 1].set_ylim(bottom=0, top=1.0)  # Set y-axis limits
axes[0, 1].set_xlabel('Absorption $\mathregular{m^-1}$ at 440 nm)')
axes[0, 1].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior3 in the third subplot
x = torch.linspace(prior3.icdf(torch.tensor(0.00001)).item(),
                   prior3.icdf(torch.tensor(0.999)).item(), 5000)
p = prior3.log_prob(x).exp().numpy()
axes[1, 0].plot(x, p, '#184e77', linewidth=2, label='$\mu=0.4$, $\sigma=1.1$')
axes[1, 0].hist(samples_nap, bins=1000, alpha=0.5, color='#76c893', density=True, label='Samples (n = 3000)')
axes[1, 0].legend()
axes[1, 0].set_title('NAP')
axes[1, 0].set_xlim(left=0, right=50)  # Set x-axis limits
axes[1, 0].set_ylim(bottom=0, top=1.0)  # Set y-axis limits
axes[1, 0].set_xlabel('Concentration (g/$\mathregular{m^3}$)')
axes[1, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior4 in the fourth subplot
x = torch.linspace(prior4.icdf(torch.tensor(0.00001)).item(),
                   prior4.icdf(torch.tensor(0.999)).item(), 5000)
p = prior4.log_prob(x).exp().numpy()
axes[1, 1].plot(x, p, '#184e77', linewidth=2, label='$\mu=1.85$, $\sigma=0.33$')
axes[1, 1].hist(samples_wind, bins=1000, alpha=0.5, color='#76c893', density=True, label='Samples (n = 3000)')
axes[1, 1].legend()
axes[1, 1].set_title('Wind')
axes[1, 1].set_xlim(left=0, right=20)  # Set x-axis limits
axes[1, 1].set_ylim(bottom=0, top=1.0)  # Set y-axis limits
axes[1, 1].set_xlabel('Wind speed (m/s)')
axes[1, 1].set_ylabel('Probability Density')

# Adjust layout to prevent clipping of titles
fig.tight_layout()

# Show the plot
plt.show()

